import cv2
import numpy as np
import sys
import os
import threading
import subprocess
from ultralytics import YOLO
import PySimpleGUI as sg
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort
from pyui2 import GUI as g
from color_detection import color_detection_main
from background_subtraction import background_subtraction_main

class Demo():
    sg.theme('DarkTeal9')
    pixel_per_metric = 16.79  # Giá trị mặc định, có thể điều chỉnh
    #pixel_per_metric = 63.91  # Giá trị mặc định, có thể điều chỉnh
    colors = [
        (255, 0, 0),   # Màu cho class 0
        (0, 255, 0),   # Màu cho class 1
        (0, 0, 255),   # Màu cho class 2
        (0, 255, 255), # Màu cho class 3
        (255, 255, 255), # Màu cho class 4
        # Thêm màu cho các class khác nếu có nhiều class hơn
    ]

    def create_tracker():
        tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.1)
        print(f"New Sort object created at: {id(tracker)}")
        return tracker
    
    def reset_state(num_classes):
        # Khởi tạo lại đối tượng Sort
        global mot_tracker, class_dict, object_appearances, roi_counts, object_ids_in_roi
        mot_tracker = Demo.create_tracker()
        class_dict = {}
        object_appearances = {}
        roi_counts = {i: 0 for i in range(num_classes)}
        object_ids_in_roi = set()
        
    def load_model(model_path):
        model = YOLO(model_path)
        class_names = model.names
        num_classes = len(class_names)
        return model, class_names, num_classes
    
    def calculate_ppm(image_path, metric, dimension):
        global model
        if model is None:
            sg.popup_error('Model chưa được chọn. Vui lòng chọn model trước.')
            return None
        
        # Đọc ảnh chứa đối tượng tham chiếu
        image = cv2.imread(image_path)

        # Phát hiện đối tượng tham chiếu trong ảnh bằng mô hình đã chọn
        results = model(image, conf=0.85)

        max_confidence = 0
        best_pixel_size = None

        for result in results:
            for bbox in result.boxes:
                x_center, y_center, w, h = bbox.xywh[0]
                w = int(w)
                h = int(h)
                confidence = bbox.conf[0]  # Xác suất phát hiện đối tượng

                # Kiểm tra và cập nhật nếu confidence cao hơn
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_pixel_size = w if dimension == 'Width' else h

        if best_pixel_size is not None:
            ppm_value = best_pixel_size / metric
            Demo.pixel_per_metric = ppm_value  # Cập nhật giá trị PPM trong đối tượng Demo
            return ppm_value
        return None

    
    def get_detections(frame, model, conf_thresh):
        results = model(frame)[0]
        detections = []
        for result in results.boxes:
            x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
            score = result.conf[0].cpu().numpy()
            cls = int(result.cls[0].cpu().numpy())
            if score >= conf_thresh:  # Lọc theo ngưỡng confident
                detections.append([x1, y1, x2, y2, score, cls])
        detections = np.array(detections)
        detections = Demo.non_max_suppression_fast(detections, 0.3)
        return detections

    def non_max_suppression_fast(boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    
    def draw_boxes(frame, tracks, class_dict, class_names, pixel_per_metric, original_width, original_height, resized_width, resized_height, roi_coords):
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cls = class_dict.get(track_id, -1)
            class_name = class_names.get(cls, "Unknown")

            if cls == 0:
                color = (255, 0, 0)
            elif cls == 1:
                color = (0, 255, 0)
            elif cls == 2:
                color = (0, 0, 255)
            elif cls == 3:
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)

            # Chuyển đổi tọa độ về kích thước gốc
            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)

            # Kiểm tra xem bounding box có nằm trong ROI không
            if Demo.is_in_roi([x1_orig, y1_orig, x2_orig, y2_orig], roi_coords):
                # Tính toán kích thước vật thể
                width = (x2_orig - x1_orig) / pixel_per_metric
                height = (y2_orig - y1_orig) / pixel_per_metric
                label = f'ID: {int(track_id)} Class: {class_name} W: {width:.2f}cm H: {height:.2f}cm'
            else:
                label = f'ID: {int(track_id)} Class: {class_name}'

            # Chuyển đổi tọa độ về kích thước đã resize để vẽ
            x1 = int(x1_orig / scale_x)
            y1 = int(y1_orig / scale_y)
            x2 = int(x2_orig / scale_x)
            y2 = int(y2_orig / scale_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    
    def is_in_roi(bbox, roi):
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi
        in_roi = x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2
        print(f"Checking if bbox {bbox} is in ROI {roi}: {in_roi}")
        return in_roi

    
    def draw_roi(frame, roi_coords):
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    
    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_t, y1_t, x2_t, y2_t = box2

        xi1 = max(x1, x1_t)
        yi1 = max(y1, y1_t)
        xi2 = min(x2, x2_t)
        yi2 = min(y2, y2_t)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_t - x1_t) * (y2_t - y1_t)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        print(f"xi1: {xi1}, yi1: {yi1}, xi2: {xi2}, yi2: {yi2}")
        print(f"inter_area: {inter_area}, box1_area: {box1_area}, box2_area: {box2_area}, union_area: {union_area}, iou: {iou}")
        return iou

    
    def update_frame(frame, window):
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)

    
    def main_track(video_source, use_webcam, model, class_names, num_classes):
        Demo.reset_state(num_classes)
    
        # roi_coords = (880, 185, 1490, 1050)  # Default ROI coordinates
        roi_coords = (150, 150, 305, 400)  # Default ROI coordinates
        if use_webcam:
            cap = cv2.VideoCapture(0)
            # Thiết lập độ phân giải camera
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2688)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1520)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            sg.popup_error("Error: Cannot open video source.")
            return

        frame_skip = 1
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        paused = False
        # Cập nhật giá trị pixel_per_metric hiện tại
        window = g.create_object_tracking_window(roi_coords)
        window['pixel_per_metric_value'].update(f'{Demo.pixel_per_metric:.2f}')
        
        while cap.isOpened():
            event, values = window.read(timeout=20)
            if event in ('Exit', sg.WIN_CLOSED):
                break
            if event == 'Pause':
                paused = not paused
            if event == 'Minimize':
                window.minimize()
            if event == 'Back':
                window.close()
                return
            if event == 'UpdateROI':
                try:
                    x1 = int(values['x1'])
                    y1 = int(values['y1'])
                    x2 = int(values['x2'])
                    y2 = int(values['y2'])
                    roi_coords = (x1, y1, x2, y2)
                    window['current_roi'].update(f'({x1}, {y1}, {x2}, {y2})')
                    print(f"Updated ROI coordinates: {roi_coords}")
                except ValueError:
                    sg.popup_error('Tọa độ không hợp lệ! Vui lòng nhập lại.')
            if event == 'UpdatePPM':
                try:
                    Demo.pixel_per_metric = float(values['pixel_per_metric'])
                    window['pixel_per_metric_value'].update(f'{Demo.pixel_per_metric:.2f}')
                    print(f"Cập nhật pixel_per_metric: {Demo.pixel_per_metric}")
                except ValueError:
                     sg.popup_error("Giá trị pixel_per_metric không hợp lệ! Vui lòng nhập một số.")
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                original_height, original_width = frame.shape[:2]
                resized_frame = cv2.resize(frame, (800, 600))
                resized_height, resized_width, channels = resized_frame.shape

                # Chuyển đổi tọa độ ROI dựa trên tỉ lệ thay đổi kích thước
                converted_roi_coords = (
                    int(roi_coords[0] * resized_width / original_width),
                    int(roi_coords[1] * resized_height / original_height),
                    int(roi_coords[2] * resized_width / original_width),
                    int(roi_coords[3] * resized_height / original_height)
                )
                print(f"Converted ROI coordinates: {converted_roi_coords}")
                conf_thresh = values['conf_thresh']
                detections = Demo.get_detections(resized_frame, model, conf_thresh)
                print(f"Detections: {detections}")

                if len(detections) > 0:
                    tracks = mot_tracker.update(detections[:, :5])

                    for track in tracks:
                        track_id = int(track[4])
                        if track_id not in object_appearances:
                            object_appearances[track_id] = 0
                        object_appearances[track_id] += 1

                        if object_appearances[track_id] > 3:  # Đợi 3 khung hình trước khi cập nhật class
                            for detection in detections:
                                x1, y1, x2, y2, score, cls = detection
                                iou = Demo.compute_iou([x1, y1, x2, y2], track[:4])
                                if iou > 0.3:
                                    class_dict[track_id] = cls
                                    break

                    current_object_ids_in_roi = set()

                    for track in tracks:
                        x1, y1, x2, y2, track_id = track
                        cls = class_dict.get(track_id, -1)
                        if cls != -1 and Demo.is_in_roi([x1, y1, x2, y2], converted_roi_coords):
                            current_object_ids_in_roi.add(track_id)
                            if track_id not in object_ids_in_roi:
                                print(f"Object {track_id} of class {cls} entered ROI at coordinates {converted_roi_coords}")
                                roi_counts[cls] += 1

                    object_ids_in_roi = current_object_ids_in_roi

                    resized_frame = Demo.draw_boxes(resized_frame, tracks, class_dict, class_names, Demo.pixel_per_metric, original_width, original_height, resized_width, resized_height, roi_coords)

                if converted_roi_coords:
                    resized_frame = Demo.draw_roi(resized_frame, converted_roi_coords)

                # Vẽ các số đếm class trên khung hình
                for i in range(num_classes):
                    color = Demo.colors[i % len(Demo.colors)]  # Sử dụng màu tương ứng cho class
                    cv2.putText(resized_frame, f'{class_names[i]}: {roi_counts[i]}', (10, 30 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                Demo.update_frame(resized_frame, window)

        cap.release()
        cv2.destroyAllWindows()
        window.close()
        print("Video processing completed.")

    
    def main_capture(video_source, use_webcam):
        if use_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            sg.popup_error("Error: Cannot open video source.")
            return

        frame_count = 0
        paused = False

        window = g.create_capture_window()

        while cap.isOpened():
            event, values = window.read(timeout=20)
            if event in ('Exit', sg.WIN_CLOSED):
                break
            if event == 'Pause':
                paused = not paused
            if event == 'Capture':
                img_name = f"capture_{frame_count}.png"
                class_name = sg.popup_get_text('Nhập tên lớp cho đối tượng:')
                if class_name:
                    img_dir = os.path.join('captured_images', class_name)
                    os.makedirs(img_dir, exist_ok=True)
                    img_path = os.path.join(img_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    print(f"{img_path} saved!")
            if event == 'Minimize':
                window.minimize()
            if event == 'Back':
                window.close()
                return

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                frame = cv2.resize(frame, (800, 600))

                Demo.update_frame(frame, window)

        cap.release()
        cv2.destroyAllWindows()
        window.close()
        print("Video capturing completed.")

    def run_training_script():
            try:
                # Chạy file Python
                subprocess.run(['python', 'testtrain.py'], check=True)
            except subprocess.CalledProcessError as e:
                sg.popup_error(f"Lỗi khi chạy script: {e}")
            except Exception as e:
                sg.popup_error(f"Lỗi không xác định: {e}")

# Khởi tạo các cửa sổ
model_list = ['train3/weights/best.pt', 'runs/detect/yolov8n_new/weights/best.pt', 'runs/detect/BangChuyenCu/weights/best.pt', 'yolov8n.pt', 'yolov8m.pt', 'yolov8x.pt']

model, model_path, class_names, num_classes = g.open_model_selection(model_list)
if model is None:
    sg.popup_error('Bạn phải chọn một model YOLOv8!')
    sys.exit()

roi_counts = {i: 0 for i in range(num_classes)}

window_main = g.create_main_window(model_path)

while True:
    event, values = window_main.read()
    if event == sg.WIN_CLOSED or event == 'Thoát':
        break
    if event == 'Đổi mô hình':
        result = g.open_model_selection(model_list)
        if result:
            model, model_path, class_names, num_classes = result
            window_main['model_text'].update(f'Mô hình hiện tại: {model_path}')
            
    if event == 'Tạo dữ liệu mới':
        add_data_window = g.create_add_data_window()
        while True:
            add_event, add_values = add_data_window.read()
            if add_event == sg.WIN_CLOSED or add_event == 'Quay lại':
                add_data_window.close()
                break
            if add_event == 'Chụp ảnh từ webcam':
                Demo.main_capture(None, True)
            if add_event == 'Chọn video từ thư mục':
                video_source = sg.popup_get_file('Chọn video', file_types=(("Video Files", "*.mp4"),))
                if video_source:
                    Demo.main_capture(video_source, False)
                    
    if event == 'Theo dõi và tính kích thước đối tượng':
        track_object_window = g.create_track_object_window()
        while True:
            track_event, track_values = track_object_window.read()
            if track_event == sg.WIN_CLOSED or track_event == 'Quay lại':
                track_object_window.close()
                break
            if track_event == 'Theo dõi đối tượng từ webcam':
                Demo.main_track(None, True, model, class_names, num_classes)
            if track_event == 'Theo dõi đối tượng từ video':
                video_source = sg.popup_get_file('Chọn video', file_types=(("Video Files", "*.mp4"),))
                if video_source:
                    Demo.main_track(video_source, False, model, class_names, num_classes)
            if 'pixel_per_metric' in track_values:
                try:
                    Demo.pixel_per_metric = float(track_values['pixel_per_metric'])
                    window_main['pixel_per_metric_value'].update(f'{Demo.pixel_per_metric:.2f}')
                    print(f"Cập nhật pixel_per_metric: {Demo.pixel_per_metric}")
                except ValueError:
                    sg.popup_error("Giá trị pixel_per_metric không hợp lệ! Vui lòng nhập một số.")

    if event == 'Tính PPM':
        ppm_window = g.create_calculate_ppm_window()
        while True:
            ppm_event, ppm_values = ppm_window.read()
            if ppm_event == sg.WIN_CLOSED or ppm_event == 'Quay lại':
                ppm_window.close()
                break
            if ppm_event == 'Tính PPM':
                image_path = ppm_values['image_path']
                metric = float(ppm_values['metric'])
                dimension = ppm_values['dimension']
                ppm_value = Demo.calculate_ppm(image_path, metric, dimension)
                if ppm_value:
                    Demo.pixel_per_metric = ppm_value  # Cập nhật giá trị PPM
                    ppm_window['ppm_value'].update(f'{ppm_value:.2f} pixels/cm')
                else:
                    sg.popup_error('Không thể tính toán PPM. Hãy kiểm tra lại dữ liệu.')
            if ppm_event == 'image_path':
                image_path = ppm_values['image_path']
                if image_path:
                    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
                    if not image_path.lower().endswith(valid_extensions):
                        sg.popup_error('Vui lòng chọn một tập tin ảnh hợp lệ (PNG, JPG, JPEG, BMP).')
                        ppm_window['image_path'].update('')
                    else:
                        image = cv2.imread(image_path)
                        image = cv2.resize(image, (400, 300))
                        imgbytes = cv2.imencode('.png', image)[1].tobytes()
                        ppm_window['image'].update(data=imgbytes)
            if ppm_event == 'capture_image':
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        sg.popup_error("Không thể mở webcam!")
                        continue
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        image_path = 'captured_image.png'
                        cv2.imwrite(image_path, frame)
                        ppm_window['image_path'].update(value=image_path)

                        image = cv2.resize(frame, (400, 300))
                        imgbytes = cv2.imencode('.png', image)[1].tobytes()
                        ppm_window['image'].update(data=imgbytes)
                    else:
                        sg.popup_error("Không thể chụp ảnh từ webcam!")

    if event == 'color_detection':
        color_detection_window = g.create_color_detection_window()
        while True:
            color_event, color_values = color_detection_window.read()
            if color_event == sg.WIN_CLOSED or color_event == 'Quay lại':
                color_detection_window.close()
                break
            if color_event == 'Từ webcam':
                color_detection_window.close()
                color_detection_main(use_webcam=True)
            if color_event == 'Từ video':
                video_source = sg.popup_get_file('Chọn video', file_types=(("Video Files", "*.mp4"),))
                if video_source:
                    color_detection_window.close()
                    color_detection_main(video_source, use_webcam=False)
    if event == 'background_subtraction':
        bg_subtraction_window = g.create_background_subtraction_window()
        while True:
            bg_event, bg_values = bg_subtraction_window.read()
            if bg_event == sg.WIN_CLOSED or bg_event == 'Quay lại':
                bg_subtraction_window.close()
                break
            if bg_event == 'Từ webcam':
                bg_subtraction_window.close()
                background_subtraction_main(use_webcam=True)
            if bg_event == 'Từ video':
                video_source = sg.popup_get_file('Chọn video', file_types=(("Video Files", "*.mp4"),))
                if video_source:
                    bg_subtraction_window.close()
                    background_subtraction_main(video_source, use_webcam=False)
    if event == 'Bắt đầu huấn luyện':
        Demo.run_training_script()
        sg.popup('Thoát huấn luyện!')
window_main.close()

