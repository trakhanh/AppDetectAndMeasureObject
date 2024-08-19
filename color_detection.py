import cv2
import numpy as np
import sys
import os
import PySimpleGUI as sg
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort


def color_detection(frame, lower_bound, upper_bound):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    return mask

def is_within_roi(x, y, w, h, roi_coords):
    x1, y1, x2, y2 = roi_coords
    return (x >= x1 and y >= y1 and (x + w) <= x2 and (y + h) <= y2)

def color_detection_main(video_source=None, use_webcam=False):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        sg.popup_error("Error: Cannot open video source.")
        return

    # Create window layout
    layout = [
        [sg.Image(filename='', key='image')],
        [sg.Image(filename='', key='mask_image')],
        [sg.Button('Tạm dừng/Chạy', key='Pause'), sg.Button('Quay lại', key='Back'), sg.Button('Thoát', key='Exit')]
    ]
    window = sg.Window('Thuật toán phát hiện màu', layout, location=(100, 100), finalize=True)

    # Khởi tạo SORT tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.1)
    
    # Định nghĩa ROI
    roi_coords = (150, 35, 520, 300)  # (x1, y1, x2, y2)

    # Biến để đếm tổng số đối tượng trong ROI
    total_objects_in_roi = 0
    total_white_objects = 0
    total_orange_objects = 0

    # Tập hợp các ID đã đếm
    counted_ids = set()
    paused = False

    while True:
        event, values = window.read(timeout=20)
        if event in ('Exit', sg.WIN_CLOSED):
            break
        if event == 'Pause':
            paused = not paused
        if event == 'Back':
            window.close()
            return

        if not paused:
            ret, frame = cap.read()
            if not ret:
                break  # Kết thúc khi đọc hết video

            frame = cv2.resize(frame, (640, 360))

            # Áp dụng Gaussian Blur để giảm nhiễu
            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

            # Thiết lập các ngưỡng cho màu sắc
            lower_white = np.array([0, 0, 168])
            upper_white = np.array([172, 111, 255])
            lower_orange = np.array([1, 40, 185])
            upper_orange = np.array([25, 255, 255])

            # Tạo mặt nạ từ ngưỡng cho màu cam và màu trắng
            mask_white = color_detection(blurred_frame, lower_white, upper_white)
            mask_orange = color_detection(blurred_frame, lower_orange, upper_orange)

            # Biến đổi hình thái để làm sạch mặt nạ cho màu cam và màu trắng
            kernel = np.ones((5, 5), np.uint8)
            mask_white_clean = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask_white_clean = cv2.morphologyEx(mask_white_clean, cv2.MORPH_OPEN, kernel, iterations=3)
            mask_orange_clean = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask_orange_clean = cv2.morphologyEx(mask_orange_clean, cv2.MORPH_OPEN, kernel, iterations=3)

            # Kết hợp các mặt nạ cho màu cam và màu trắng
            final_mask = cv2.bitwise_or(mask_white_clean, mask_orange_clean)

            # Tìm contours và vẽ chúng cho màu trắng
            contours_white, _ = cv2.findContours(mask_white_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_orange, _ = cv2.findContours(mask_orange_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Chuẩn bị danh sách phát hiện cho SORT
            detections = []

            # Vẽ contours cho màu trắng
            for contour in contours_white:
                area = cv2.contourArea(contour)
                if area > 8000 and area < 70000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    if 0.5 < aspect_ratio < 2.0:
                        detections.append([x, y, x + w, y + h, 1])  # Thêm vào danh sách phát hiện (dạng [x1, y1, x2, y2, score])

            # Vẽ contours cho màu cam
            for contour in contours_orange:
                area = cv2.contourArea(contour)
                if area > 8000 and area < 70000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    if 0.5 < aspect_ratio < 2.0:
                        detections.append([x, y, x + w, y + h, 1])  # Thêm vào danh sách phát hiện (dạng [x1, y1, x2, y2, score])

            # Chuyển đổi danh sách phát hiện thành numpy array
            if len(detections) > 0:
                detections = np.array(detections)
            else:
                detections = np.empty((0, 5))

            # Cập nhật tracker với các phát hiện mới
            tracks = tracker.update(detections)

            # Đếm số lượng đối tượng màu trắng và màu cam
            current_roi_object_count = 0

            # Vẽ các khung chứa đối tượng và ID của đối tượng từ SORT tracker
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Kiểm tra màu sắc của đối tượng
                color = 'Unknown'
                if np.any(mask_white_clean[y1:y2, x1:x2]):
                    color = 'White'
                elif np.any(mask_orange_clean[y1:y2, x1:x2]):
                    color = 'Orange'

                cv2.putText(frame, f'Color: {color}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if is_within_roi(x1, y1, x2 - x1, y2 - y1, roi_coords):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        current_roi_object_count += 1
                        if color == 'White':
                            total_white_objects += 1
                        elif color == 'Orange':
                            total_orange_objects += 1

            total_objects_in_roi += current_roi_object_count

            # Vẽ khung ROI
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

            # Hiển thị số lượng đối tượng
            # Đặt văn bản ở vị trí phía dưới cùng bên trái khung hình
            height, width, _ = frame.shape
            cv2.putText(frame, f'Total ROI Count: {total_objects_in_roi}', (5, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f'Total White in ROI: {total_white_objects}', (5, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f'Total Orange in ROI: {total_orange_objects}', (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            # Hiển thị kết quả trong cửa sổ GUI
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)

            # Hiển thị mask trong cửa sổ GUI
            mask_imgbytes = cv2.imencode('.png', final_mask)[1].tobytes()
            window['mask_image'].update(data=mask_imgbytes)

    cap.release()
    cv2.destroyAllWindows()
    window.close()
