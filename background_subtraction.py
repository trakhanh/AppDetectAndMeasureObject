import cv2
import numpy as np
import PySimpleGUI as sg
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort
# Khởi tạo các bộ trừ nền
KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Hàm lựa chọn bộ trừ nền
def select_subtractor(subtractor_type='KNN'):
    if subtractor_type == 'MOG2':
        return MOG2_subtractor
    return KNN_subtractor

# Hàm làm mượt bounding box
def smooth_bbox(current_bbox, previous_bbox, alpha=0.7):
    """Làm mượt bbox bằng trung bình động"""
    if previous_bbox is None:
        return current_bbox
    return alpha * previous_bbox + (1 - alpha) * current_bbox

# Hàm chính xử lý trừ nền
def background_subtraction_main(video_source=None, use_webcam=False, subtractor_type='KNN'):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        sg.popup_error("Error: Cannot open video source.")
        return

    # Chọn bộ trừ nền
    bg_subtractor = select_subtractor(subtractor_type)

    # Create window layout
    layout = [
        [sg.Image(filename='', key='image')],
        [sg.Image(filename='', key='mask_image')],
        [sg.Button('Tạm dừng/Chạy', key='Pause'), sg.Button('Quay lại', key='Back'), sg.Button('Thoát', key='Exit')]
    ]
    window = sg.Window('Thuật toán trừ nền', layout, location=(100, 100), finalize=True)

    # Khởi tạo SORT tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Khởi tạo từ điển lưu trữ bbox trước đó
    previous_bboxes = {}
    # Tạo một tập hợp để theo dõi ID đã đi qua ROI
    detected_ids = set()
    # Biến đếm tổng số lượng đối tượng
    total_count = 0

    # Định nghĩa vùng ROI
    roi_coords = (150, 35, 520, 300)  # (x1, y1, x2, y2)
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
                break

            # Resize frame
            frame = cv2.resize(frame, (640, 360))

            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (0, 255, 255), 2)

            # Apply Gaussian Blur to reduce noise
            blurred_frame = cv2.GaussianBlur(frame, (35, 35), 0)

            # Apply background subtraction
            foreground_mask = bg_subtractor.apply(blurred_frame)

            # Create binary image
            ret, threshold = cv2.threshold(foreground_mask.copy(), 240, 255, cv2.THRESH_BINARY)

            # Apply morphological transformations
            kernel = np.ones((7, 7), np.uint8)
            cleaned_mask = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=5)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=5)

            # Dilation to expand regions
            dilated = cv2.dilate(cleaned_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=3)

            # Lọc mask trên diện tích
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
            mask_filtered = np.zeros(dilated.shape, dtype=np.uint8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 10000:  # ngưỡng diện tích object
                    mask_filtered[labels == i] = 255

            # Find contours
            contours, hier = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Chuẩn bị danh sách phát hiện cho SORT
            detections = []

            # Draw bounding boxes
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 0.5 < aspect_ratio < 2.0:
                    detections.append([x, y, x + w, y + h, 1])  # Thêm vào danh sách phát hiện

            # Chuyển đổi danh sách phát hiện thành numpy array
            if len(detections) > 0:
                detections = np.array(detections)
            else:
                detections = np.empty((0, 5))

            # Cập nhật SORT tracker với các phát hiện mới
            tracks = tracker.update(detections)

            # Vẽ các bounding boxes và ID của đối tượng từ SORT tracker
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Làm mượt bbox
                previous_bbox = previous_bboxes.get(track_id)
                current_bbox = np.array([x1, y1, x2, y2])
                smoothed_bbox = smooth_bbox(current_bbox, previous_bbox)

                # Cập nhật bbox trước đó
                previous_bboxes[track_id] = smoothed_bbox

                # Vẽ bbox đã làm mượt
                x1, y1, x2, y2 = smoothed_bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Kiểm tra nếu toàn bộ bbox nằm trong ROI
                if (x1 >= roi_coords[0] and y1 >= roi_coords[1] and x2 <= roi_coords[2] and y2 <= roi_coords[3]):
                    if track_id not in detected_ids:
                        detected_ids.add(track_id)
                        total_count += 1

            # Hiển thị tổng số lượng đối tượng đã đi qua ROI trên khung hình
            cv2.putText(frame, f'Total count: {total_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Hiển thị kết quả trong cửa sổ GUI
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)

            # Hiển thị mask trong cửa sổ GUI
            mask_imgbytes = cv2.imencode('.png', cleaned_mask)[1].tobytes()
            window['mask_image'].update(data=mask_imgbytes)

    cap.release()
    cv2.destroyAllWindows()
    window.close()
