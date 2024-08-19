import PySimpleGUI as sg
from ultralytics import YOLO
import threading
import queue
import time

sg.theme('DarkTeal9')
# Khởi tạo hàng đợi và cờ dừng
message_queue = queue.Queue()
stop_flag = threading.Event()

# Hàm huấn luyện mô hình
def train_model(yaml_path, model_path, epochs, batch_size, img_size, output_dir):
    try:
        model = YOLO(model_path)
        for epoch in range(epochs):
            if stop_flag.is_set():
                message_queue.put("Huấn luyện đã bị dừng.")
                break
            model.train(data=yaml_path, epochs=1, batch=batch_size, imgsz=img_size, project=output_dir, name='custom_train')
            message_queue.put(f"Epoch {epoch+1}/{epochs} hoàn thành.")
        else:
            message_queue.put("Huấn luyện hoàn thành!")
    except Exception as e:
        message_queue.put(f'Lỗi khi huấn luyện: {e}')

# Giao diện PySimpleGUI
layout = [
    [sg.Text('Đường dẫn tới file YAML cấu hình dữ liệu'), 
     sg.InputText(key='-YAML-'), 
     sg.FileBrowse(file_types=(("YAML Files", "*.yaml"),))],
    [sg.Text('Đường dẫn tới mô hình YOLOv8'), 
     sg.InputText(key='-MODEL-'), 
     sg.FileBrowse(file_types=(("PT Files", "*.pt"),))],
    [sg.Text('Số lượng epochs'), 
     sg.InputText(default_text='10', key='-EPOCHS-', size=(10, 1))],
    [sg.Text('Batch size'), 
     sg.InputText(default_text='16', key='-BATCH-', size=(10, 1))],
    [sg.Text('Kích thước hình ảnh (imgsz)'), 
     sg.InputText(default_text='640', key='-IMGSZ-', size=(10, 1))],
    [sg.Text('Trạng thái'), sg.Text('', key='-STATUS-', size=(40, 1))],
    [sg.Button('Train'), sg.Button('Dừng'), sg.Button('Thoát')]
]

# Tạo cửa sổ
window = sg.Window('Train YOLOv8 Model', layout)

# Vòng lặp sự kiện
def main():
    training_thread = None

    while True:
        event, values = window.read(timeout=100)
        
        # Kiểm tra hàng đợi
        try:
            message = message_queue.get_nowait()
            sg.popup(message)
            window['-STATUS-'].update(message)
        except queue.Empty:
            pass
        
        if event == sg.WINDOW_CLOSED or event == 'Thoát':
            if training_thread and training_thread.is_alive():
                stop_flag.set()  # Đặt cờ dừng
                window['-STATUS-'].update('Đang dừng huấn luyện...')
                training_thread.join()  # Đợi luồng huấn luyện kết thúc
            break

        if event == 'Train':
            yaml_path = values['-YAML-']
            model_path = values['-MODEL-']
            
            # Kiểm tra định dạng tệp
            if not yaml_path.endswith('.yaml'):
                sg.popup_error('Vui lòng chọn file YAML cho cấu hình dữ liệu.')
                continue
            if not model_path.endswith('.pt'):
                sg.popup_error('Vui lòng chọn file mô hình với định dạng .pt.')
                continue
            
            # Lấy các giá trị từ các trường nhập liệu
            try:
                epochs = int(values['-EPOCHS-'])
                batch_size = int(values['-BATCH-'])
                img_size = int(values['-IMGSZ-'])
            except ValueError:
                sg.popup_error('Vui lòng nhập các giá trị hợp lệ cho các tham số.')
                continue
            
            # Cập nhật trạng thái
            window['-STATUS-'].update('Đang huấn luyện...')
            stop_flag.clear()

            # Khởi tạo và chạy luồng huấn luyện
            output_dir = 'runs'
            training_thread = threading.Thread(target=train_model, args=(yaml_path, model_path, epochs, batch_size, img_size, output_dir))
            training_thread.start()

        if event == 'Dừng':
            if training_thread and training_thread.is_alive():
                stop_flag.set()  # Đặt cờ dừng
                window['-STATUS-'].update('Đang dừng huấn luyện...')
    
    window.close()

if __name__ == "__main__":
    main()
