import PySimpleGUI as sg
class GUI():
    def create_main_window(model_path):
        layout_main = [
            [sg.Text('Chọn chức năng:', font=('Helvetica', 18), justification='center', pad=(0, 20), expand_x=True)],
            [sg.Text(f'Mô hình hiện tại: {model_path}', key='model_text', font=('Helvetica', 12), justification='center', pad=(0, 10), expand_x=True)],
            [sg.Column([
                [sg.Frame('Tính toán & Theo dõi', [
                    [sg.Button('Đổi mô hình', size=(30, 2), font=('Helvetica', 14), pad=(5, 5))],
                    [sg.Button('Tính PPM', size=(30, 2), font=('Helvetica', 14), pad=(5, 5))],
                    [sg.Button('Theo dõi và tính kích thước đối tượng', size=(30, 2), font=('Helvetica', 14), pad=(5, 5))]
                ], title_color='white', relief=sg.RELIEF_RIDGE, pad=(0, 10))]
            ], pad=(5, 10), element_justification='center', vertical_alignment='top'),
            sg.Column([
                [sg.Frame('Phương pháp ML', [
                    [sg.Button('Thuật toán trừ nền', size=(30, 2), font=('Helvetica', 14), pad=(5, 5), key='background_subtraction')],
                    [sg.Button('Thuật toán phát hiện màu', size=(30, 2), font=('Helvetica', 14), pad=(5, 5), key='color_detection')]
                ], title_color='white', relief=sg.RELIEF_RIDGE, pad=(0, 10))],
                [sg.Frame('Dữ liệu & Huấn luyện', [
                    [sg.Button('Tạo dữ liệu mới', size=(30, 2), font=('Helvetica', 14), pad=(5, 5))],
                    [sg.Button('Bắt đầu huấn luyện', size=(30, 2), font=('Helvetica', 14), pad=(5, 5))]
                ], title_color='white', relief=sg.RELIEF_RIDGE, pad=(0, 10))]
            ], pad=(5, 10), element_justification='center')],
            [sg.Button('Thoát', size=(30, 2), font=('Helvetica', 14), pad=(5, 20))]
        ]

        return sg.Window('Theo dõi và tính kích thước vật thể', layout_main, size=(900, 590), element_justification='center', finalize=True, resizable=True)
    def create_background_subtraction_window():
        layout = [
            [sg.Text('Chọn nguồn:', font=('Helvetica', 16), justification='center')],
            [sg.Button('Từ webcam', size=(30, 2), font=('Helvetica', 14)), sg.Button('Từ video', size=(30, 2), font=('Helvetica', 14))],
            [sg.Button('Quay lại', size=(30, 2), font=('Helvetica', 14))]
        ]
        return sg.Window('Thuật toán trừ nền', layout, size=(800, 300), element_justification='center', finalize=True)

    def create_color_detection_window():
        layout = [
            [sg.Text('Chọn nguồn:', font=('Helvetica', 16), justification='center')],
            [sg.Button('Từ webcam', size=(30, 2), font=('Helvetica', 14)), sg.Button('Từ video', size=(30, 2), font=('Helvetica', 14))],
            [sg.Button('Quay lại', size=(30, 2), font=('Helvetica', 14))]
        ]
        return sg.Window('Thuật toán phát hiện màu', layout, size=(800, 300), element_justification='center', finalize=True, resizable=True)

    def create_calculate_ppm_window():
        layout = [
            [sg.Text('Tính toán Pixel Per Metric (PPM)', font=('Helvetica', 16))],
            [sg.Text('Đường dẫn ảnh:', size=(15, 1)), 
            sg.InputText(key='image_path', size=(40, 1), enable_events=True), 
            sg.FileBrowse('Chọn ảnh', size=(10, 1)),sg.Button('Chụp ảnh', size=(10, 1), key='capture_image')],
            [sg.Frame('Hình', layout=[[sg.Image(key='image', size=(400, 300))]], title_color='white', relief=sg.RELIEF_RIDGE)],
            [sg.Text('Kích thước thực tế (cm):', size=(20, 1)), 
            sg.InputText(key='metric', size=(40, 1))],
            [sg.Text('Chọn chiều dài sử dụng (metric):', size=(24, 1)), 
            sg.Combo(['Width', 'Height'], default_value='Width', key='dimension', size=(10, 1))],
            [sg.Button('Tính PPM', size=(15, 1)), 
            sg.Button('Quay lại', size=(15, 1))],
            [sg.Text('Giá trị PPM hiện tại:', size=(20, 1)), 
            sg.Text('', key='ppm_value', size=(15, 1))]
        ]
        return sg.Window('Tính PPM', layout, size=(800, 600), finalize=True, resizable=True)

    def create_source_selection_window(title):
        layout = [
            [sg.Text(f'{title}', font=('Helvetica', 16), justification='center', pad=(0, 20))],
            [sg.Button('Chọn từ webcam', size=(30, 2), font=('Helvetica', 14))],
            [sg.Button('Chọn từ video', size=(30, 2), font=('Helvetica', 14))],
            [sg.Button('Quay lại', size=(30, 2), font=('Helvetica', 14))]
        ]
        return sg.Window(title, layout, size=(400, 300), element_justification='center', finalize=True, resizable=True)

    def create_add_data_window():
        add_data_layout = [
            [sg.Text('Chọn nguồn:', font=('Helvetica', 16), justification='center')],
            [sg.Button('Chụp ảnh từ webcam', size=(30, 2), font=('Helvetica', 14)), sg.Button('Chọn video từ thư mục', size=(30, 2), font=('Helvetica', 14))],
            [sg.Button('Quay lại', size=(30, 2), font=('Helvetica', 14))],
            [sg.Image(sg.EMOJI_BASE64_READING, size=(300, 300))]
        ]
        return sg.Window('Tạo dữ liệu mới', add_data_layout, size=(800, 300), element_justification='center', finalize=True, resizable=True)

    def create_track_object_window():
        track_object_layout = [
            [sg.Text('Chọn nguồn:', font=('Helvetica', 16), justification='center')],
            [sg.Button('Theo dõi đối tượng từ webcam', size=(30, 2), font=('Helvetica', 14)), sg.Button('Theo dõi đối tượng từ video', size=(30, 2), font=('Helvetica', 14))],
            [sg.Button('Quay lại', size=(30, 2), font=('Helvetica', 14))],
            [sg.Image(sg.EMOJI_BASE64_READING, size=(300, 300))]
        ]
        return sg.Window('Theo dõi và tính kích thước đối tượng', track_object_layout, size=(800, 300), element_justification='center', finalize=True, resizable=True)

    def create_model_selection_window(model_list):
        layout_model_selection = [
            [sg.Text('Chọn model YOLOv8:')],
            [sg.Listbox(values=model_list, size=(40, 5), key='model')],
            [sg.Button('Xác nhận'), sg.Button('Chọn đường dẫn model')]
        ]
        return sg.Window('Chọn Model', layout_model_selection, size=(400, 200))

   
    def create_object_tracking_window(roi_coords=(150, 150, 305, 400),pixel_per_metric=16.79):
        layout = [
            [sg.Image(filename='', key='image', pad=(0,0), background_color='black')],
            [sg.Button('Tạm dừng/Chạy', key='Pause', size=(15, 1)), sg.Button('Quay lại', key='Back', size=(10, 1)), sg.Button('Thoát', key='Exit', size=(10, 1))],
            [
                sg.Text('Confidence Threshold', size=(20, 1), pad=((5, 5), (5, 5))),
                sg.Slider(range=(0, 1), resolution=0.01, default_value=0.8, size=(20, 20), orientation='horizontal', key='conf_thresh', pad=((5, 5), (5, 5))),
                sg.Text('Pixel Per Metric', size=(20, 1), justification='right', pad=((5, 5), (5, 5))),
                sg.InputText(f'', key='pixel_per_metric', size=(10, 1), justification='right', pad=((5, 5), (5, 5))),
                sg.Button('Cập nhật PPM', key='UpdatePPM', size=(10, 1), pad=((5, 5), (5, 5)))
            ],
            [
                sg.Text('x1:', size=(2, 1), pad=((5, 5), (5, 5))), sg.InputText(f'{roi_coords[0]}', key='x1', size=(5, 1), pad=((5, 5), (5, 5))),
                sg.Text('y1:', size=(2, 1), pad=((5, 5), (5, 5))), sg.InputText(f'{roi_coords[1]}', key='y1', size=(5, 1), pad=((5, 5), (5, 5))),
                sg.Text('x2:', size=(2, 1), pad=((5, 5), (5, 5))), sg.InputText(f'{roi_coords[2]}', key='x2', size=(5, 1), pad=((5, 5), (5, 5))),
                sg.Text('y2:', size=(2, 1), pad=((5, 5), (5, 5))), sg.InputText(f'{roi_coords[3]}', key='y2', size=(5, 1), pad=((5, 5), (5, 5))),
                sg.Button('Cập nhật tọa độ', key='UpdateROI', size=(15, 1), pad=((10, 5), (5, 5))),
                sg.Text('Giá trị PPM hiện tại:', size=(15, 1), justification='right', pad=((10, 5), (5, 5))),
                sg.Text(f'', key='pixel_per_metric_value', size=(10, 1), justification='right', pad=((5, 5), (5, 5)))
            ],
            [sg.Text('ROI hiện tại:', size=(12, 1), pad=((5, 5), (5, 5))), sg.Text(f'({roi_coords[0]}, {roi_coords[1]}, {roi_coords[2]}, {roi_coords[3]})', key='current_roi', pad=((5, 5), (5, 5)))]
        ]
        
        return sg.Window('Theo Dõi và tính kích thước', layout, location=(700, 400), finalize=True, resizable=True)



    def create_capture_window():
        layout = [
            [sg.Image(filename='', key='image', enable_events=True, pad=(0,0), background_color='black')],
            [sg.Button('Tạm dừng/Chạy', key='Pause'), sg.Button('Chụp ảnh', key='Capture'), sg.Button('Quay lại', key='Back'), sg.Button('Thoát', key='Exit')]
        ]
        return sg.Window('Capture Images', layout, location=(400, 200), finalize=True, resizable=True)

    def open_model_selection(model_list):
        layout_model_selection_new = [
            [sg.Text('Chọn model YOLOv8:')],
            [sg.Listbox(values=model_list, size=(40, 5), key='model')],
            [sg.Button('Xác nhận'), sg.Button('Chọn đường dẫn model')]
        ]
        window_model_selection = sg.Window('Chọn Model', layout_model_selection_new, size=(400, 200))
        model = None
        model_path = None
        class_names = {}
        num_classes = 0

        while model is None:
            event, values = window_model_selection.read()
            if event == sg.WIN_CLOSED:
                break
            if event == 'Xác nhận':
                model_path = values['model'][0] if values['model'] else None
                if model_path:
                    from demo import Demo  # Import tại đây để tránh import vòng tròn
                    model, class_names, num_classes = Demo.load_model(model_path)
                    window_model_selection.close()
            if event == 'Chọn đường dẫn model':
                model_path = sg.popup_get_file('Chọn file model YOLO', file_types=(("YOLO Model Files", "*.pt"),))
                if model_path:
                    from demo import Demo  # Import tại đây để tránh import vòng tròn
                    model, class_names, num_classes = Demo.load_model(model_path)
                    window_model_selection.close()

        return model, model_path, class_names, num_classes
