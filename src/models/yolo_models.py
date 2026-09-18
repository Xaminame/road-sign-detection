"Для обучения YOLO-моделей (YOLOv8n, YOLOv10n) через ultralytics."
from ultralytics import YOLO


def train_yolo(model_name, data_yaml, run_name, epochs=30, imgsz=320, batch=16, patience=10, device=0):
    "
    Обучает YOLO-модель.

    Args:
        model_name: имя предобученной модели, например 'yolov8n.pt' или 'yolov10n.pt'
        data_yaml: путь к data.yaml
        run_name: имя запуска (папка для результатов)
        epochs, imgsz, batch, patience, device: гиперпараметры обучения

    Returns:
        обученная модель и объект результатов
    "
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project='results/runs',
        name=run_name,
        device=device
    )
    return model, results

"Обучает RT-DETR через ultralytics."
def train_rtdetr(data_yaml, run_name='road_signs_rtdetr', epochs=25, imgsz=320, batch=8, patience=10, device=0):
    from ultralytics import RTDETR
    model = RTDETR('rtdetr-l.pt')
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project='results/runs',
        name=run_name,
        device=device
    )
    return model, results