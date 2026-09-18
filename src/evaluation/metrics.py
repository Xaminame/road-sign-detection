"Вычисление метрик качества (mAP, Precision, Recall) для моделей детекции."
import numpy as np
from collections import Counter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch


def evaluate_torchvision_model(model, val_loader, device):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            metric.update(preds_cpu, targets)
            all_preds.extend(preds_cpu)
            all_targets.extend(targets)

    results = metric.compute()
    return results, all_preds, all_targets

def summarize_results(model_name, results):
    "Печатает краткую сводку метрик."
    print(f"--- {model_name} ---")
    print(f"mAP50-95: {results['map'].item():.3f}")
    print(f"mAP50: {results['map_50'].item():.3f}")
    print(f"mAP75: {results['map_75'].item():.3f}")