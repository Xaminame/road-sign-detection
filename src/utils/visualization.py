"Построение графиков для оценки моделей: loss, метрики, AP по классам, сравнение предсказаний."
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image as PILImage


def plot_loss_curve(loss_history, model_name, save_path):
    "График изменения loss по эпохам."
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title(f'{model_name}: Loss по эпохам')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def plot_metrics_summary(results, model_name, save_path):
    plt.figure(figsize=(6, 5))
    metrics_names = ['mAP50-95', 'mAP50', 'mAP75']
    metrics_vals = [results['map'].item(), results['map_50'].item(), results['map_75'].item()]
    plt.bar(metrics_names, metrics_vals, color=['steelblue', 'orange', 'green'])
    plt.ylim(0, 1)
    plt.title(f'{model_name}: сводные метрики качества')
    plt.ylabel('Значение')
    plt.grid(axis='y')
    for i, v in enumerate(metrics_vals):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.savefig(save_path)
    plt.show()


def plot_ap_per_class(results, names_dict, model_name, save_path, top_n=15):
    if 'map_per_class' not in results or results['map_per_class'].numel() <= 1:
        print('AP по классам недоступен для этой модели')
        return

    ap_per_class = results['map_per_class'].numpy()
    classes_idx = results['classes'].numpy()
    order = np.argsort(-ap_per_class)[:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh([names_dict.get(classes_idx[i] - 1, str(classes_idx[i])) for i in order],
             ap_per_class[order], color='mediumseagreen')
    plt.xlabel('AP')
    plt.title(f'{model_name}: AP по классам (топ-{top_n})')
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_pred_vs_true(all_preds, all_targets, names_dict, model_name, save_path, top_n=15):
    "Сравнение количества истинных и предсказанных меток по классам."
    pred_labels_flat, true_labels_flat = [], []
    for p, t in zip(all_preds, all_targets):
        pred_labels_flat.extend(p['labels'].tolist())
        true_labels_flat.extend(t['labels'].tolist())

    pred_counts = Counter(pred_labels_flat)
    true_counts = Counter(true_labels_flat)
    all_cls = sorted(set(list(pred_counts.keys()) + list(true_counts.keys())))[:top_n]

    x = np.arange(len(all_cls))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, [true_counts.get(c, 0) for c in all_cls], width=0.4, label='Истинные', color='steelblue')
    plt.bar(x + 0.2, [pred_counts.get(c, 0) for c in all_cls], width=0.4, label='Предсказанные', color='salmon')
    plt.xticks(x, [names_dict.get(c - 1, str(c))[:15] for c in all_cls], rotation=75)
    plt.legend()
    plt.title(f'{model_name}: истинные vs предсказанные (по классам)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def show_yolo_training_plots(run_dir, model_name):
    "Показывает автоматически сгенерированные ultralytics-графики (results.png, confusion_matrix.png)."
    for img_name, title in [('results.png', f'{model_name}: графики обучения'),
                             ('confusion_matrix.png', f'{model_name}: матрица ошибок')]:
        img = PILImage.open(f'{run_dir}/{img_name}')
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.show()