import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset.dataset import prepare_yolo_dataset, YoloDataset, collate_fn
from src.utils.class_names import NAMES_RU, generate_data_yaml
from src.models.yolo_models import train_yolo, train_rtdetr
from src.models.torchvision_models import build_faster_rcnn, build_ssd
from src.training.train import train_torchvision_model
from src.evaluation.metrics import evaluate_torchvision_model, summarize_results
from src.utils.visualization import (
    plot_loss_curve, plot_metrics_summary, plot_ap_per_class,
    plot_pred_vs_true, show_yolo_training_plots
)

def load_config(path='configs/default.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                         choices=['yolov8n', 'yolov10n', 'rtdetr', 'faster_rcnn', 'ssd'])
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Модели через ultralytics (YOLO, RT-DETR)
    if args.model in ['yolov8n', 'yolov10n', 'rtdetr']:
        weights = next(m['weights'] for m in cfg['models'] if m['name'] == args.model)
        if args.model == 'rtdetr':
            model, results = train_rtdetr(
                data_yaml=cfg['dataset']['data_yaml_path'],
                run_name=f'road_signs_{args.model}',
                epochs=cfg['training']['epochs_rtdetr'],
                imgsz=cfg['training']['imgsz'],
                batch=cfg['training']['batch_rtdetr'],
                patience=cfg['training']['patience'],
                device=cfg['training']['device']
            )
        else:
            model, results = train_yolo(
                model_name=weights,
                data_yaml=cfg['dataset']['data_yaml_path'],
                run_name=f'road_signs_{args.model}',
                epochs=cfg['training']['epochs_yolo'],
                imgsz=cfg['training']['imgsz'],
                batch=cfg['training']['batch_yolo'],
                patience=cfg['training']['patience'],
                device=cfg['training']['device']
            )
        show_yolo_training_plots(f'results/runs/road_signs_{args.model}', args.model)

    # Модели через torchvision (Faster R-CNN, SSD)
    else:
        train_ds = YoloDataset('data/processed/images/train', 'data/processed/labels/train')
        val_ds = YoloDataset('data/processed/images/val', 'data/processed/labels/val')
        train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_torchvision'],
                                   shuffle=True, collate_fn=collate_fn, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_torchvision'],
                                 shuffle=False, collate_fn=collate_fn, num_workers=2)

        model = build_faster_rcnn() if args.model == 'faster_rcnn' else build_ssd()
        model, loss_history = train_torchvision_model(
            model, train_loader, device, num_epochs=cfg['training']['epochs_torchvision']
        )

        results, all_preds, all_targets = evaluate_torchvision_model(model, val_loader, device)
        summarize_results(args.model, results)

        plot_loss_curve(loss_history, args.model, f'results/plots/{args.model}_loss.png')
        plot_metrics_summary(results, args.model, f'results/plots/{args.model}_metrics.png')
        plot_ap_per_class(results, NAMES_RU, args.model, f'results/plots/{args.model}_ap_per_class.png')
        plot_pred_vs_true(all_preds, all_targets, NAMES_RU, args.model, f'results/plots/{args.model}_pred_vs_true.png')

        torch.save(model.state_dict(), f'results/{args.model}_road_signs.pth')

    print(f'Готово: {args.model}')


if __name__ == '__main__':
    main()