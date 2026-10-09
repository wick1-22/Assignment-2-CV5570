"""Train and compare Faster R-CNN, RetinaNet, and RT-DETR on the ACID subset."""

import argparse
import csv
import json
import math
import random
import shutil
import time
from datetime import datetime
from functools import partial
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from data_tools import CLASS_ORDER, COLORS, load_coco


def _font(size):
    """Load a readable PIL font, falling back to its built-in font."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _to_device(target, device):
    import torch
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}


class CocoDetectionDataset:
    """A small COCO reader returning the format expected by torchvision detectors."""

    def __init__(self, json_path, image_root):
        coco, _ = load_coco(json_path)
        self.coco = coco
        self.image_root = str(image_root)
        category_names = {item["id"]: item["name"] for item in coco["categories"]}
        self.cat_to_label = {cat_id: CLASS_ORDER.index(name) + 1 for cat_id, name in category_names.items()}
        self.annotations = defaultdict(list)
        for ann in coco["annotations"]:
            self.annotations[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.coco["images"])

    def __getitem__(self, index):
        import numpy as np
        import torch

        meta = self.coco["images"][index]
        with Image.open(Path(self.image_root) / meta["file_name"]) as source:
            image = source.convert("RGB")
            image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        boxes, labels, area, crowd = [], [], [], []
        for ann in self.annotations[meta["id"]]:
            x, y, width, height = ann["bbox"]
            boxes.append([x, y, x + width, y + height])
            labels.append(self.cat_to_label[ann["category_id"]])
            area.append(width * height)
            crowd.append(ann.get("iscrowd", 0))
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([meta["id"]], dtype=torch.int64),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(crowd, dtype=torch.int64),
        }
        return image_tensor, target


def _collate(batch):
    return tuple(zip(*batch))


def _make_torchvision_model(family, num_classes, image_size, pretrained=True):
    import torch
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetHead
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights

    if family == "faster_rcnn":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.score_thresh = 0.001
        model.roi_heads.detections_per_img = 200
    elif family == "retinanet":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = retinanet_resnet50_fpn_v2(weights=weights)
        anchors_per_location = model.head.classification_head.num_anchors
        model.head = RetinaNetHead(model.backbone.out_channels, anchors_per_location, num_classes,
                                   norm_layer=partial(torch.nn.GroupNorm, 32))
        model.head.regression_head._loss_type = "giou"
        model.score_thresh = 0.001
        model.detections_per_img = 200
    else:
        raise ValueError(f"Unknown torchvision detector: {family}")
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    return model


def _box_iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(tp_flags, fp_flags, number_gt):
    if number_gt == 0:
        return None
    if not tp_flags:
        return 0.0
    true_positive, false_positive = 0, 0
    recalls, precisions = [], []
    for tp, fp in zip(tp_flags, fp_flags):
        true_positive += int(tp)
        false_positive += int(fp)
        recalls.append(true_positive / number_gt)
        precisions.append(true_positive / max(true_positive + false_positive, 1))
    samples = []
    for point in range(101):
        recall_level = point / 100
        samples.append(max((p for r, p in zip(recalls, precisions) if r >= recall_level), default=0.0))
    return sum(samples) / len(samples)


def calculate_metrics(ground_truth, predictions):
    """Compute class AP with 101-point interpolation at IoU .50 and .50:.95."""
    thresholds = [round(0.50 + step * 0.05, 2) for step in range(10)]
    per_class = {}
    for label, class_name in enumerate(CLASS_ORDER, start=1):
        gt_by_image = defaultdict(list)
        for item in ground_truth:
            if item["label"] == label:
                gt_by_image[item["image_id"]].append(item["box"])
        class_dets = [item for item in predictions if item["label"] == label]
        class_dets.sort(key=lambda item: item["score"], reverse=True)
        aps = []
        for threshold in thresholds:
            matched = {image_id: set() for image_id in gt_by_image}
            tp_flags, fp_flags = [], []
            for det in class_dets:
                image_id = det["image_id"]
                best_iou, best_index = 0.0, None
                for index, gt_box in enumerate(gt_by_image.get(image_id, [])):
                    if index in matched.get(image_id, set()):
                        continue
                    overlap = _box_iou(det["box"], gt_box)
                    if overlap > best_iou:
                        best_iou, best_index = overlap, index
                is_tp = best_index is not None and best_iou >= threshold
                tp_flags.append(is_tp)
                fp_flags.append(not is_tp)
                if is_tp:
                    matched[image_id].add(best_index)
            aps.append(_average_precision(tp_flags, fp_flags, sum(len(boxes) for boxes in gt_by_image.values())))
        available = [value for value in aps if value is not None]
        per_class[class_name] = {
            "AP50": aps[0],
            "AP50_95": sum(available) / len(available) if available else None,
            "ground_truth_instances": sum(len(boxes) for boxes in gt_by_image.values()),
        }
    ap50_values = [row["AP50"] for row in per_class.values() if row["AP50"] is not None]
    ap_values = [row["AP50_95"] for row in per_class.values() if row["AP50_95"] is not None]
    return {
        "mAP50": sum(ap50_values) / len(ap50_values) if ap50_values else None,
        "mAP50_95": sum(ap_values) / len(ap_values) if ap_values else None,
        "per_class": per_class,
        "metric_note": "Class-wise AP uses 101-point interpolated precision-recall; mean over IoU 0.50:0.95 in 0.05 steps.",
    }


def _collect_ground_truth(dataset):
    records = []
    for ann in dataset.coco["annotations"]:
        x, y, width, height = ann["bbox"]
        label = dataset.cat_to_label[ann["category_id"]]
        records.append({"image_id": ann["image_id"], "label": label, "box": [x, y, x + width, y + height]})
    return records


def _torchvision_predictions(model, dataset, device, family):
    import torch
    predictions = []
    image_times = []
    model.eval()
    for index in range(len(dataset)):
        image_tensor, target = dataset[index]
        _sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model([image_tensor.to(device)])[0]
        _sync(device)
        image_times.append(time.perf_counter() - start)
        image_id = int(target["image_id"].item())
        for box, label, score in zip(output["boxes"].cpu().tolist(), output["labels"].cpu().tolist(), output["scores"].cpu().tolist()):
            if family == "retinanet":
                label += 1
            predictions.append({"image_id": image_id, "label": int(label), "score": float(score), "box": box})
    return predictions, image_times


def train_torchvision(family, train_json, val_json, image_root, output_dir, epochs=10, batch_size=4, image_size=640, seed=42, resume=False):
    import torch
    from torch.utils.data import DataLoader

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    train_set = CocoDetectionDataset(train_json, image_root)
    val_set = CocoDetectionDataset(val_json, image_root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=_collate, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=_collate, pin_memory=device.type == "cuda")

    class_count = len(CLASS_ORDER) + 1 if family == "faster_rcnn" else len(CLASS_ORDER)
    model = _make_torchvision_model(family, class_count, image_size).to(device)
    learning_rate = 0.0005 if family == "retinanet" else 0.005
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs - 2, 1), gamma=0.1)
    run_dir = Path(output_dir) / family
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    start_epoch = 0
    prior_training_seconds = 0.0
    progress_path = run_dir / "progress.json"
    last_path = run_dir / "last.pth"
    if resume and progress_path.exists() and last_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        model.load_state_dict(torch.load(last_path, map_location=device, weights_only=True))
        history = progress.get("history", [])
        best_val = float(progress.get("best_val", float("inf")))
        start_epoch = int(progress.get("completed_epochs", 0))
        prior_training_seconds = float(progress.get("training_seconds", 0.0))
        print(f"Resuming {family} from epoch {start_epoch}; optimizer momentum restarts.", flush=True)
    start_all = time.perf_counter()

    segment_start = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_total, train_batches = 0.0, 0
        for images, targets in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [_to_device(target, device) for target in targets]
            if family == "retinanet":
                for target in targets:
                    target["labels"] = target["labels"] - 1
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss).item():
                diagnostic = {"model": family, "epoch": epoch + 1, "batch": train_batches + 1,
                              "learning_rate": learning_rate, "error": "non-finite training loss"}
                (run_dir / "non_finite_loss.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
                raise FloatingPointError(
                    f"{family} produced a non-finite training loss at epoch {epoch + 1}, "
                    f"batch {train_batches + 1}; see non_finite_loss.json"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.detach().cpu())
            train_batches += 1

        # Torchvision returns detection losses only while the model is in train mode.
        model.train()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(device, non_blocking=True) for image in images]
                targets = [_to_device(target, device) for target in targets]
                if family == "retinanet":
                    for target in targets:
                        target["labels"] = target["labels"] - 1
                losses = model(images, targets)
                batch_val_loss = sum(losses.values())
                if not torch.isfinite(batch_val_loss).item():
                    diagnostic = {"model": family, "epoch": epoch + 1, "batch": val_batches + 1,
                                  "learning_rate": learning_rate, "error": "non-finite validation loss"}
                    (run_dir / "non_finite_loss.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
                    raise FloatingPointError(
                        f"{family} produced a non-finite validation loss at epoch {epoch + 1}, "
                        f"batch {val_batches + 1}; see non_finite_loss.json"
                    )
                val_loss_total += float(batch_val_loss.detach().cpu())
                val_batches += 1
        scheduler.step()
        train_loss = train_loss_total / max(train_batches, 1)
        val_loss = val_loss_total / max(val_batches, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        torch.save(model.state_dict(), last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), run_dir / "best.pth")
        elapsed = prior_training_seconds + time.perf_counter() - segment_start
        progress = {"completed_epochs": epoch + 1, "best_val": best_val, "history": history,
                    "training_seconds": elapsed}
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        with open(run_dir / "loss_history.json", "w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
        _plot_loss_curves(history, run_dir / "loss_curves.png", f"{family.replace('_', ' ').title()} training and validation loss")
        epoch_seconds = time.perf_counter() - epoch_start
        completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"{family} epoch {epoch + 1}/{epochs}: train loss={train_loss:.4f}, "
              f"val loss={val_loss:.4f}, epoch time={epoch_seconds / 60:.1f} min, "
              f"completed at {completed_at}", flush=True)

    training_seconds = prior_training_seconds + time.perf_counter() - segment_start
    with open(run_dir / "loss_history.json", "w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    _plot_loss_curves(history, run_dir / "loss_curves.png", f"{family.replace('_', ' ').title()} training and validation loss")
    with open(run_dir / "training_config.json", "w", encoding="utf-8") as stream:
        json.dump({"pretrained": "TorchVision COCO ResNet50-FPN v2", "epochs": epochs, "batch_size": batch_size,
                   "image_size": image_size, "optimizer": "SGD", "learning_rate": learning_rate,
                   "momentum": 0.9, "weight_decay": 0.0005, "augmentation": "none",
                   "device": str(device), "gpu": gpu_name, "training_seconds": training_seconds}, stream, indent=2)
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device, weights_only=True))
    predictions, image_times = _torchvision_predictions(model, val_set, device, family)
    metrics = calculate_metrics(_collect_ground_truth(val_set), predictions)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_name = ("Faster R-CNN ResNet50-FPN v2 COCO weights" if family == "faster_rcnn"
                       else "RetinaNet ResNet50-FPN v2 COCO weights")
    result = {"model": family, "family": "two-stage" if family == "faster_rcnn" else "one-stage",
              "pretrained_checkpoint": checkpoint_name, "optimizer": "SGD", "learning_rate": learning_rate,
              "weight_decay": 0.0005, "momentum": 0.9,
              "training_seconds": training_seconds, "parameters": n_params,
              "fps": len(val_set) / sum(image_times) if image_times else 0,
              "inference_ms_per_image": 1000 * sum(image_times) / max(len(image_times), 1),
              "epochs": epochs, "batch_size": batch_size, "image_size": image_size,
              "device": str(device), "gpu": gpu_name, **metrics}
    _save_predictions_and_examples(family, val_set, predictions, run_dir)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def evaluate_saved_torchvision(family, val_json, image_root, output_dir, image_size=640, examples_only=False):
    """Evaluate a saved best.pth, optionally regenerating only the example board."""
    import torch
    run_dir = Path(output_dir) / family
    checkpoint = run_dir / "best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Cannot evaluate saved {family}: missing {checkpoint}")
    config_path = run_dir / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_count = len(CLASS_ORDER) + 1 if family == "faster_rcnn" else len(CLASS_ORDER)
    model = _make_torchvision_model(family, class_count, image_size, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    val_set = CocoDetectionDataset(val_json, image_root)
    predictions, image_times = _torchvision_predictions(model, val_set, device, family)
    metrics = calculate_metrics(_collect_ground_truth(val_set), predictions)
    result = {
        "model": family,
        "family": "two-stage" if family == "faster_rcnn" else "one-stage",
        "pretrained_checkpoint": config.get("pretrained", "TorchVision COCO ResNet50-FPN v2"),
        "checkpoint_used": str(checkpoint),
        "training_seconds": config.get("training_seconds"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fps": len(val_set) / sum(image_times) if image_times else 0,
        "inference_ms_per_image": 1000 * sum(image_times) / max(len(image_times), 1),
        "epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "image_size": config.get("image_size", image_size),
        "device": str(device),
        "gpu": config.get("gpu", torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
        "optimizer": config.get("optimizer", "not recorded"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "momentum": config.get("momentum"),
        **metrics,
    }
    _save_predictions_and_examples(family, val_set, predictions, run_dir)
    if examples_only:
        print(f"Regenerated validation examples for {family}; existing metrics were preserved.", flush=True)
    else:
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Evaluated saved {family} checkpoint and wrote {run_dir / 'metrics.json'}", flush=True)
    return result


def _prepare_ultralytics_dataset(splits_dir, annotation_json, image_root, staging_dir):
    """Copy images to Colab-local storage and write YOLO-format label files."""
    import yaml
    splits_dir, staging_dir = Path(splits_dir), Path(staging_dir)
    coco, _ = load_coco(annotation_json)
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    cat_to_index = {cat_id: CLASS_ORDER.index(name) for cat_id, name in categories.items()}
    for split_name in ("train", "val", "test"):
        split_json = json.loads((splits_dir / f"{split_name}.json").read_text(encoding="utf-8"))
        image_out = staging_dir / "images" / split_name
        label_out = staging_dir / "labels" / split_name
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        anns = defaultdict(list)
        for ann in split_json["annotations"]:
            anns[ann["image_id"]].append(ann)
        for meta in split_json["images"]:
            source = Path(image_root) / meta["file_name"]
            destination = image_out / meta["file_name"]
            if not destination.exists():
                shutil.copy2(source, destination)
            label_lines = []
            for ann in anns[meta["id"]]:
                x, y, width, height = ann["bbox"]
                center_x = (x + width / 2) / meta["width"]
                center_y = (y + height / 2) / meta["height"]
                label_lines.append(f"{cat_to_index[ann['category_id']]} {center_x:.7f} {center_y:.7f} {width / meta['width']:.7f} {height / meta['height']:.7f}")
            (label_out / f"{Path(meta['file_name']).stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
    yaml_path = staging_dir / "acid.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(staging_dir), "train": "images/train", "val": "images/val",
                                         "test": "images/test", "names": CLASS_ORDER}, sort_keys=False), encoding="utf-8")
    return str(yaml_path)


def _ultralytics_prediction(model, image):
    result = model.predict(source=image, imgsz=640, conf=0.001, verbose=False)[0]
    records = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        labels = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        for box, label, score in zip(boxes, labels, scores):
            records.append({"label": int(label) + 1, "score": float(score), "box": box})
    return records


def train_ultralytics(family, model_name, data_yaml, val_json, image_root, output_dir,
                      epochs=10, batch_size=4, image_size=640, device="0", resume=False):
    from ultralytics import YOLO, RTDETR
    model_class = YOLO if family == "yolo" else RTDETR
    import torch
    if not torch.cuda.is_available() and device != "cpu":
        device = "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    run_dir = Path(output_dir) / family
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    last_weights = run_dir / "weights" / "last.pt"
    resume_checkpoint = bool(resume and last_weights.exists())
    model = model_class(str(last_weights) if resume_checkpoint else model_name)
    if resume_checkpoint:
        print(f"Resuming {family} from {last_weights}", flush=True)
    elif resume:
        print(f"No completed RT-DETR checkpoint found in {run_dir / 'weights'}; starting from pretrained {model_name}.", flush=True)
    last_reported_epoch = {"value": -1}

    def report_epoch_end(trainer):
        epoch = int(trainer.epoch) + 1
        if epoch == last_reported_epoch["value"]:
            return
        last_reported_epoch["value"] = epoch
        completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"{family} epoch {epoch}/{epochs} completed at {completed_at}", flush=True)

    model.add_callback("on_fit_epoch_end", report_epoch_end)
    start = time.perf_counter()
    model.train(data=data_yaml, epochs=epochs, imgsz=image_size, batch=batch_size,
                          project=str(run_dir.parent), name=family, exist_ok=True,
                          device=device, workers=2, pretrained=True, seed=42,
                          resume=resume_checkpoint,
                          augment=False, mosaic=0.0, fliplr=0.0, flipud=0.0,
                          degrees=0.0, translate=0.0, scale=0.0, plots=True, verbose=True)
    session_seconds = time.perf_counter() - start
    trainer = model.trainer
    run_path = Path(trainer.save_dir)
    training_seconds = _ultralytics_training_seconds(run_path / "results.csv", session_seconds)
    trainer_args = trainer.args
    best_weights = run_path / "weights" / "best.pt"
    model = model_class(str(best_weights))
    val_set = CocoDetectionDataset(val_json, image_root)
    predictions = []
    image_times = []
    for index, meta in enumerate(val_set.coco["images"]):
        image_path = Path(image_root) / meta["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        start_one = time.perf_counter()
        found = _ultralytics_prediction(model, image)
        image_times.append(time.perf_counter() - start_one)
        for item in found:
            item["image_id"] = meta["id"]
            predictions.append(item)
    metrics = calculate_metrics(_collect_ground_truth(val_set), predictions)
    model_object = getattr(model, "model", None)
    n_params = sum(parameter.numel() for parameter in model_object.parameters()) if model_object else 0
    result = {"model": family, "family": "one-stage" if family == "yolo" else "transformer",
              "pretrained_checkpoint": model_name, "best_checkpoint": str(best_weights),
              "training_seconds": training_seconds, "parameters": n_params,
              "fps": len(val_set) / sum(image_times) if image_times else 0,
              "inference_ms_per_image": 1000 * sum(image_times) / max(len(image_times), 1),
              "epochs": epochs, "batch_size": batch_size, "image_size": image_size,
              "device": str(device), "gpu": gpu_name, **metrics}
    result["optimizer"] = str(getattr(trainer_args, "optimizer", "library default"))
    result["learning_rate"] = float(getattr(trainer_args, "lr0", 0.0))
    with open(run_path / "training_config.json", "w", encoding="utf-8") as stream:
        json.dump({"pretrained_checkpoint": model_name, "epochs": epochs, "batch_size": batch_size,
                   "image_size": image_size, "optimizer": result["optimizer"],
                   "learning_rate": result["learning_rate"], "augmentation": "disabled",
                   "device": str(device), "gpu": gpu_name, "training_seconds": training_seconds}, stream, indent=2)
    loss_history = _ultralytics_loss_history(run_path / "results.csv")
    if loss_history:
        with open(run_path / "loss_history.json", "w", encoding="utf-8") as stream:
            json.dump(loss_history, stream, indent=2)
        _plot_loss_curves(loss_history, run_path / "loss_curves.png", "RT-DETR training and validation loss")
    _save_predictions_and_examples(family, val_set, predictions, run_path)
    with open(run_path / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def _image_status(image_id, gt_records, pred_records, threshold=0.5):
    gts = [item for item in gt_records if item["image_id"] == image_id]
    preds = sorted([item for item in pred_records if item["image_id"] == image_id], key=lambda item: item["score"], reverse=True)
    matched = set()
    tp, fp = 0, 0
    for pred in preds:
        best_iou, best_index = 0.0, None
        for index, gt in enumerate(gts):
            if index in matched or gt["label"] != pred["label"]:
                continue
            overlap = _box_iou(gt["box"], pred["box"])
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_index is not None and best_iou >= threshold:
            matched.add(best_index)
            tp += 1
        else:
            fp += 1
    fn = len(gts) - len(matched)
    recall = tp / max(len(gts), 1)
    precision = tp / max(tp + fp, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision,
            "quality": recall, "success": tp > 0 and recall >= 0.5 and precision >= 0.5}


def _save_predictions_and_examples(model_name, dataset, predictions, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    gts = _collect_ground_truth(dataset)
    statuses = []
    visible_predictions = [item for item in predictions if item["score"] >= 0.25]
    for meta in dataset.coco["images"]:
        status = _image_status(meta["id"], gts, visible_predictions)
        statuses.append({"id": meta["id"], "name": meta["file_name"], **status})
    successes = sorted((row for row in statuses if row["success"]), key=lambda row: row["name"])[:3]
    failures = sorted((row for row in statuses if not row["success"]), key=lambda row: (row["fn"] + row["fp"], row["quality"]), reverse=True)[:3]
    selected = [(row, "Success") for row in successes] + [(row, "Failure") for row in failures]
    board = Image.new("RGB", (960, 3 * 350), "white")
    board_draw = ImageDraw.Draw(board)
    cat_names = {index + 1: name for index, name in enumerate(CLASS_ORDER)}
    for tile_index, (status, label) in enumerate(selected):
        meta = next(item for item in dataset.coco["images"] if item["id"] == status["id"])
        with Image.open(Path(dataset.image_root) / meta["file_name"]) as source:
            source = source.convert("RGB")
            scale = min(930 / source.width, 300 / source.height)
            size = (int(source.width * scale), int(source.height * scale))
            tile = source.resize(size)
        tile_draw = ImageDraw.Draw(tile)
        for gt in gts:
            if gt["image_id"] == status["id"]:
                x0, y0, x1, y1 = [coordinate * scale for coordinate in gt["box"]]
                tile_draw.rectangle((x0, y0, x1, y1), outline=(25, 190, 75), width=3)
                tile_draw.text((x0, max(0, y0 - 17)), f"GT {cat_names[gt['label']]}", fill=(10, 125, 45))
        for pred in predictions:
            if pred["image_id"] == status["id"] and pred["score"] >= 0.25:
                x0, y0, x1, y1 = [coordinate * scale for coordinate in pred["box"]]
                tile_draw.rectangle((x0, y0, x1, y1), outline=(235, 45, 45), width=3)
                tile_draw.text((x0, min(tile.height - 16, y1)), f"P {cat_names.get(pred['label'], pred['label'])} {pred['score']:.2f}", fill=(175, 20, 20))
        row, column = divmod(tile_index, 2)
        y, x = row * 350, column * 480 + 8
        board_draw.text((x, y + 5), f"{label}: {status['name']} | TP {status['tp']} FP {status['fp']} FN {status['fn']}", fill=(25, 35, 45), font=_font(16))
        board.paste(tile, (x, y + 35))
    board.save(run_dir / "qualitative_examples.png")
    with open(run_dir / "qualitative_selection.json", "w", encoding="utf-8") as stream:
        json.dump({"rule": "Success examples have at least one correct match, at least 50% ground-truth recall, and at least 50% precision at IoU >= 0.5 with a 0.25 display score threshold. Failure examples do not meet this rule.", "selected": selected}, stream, indent=2)


def _make_comparison_plot(results, output_path):
    width, height = 900, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 65, width - 45, height - 80
    draw.text((left, 22), "Validation speed and accuracy", fill=(30, 40, 50), font=_font(22))
    draw.line((left, top, left, bottom, right, bottom), fill=(50, 60, 70), width=2)
    max_fps = max([row.get("fps", 0) for row in results] + [1])
    for row in results:
        x = left + row.get("fps", 0) / max_fps * (right - left)
        y = bottom - row.get("mAP50_95", 0) * (bottom - top)
        color = {"two-stage": (220, 95, 70), "one-stage": (55, 145, 220), "transformer": (70, 175, 100)}.get(row["family"], (100, 100, 100))
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="white", width=2)
        draw.text((x + 12, y - 10), row["model"], fill=(35, 40, 50), font=_font(15))
    draw.text((left, bottom + 20), "FPS (higher is faster)", fill=(40, 45, 55), font=_font(15))
    draw.text((8, top - 20), "mAP@0.5:0.95", fill=(40, 45, 55), font=_font(15))
    image.save(output_path)


def _plot_loss_curves(history, output_path, title):
    """Save a simple train/validation loss line chart."""
    if not history:
        return
    width, height = 820, 480
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 80, 55, width - 35, height - 65
    draw.text((left, 18), title, fill=(30, 40, 50), font=_font(21))
    draw.line((left, top, left, bottom, right, bottom), fill=(55, 65, 75), width=2)
    values = [item[key] for item in history for key in ("train_loss", "val_loss") if item.get(key) is not None]
    max_loss = max(values) if values else 1.0
    max_epoch = max(item["epoch"] for item in history)
    colors = {"train_loss": (45, 125, 205), "val_loss": (225, 95, 65)}
    for key in ("train_loss", "val_loss"):
        points = []
        for item in history:
            if item.get(key) is None:
                continue
            x = left + (item["epoch"] - 1) / max(max_epoch - 1, 1) * (right - left)
            y = bottom - item[key] / max_loss * (bottom - top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=colors[key], width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[key])
    draw.text((left, bottom + 20), "Epoch", fill=(40, 45, 55), font=_font(14))
    draw.text((left, top + 5), "Loss", fill=(40, 45, 55), font=_font(14))
    draw.line((right - 160, top + 8, right - 135, top + 8), fill=colors["train_loss"], width=3)
    draw.text((right - 128, top), "Train", fill=(40, 45, 55), font=_font(14))
    draw.line((right - 80, top + 8, right - 55, top + 8), fill=colors["val_loss"], width=3)
    draw.text((right - 48, top), "Val", fill=(40, 45, 55), font=_font(14))
    image.save(output_path)


def _ultralytics_loss_history(results_csv):
    if not Path(results_csv).exists():
        return []
    rows = []
    with open(results_csv, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for epoch, row in enumerate(reader, start=1):
            parsed = {key.strip(): value for key, value in row.items() if key}
            def sum_losses(prefix):
                values = []
                for key, value in parsed.items():
                    if key.lower().startswith(prefix) and "loss" in key.lower():
                        try:
                            values.append(float(value))
                        except (TypeError, ValueError):
                            pass
                return sum(values) if values else None
            rows.append({"epoch": epoch, "train_loss": sum_losses("train/"), "val_loss": sum_losses("val/")})
    return rows


def _ultralytics_training_seconds(results_csv, fallback_seconds=0.0):
    """Add logged elapsed-time intervals, including segments from resumed runs."""
    if not Path(results_csv).exists():
        return fallback_seconds
    elapsed = 0.0
    previous = None
    with open(results_csv, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                current = float(row.get("time", ""))
            except (TypeError, ValueError):
                continue
            if previous is None or current < previous:
                elapsed += current
            else:
                elapsed += current - previous
            previous = current
    return max(elapsed, fallback_seconds)


def run_all(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.splits)
    dataset_summary = json.loads((Path(args.analysis) / "dataset_summary.json").read_text(encoding="utf-8"))
    if not all((split_dir / f"{name}.json").exists() for name in ("train", "val", "test")):
        raise FileNotFoundError("Split JSON files are missing. Run src/data_tools.py first.")

    results = []
    def completed_or_train(family, train_function):
        if args.eval_only or args.examples_only:
            if family == "rtdetr":
                raise ValueError("Checkpoint evaluation is currently available for Faster R-CNN and RetinaNet only.")
            return evaluate_saved_torchvision(family, split_dir / "val.json", args.images, output_dir,
                                              args.image_size, examples_only=args.examples_only)
        result_path = output_dir / family / "metrics.json"
        if args.resume and result_path.exists():
            print(f"Using completed {family} result from {result_path}")
            return json.loads(result_path.read_text(encoding="utf-8"))
        if args.resume and family == "retinanet":
            failed_dir = output_dir / family
            progress_path = failed_dir / "progress.json"
            best_path = failed_dir / "best.pth"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                rows = progress.get("history", [])
                has_non_finite = any(
                    not math.isfinite(float(row.get(key, float("nan"))))
                    for row in rows for key in ("train_loss", "val_loss")
                )
                if has_non_finite or not best_path.exists():
                    archive_name = f"retinanet_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    archive_path = output_dir / archive_name
                    failed_dir.rename(archive_path)
                    print(f"Archived invalid RetinaNet attempt at {archive_path}; starting a clean run.", flush=True)
        if args.resume and family == "rtdetr":
            run_dir = output_dir / family
            last_weights = run_dir / "weights" / "last.pt"
            best_weights = run_dir / "weights" / "best.pt"
            if run_dir.exists() and not result_path.exists() and not last_weights.exists() and not best_weights.exists():
                archive_name = f"rtdetr_incomplete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                archive_path = output_dir / archive_name
                run_dir.rename(archive_path)
                print(f"Archived incomplete RT-DETR attempt at {archive_path}; starting from pretrained weights.", flush=True)
        return train_function()

    if args.model == "summary":
        results = []
        for family in ("faster_rcnn", "retinanet", "rtdetr"):
            result_path = output_dir / family / "metrics.json"
            if not result_path.exists():
                raise FileNotFoundError(f"Cannot summarize before {family} completes: {result_path}")
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
    requested = ("faster_rcnn", "retinanet", "rtdetr") if args.model == "all" else (() if args.model == "summary" else (args.model,))
    if "faster_rcnn" in requested:
        print("Starting Faster R-CNN", flush=True)
        results.append(completed_or_train("faster_rcnn", lambda: train_torchvision(
            "faster_rcnn", split_dir / "train.json", split_dir / "val.json", args.images,
        output_dir, args.epochs, args.batch_size, args.image_size, args.seed, args.resume)))
    if "retinanet" in requested:
        print("Starting RetinaNet", flush=True)
        results.append(completed_or_train("retinanet", lambda: train_torchvision(
            "retinanet", split_dir / "train.json", split_dir / "val.json", args.images,
            output_dir, args.epochs, args.batch_size, args.image_size, args.seed, args.resume)))
    if "rtdetr" in requested:
        print("Preparing YOLO-format copies in Colab-local storage ...", flush=True)
        yaml_path = _prepare_ultralytics_dataset(split_dir, args.annotations, args.images, args.staging)
        print("Starting RT-DETR", flush=True)
        results.append(completed_or_train("rtdetr", lambda: train_ultralytics(
            "rtdetr", "rtdetr-l.pt", yaml_path, split_dir / "val.json", args.images,
            output_dir, args.epochs, args.batch_size, args.image_size, args.device, args.resume)))

    # Each model is now an independent command. Only build the complete
    # comparison summary when all three saved metric files are available.
    if args.model not in ("all", "summary"):
        print(json.dumps({"model": args.model, "result": results[0] if results else None}, indent=2), flush=True)
        return

    # Keep all measurement inputs and outputs together for the report builder.
    import platform
    import torch
    import torchvision
    import ultralytics
    summary = {"dataset": dataset_summary, "models": results,
               "shared_setup": {"split_seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                                "image_size": args.image_size, "augmentation": "disabled for all models",
                                "python": platform.python_version(), "torch": torch.__version__,
                                "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
                                "gpu": next((row.get("gpu") for row in results if row.get("gpu") not in (None, "CPU")),
                                            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")}}
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with open(output_dir / "comparison.csv", "w", newline="", encoding="utf-8") as stream:
        columns = ["model", "family", "mAP50", "mAP50_95", "fps", "inference_ms_per_image", "parameters", "training_seconds"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in columns})
    _make_comparison_plot(results, output_dir / "speed_accuracy.png")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train and compare three ACID detector families.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--splits", default="outputs/dataset_analysis/splits")
    parser.add_argument("--analysis", default="outputs/dataset_analysis")
    parser.add_argument("--output", default="outputs/model_runs")
    parser.add_argument("--staging", default="/content/acid_yolo_dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Reuse completed results or resume from a saved checkpoint")
    parser.add_argument("--model", choices=("all", "faster_rcnn", "retinanet", "rtdetr", "summary"), default="all",
                        help="Train one detector independently, or all three in sequence")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate an existing TorchVision best.pth checkpoint without training")
    parser.add_argument("--examples-only", action="store_true", help="Regenerate validation examples from a TorchVision checkpoint without replacing saved metrics")
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
