"""Build an Assignment 2 report from saved dataset and experiment outputs."""

import argparse
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from xml.sax.saxutils import escape


CLASS_ORDER = ["dozer", "dump_truck", "excavator"]
MODEL_LABELS = {"faster_rcnn": "Faster R-CNN", "retinanet": "RetinaNet", "rtdetr": "RT-DETR-L"}


def fmt(value, digits=3):
    if value is None:
        return "Not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_json(path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_metrics(root):
    summary = load_json(Path(root) / "outputs/model_runs/metrics_summary.json") or {}
    by_name = {item.get("model"): item for item in summary.get("models", [])}
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        item = load_json(Path(root) / "outputs/model_runs" / model_name / "metrics.json")
        if item:
            by_name[model_name] = item
    summary["models"] = [by_name[name] for name in ("faster_rcnn", "retinanet", "rtdetr") if name in by_name]
    return summary if summary["models"] else None


def missing_submission_outputs(root):
    root = Path(root)
    missing = []
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        model_dir = root / "outputs/model_runs" / model_name
        for filename in ("loss_curves.png", "qualitative_examples.png"):
            if not (model_dir / filename).exists():
                missing.append(f"{MODEL_LABELS[model_name]} {filename}")
    for filename in ("comparison.csv", "speed_accuracy.png", "metrics_summary.json"):
        if not (root / "outputs/model_runs" / filename).exists():
            missing.append(filename)
    return missing


def convergence_summary(root):
    observations = []
    histories = {}
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        history = load_json(Path(root) / "outputs/model_runs" / model_name / "loss_history.json")
        if not history:
            continue
        histories[model_name] = history
        values = [row.get("val_loss") for row in history if row.get("val_loss") is not None]
        if len(values) < 2 or values[0] == 0:
            continue
        reduction = (values[0] - values[-1]) / abs(values[0]) * 100
        tail = values[-min(3, len(values)):]
        trend = "still decreasing at the end" if all(a > b for a, b in zip(tail, tail[1:])) else "flattening or changing direction near the end"
        observations.append(f"{MODEL_LABELS.get(model_name, model_name)}: its validation objective changed by {reduction:.1f}% from epoch 1 to the last epoch and was {trend}.")
    if not observations:
        return "No completed training histories are available yet."
    if all(name in histories for name in ("faster_rcnn", "retinanet", "rtdetr")):
        best_epochs = {}
        for name, history in histories.items():
            available = [row for row in history if row.get("val_loss") is not None]
            if available:
                best_epochs[name] = min(available, key=lambda row: row["val_loss"])["epoch"]
        times = {}
        for name in histories:
            metrics = load_json(Path(root) / "outputs/model_runs" / name / "metrics.json")
            if metrics and metrics.get("training_seconds") is not None:
                times[name] = metrics["training_seconds"] / 60
        comparison = []
        if best_epochs:
            comparison.append("The minimum recorded validation loss occurred at epoch " + ", ".join(
                f"{MODEL_LABELS[name]} {epoch}" for name, epoch in best_epochs.items()) + ".")
        if len(times) == 3:
            comparison.append("For 10 epochs, recorded training time was " + ", ".join(
                f"{MODEL_LABELS[name]} {minutes:.1f} min" for name, minutes in times.items()) + ".")
        if "rtdetr" in best_epochs and "faster_rcnn" in best_epochs and "retinanet" in best_epochs and len(times) == 3:
            comparison.append(
                f"RT-DETR-L reached its lowest recorded validation loss at epoch {best_epochs['rtdetr']}, "
                f"compared with epoch {best_epochs['faster_rcnn']} for Faster R-CNN and "
                f"epoch {best_epochs['retinanet']} for RetinaNet. Its {times['rtdetr']:.1f}-minute run was shorter in wall time "
                f"than Faster R-CNN ({times['faster_rcnn']:.1f} min) and RetinaNet ({times['retinanet']:.1f} min); "
                "loss definitions differ, so this compares convergence timing rather than loss magnitude."
            )
        observations.extend(comparison)
    return " ".join(observations) + " Loss definitions differ across architectures, so compare their within-run trends and time-to-best rather than the absolute loss values."


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22,
                              leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"), spaceAfter=10))
    styles.add(ParagraphStyle(name="ReportSub", parent=styles["Normal"], fontSize=10, leading=14,
                              alignment=TA_CENTER, textColor=colors.HexColor("#526477"), spaceAfter=12))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=15, leading=19,
                              textColor=colors.HexColor("#17324D"), spaceBefore=7, spaceAfter=7))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11.5, leading=14,
                              textColor=colors.HexColor("#24628A"), spaceBefore=7, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9, leading=13,
                              textColor=colors.HexColor("#25313D"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.5, leading=10,
                              textColor=colors.HexColor("#596777"), spaceAfter=4))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=9, leading=13,
                              textColor=colors.HexColor("#6B4A00"), backColor=colors.HexColor("#FFF2CC"),
                              borderColor=colors.HexColor("#E6C96E"), borderWidth=0.6, borderPadding=8, spaceAfter=10))
    return styles


def para(story, text, style):
    story.append(Paragraph(text, style))


def add_table(story, rows, widths=None, font_size=8):
    wrapped = []
    for row_index, row in enumerate(rows):
        text_color = colors.white if row_index == 0 else colors.HexColor("#24313E")
        font_name = "Helvetica-Bold" if row_index == 0 else "Helvetica"
        wrapped.append([Paragraph(escape(str(value)), ParagraphStyle(f"cell{row_index}", fontName=font_name, fontSize=font_size,
                                                                       leading=font_size + 2, textColor=text_color))
                        for value in row])
    table = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFD8E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 7))


def add_image(story, path, width=165 * mm, max_height=112 * mm):
    path = Path(path)
    if not path.exists():
        return False
    image = ReportImage(str(path))
    ratio = min(width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    image.hAlign = "CENTER"
    story.append(image)
    story.append(Spacer(1, 5))
    return True


def _results_table(metrics):
    rows = [["Model", "Family", "mAP@0.5", "mAP@0.5:0.95", "FPS", "Parameters (M)", "Training time (min)"]]
    if not metrics:
        rows.append(["Run models in Colab", "-", "Pending", "Pending", "Pending", "Pending", "Pending"])
        return rows
    for model in metrics.get("models", []):
        rows.append([
            MODEL_LABELS.get(model["model"], model["model"]), model.get("family", "-"),
            fmt(model.get("mAP50")), fmt(model.get("mAP50_95")), fmt(model.get("fps"), 1),
            fmt(model.get("parameters", 0) / 1_000_000, 2), f"{model.get('training_seconds', 0) / 60:.1f}",
        ])
    return rows


def build_markdown(dataset, metrics, output_path):
    root = output_path.parent.parent
    models = metrics.get("models", []) if metrics else []
    ready = len(models) == 3
    missing_outputs = missing_submission_outputs(root)
    complete = ready and not missing_outputs
    by_name = {item["model"]: item for item in models}
    lines = [
        "# CV5570 Assignment 2 - Object Detection Report", "",
        "**Indian Institute of Technology Madras**  ",
        "CV5570: Computer Vision for Construction Engineering  ",
        "Submitted by: N P J Vedhaanth | Roll number: ID26S422", "",
        "GitHub repository: https://github.com/wick1-22/Assignment-2-CV5570", "",
        ("" if complete else "**Draft: some required saved outputs are still missing.**"), "",
        "## Contents", "1. Executive Summary", "2. Objective and Scope", "3. Dataset Description and Preprocessing",
        "4. Detector Methods and Training Setup", "5. Validation Results", "6. Training and Convergence",
        "7. Detection Examples and Failure Analysis", "8. Comparison and Deployment Discussion",
        "9. Limitations", "10. Conclusion", "11. References", "",
        "## 1. Executive Summary", "",
    ]
    if ready:
        fastest = max(models, key=lambda m: m.get("fps", 0))
        most_accurate = max(models, key=lambda m: m.get("mAP50_95", 0) or 0)
        lines.append(
            f"This report compares Faster R-CNN, RetinaNet, and RT-DETR-L on {dataset['image_count']:,} ACID images. "
            f"The highest validation mAP@0.5:0.95 was from {MODEL_LABELS[most_accurate['model']]} "
            f"({most_accurate['mAP50_95']:.3f}); the highest measured speed was {MODEL_LABELS[fastest['model']]} "
            f"({fastest['fps']:.1f} FPS). All results use the same saved validation split."
        )
    else:
        lines.append("The assignment compares a two-stage, one-stage, and transformer-based detector. The executive summary will use the completed validation metrics after all runs are available.")
    lines.extend([
        "", "## 2. Objective and Scope", "",
        "The aim was to train one detector from each of the two-stage, one-stage, and transformer families on the ACID dataset. The comparison reports class-wise average precision, speed, parameter count, training time, loss history, and validation examples for Dozer, Dump Truck, and Excavator.",
        "", "## 3. Dataset Description and Preprocessing", "",
        f"The selected subset contains {dataset['image_count']:,} images and {dataset['instance_count']:,} object instances. Annotations are in COCO JSON format with pixel boxes stored as [x, y, width, height]. The split uses seed {dataset['split_seed']} and proportions 70/20/10.",
        "", "| Split | Images | Instances | Dozer | Dump Truck | Excavator |", "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("train", "val", "test"):
        split = dataset["splits"][name]
        counts = split["class_instances"]
        lines.append(f"| {name.title()} | {split['images']:,} | {split['annotations']:,} | {counts['dozer']:,} | {counts['dump_truck']:,} | {counts['excavator']:,} |")
    lines.extend([
        "", f"The median normalized box area is {dataset['bbox_area_fraction']['median']:.3f}; the median width-to-height ratio is {dataset['bbox_aspect_ratio']['median']:.2f}. The held-out test split is not used for the reported model-selection metrics.",
        "", "![Class counts](../outputs/dataset_analysis/class_frequencies.png)",
        "", "![Bounding-box area distribution](../outputs/dataset_analysis/bbox_area_distribution.png)",
        "", "![Bounding-box aspect-ratio distribution](../outputs/dataset_analysis/bbox_aspect_ratio.png)",
        "", "![Six images with ground-truth boxes](../outputs/dataset_analysis/ground_truth_samples.png)",
        "", "## 4. Detector Methods and Training Setup", "",
        "**Faster R-CNN.** The region proposal network selects candidate regions; the second stage classifies and refines them. It was chosen as a two-stage baseline for equipment that appears at different sizes in cluttered construction scenes (Ren et al., 2015). The model starts from TorchVision COCO-pretrained ResNet-50-FPN v2 weights.", "",
        "**RetinaNet.** RetinaNet predicts class scores and box offsets in one pass. Its focal loss reduces the contribution from easy background locations, which are common in object detection (Lin et al., 2017). It provides the one-stage comparison. The model starts from TorchVision COCO-pretrained ResNet-50-FPN v2 weights.", "",
        "**RT-DETR-L.** This transformer detector uses an encoder/decoder and set-based predictions. It was selected to compare a transformer-based approach on cluttered scenes and to measure its speed/accuracy trade-off (Zhao et al., 2024). It starts from the pretrained `rtdetr-l` checkpoint. The prediction interface used here does not return attention maps, so its errors are reviewed from the validation boxes.", "",
        "The models use the same saved split, 640-pixel inputs, 10-epoch budget, and disabled augmentation. The model-specific optimizer settings and runtime values are recorded below.", "",
        "| Model | Checkpoint | Optimizer | Learning rate | Batch | Epochs | Device |", "|---|---|---|---:|---:|---:|---|",
    ])
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        model = by_name.get(model_name)
        if model:
            lines.append(f"| {MODEL_LABELS[model_name]} | {model.get('pretrained_checkpoint', '-')} | {model.get('optimizer', '-')} | {fmt(model.get('learning_rate'), 4)} | {model.get('batch_size', '-')} | {model.get('epochs', '-')} | {model.get('device', '-')} |")
        else:
            lines.append(f"| {MODEL_LABELS[model_name]} | Pending | - | - | - | - | - |")
    lines.extend([
        "", "Faster R-CNN used SGD at learning rate 0.005. RetinaNet used 0.0005 after the initial 0.005 run produced non-finite losses. Both used momentum 0.9 and weight decay 0.0005. RT-DETR used the optimizer settings saved by Ultralytics.",
        "", "## 5. Validation Results", "",
        "| Model | Family | mAP@0.5 | mAP@0.5:0.95 | FPS | Parameters (M) | Training time (min) |", "|---|---|---:|---:|---:|---:|---:|",
    ])
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        model = by_name.get(model_name)
        if model:
            lines.append("| " + " | ".join(map(str, _results_table({"models": [model]})[1])) + " |")
        else:
            lines.append(f"| {MODEL_LABELS[model_name]} | pending | - | - | - | - | - |")
    lines.extend(["", "### Class-wise average precision", "", "| Model | Class | AP@0.5 | AP@0.5:0.95 |", "|---|---|---:|---:|"])
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        model = by_name.get(model_name)
        for class_name in CLASS_ORDER:
            result = model.get("per_class", {}).get(class_name) if model else None
            if result:
                lines.append(f"| {MODEL_LABELS[model_name]} | {class_name.replace('_', ' ').title()} | {fmt(result['AP50'])} | {fmt(result['AP50_95'])} |")
            else:
                lines.append(f"| {MODEL_LABELS[model_name]} | {class_name.replace('_', ' ').title()} | - | - |")
    lines.extend(["", "## 6. Training and Convergence", "", convergence_summary(output_path.parent.parent) if ready else "Convergence comparison is pending the remaining model run."])
    if ready:
        for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
            lines.extend(["", f"![{MODEL_LABELS[model_name]} loss curves](../outputs/model_runs/{model_name}/loss_curves.png)"])
    lines.extend(["", "## 7. Detection Examples and Failure Analysis", "",
                  "Green boxes mark ground-truth objects and red boxes mark predictions. The boards show selected validation successes and failures using the same selection rule for each model."])
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        if model_name in by_name:
            lines.extend(["", f"### {MODEL_LABELS[model_name]}", ""])
            board_path = root / "outputs/model_runs" / model_name / "qualitative_examples.png"
            if board_path.exists():
                lines.append(f"![{MODEL_LABELS[model_name]} validation detections](../outputs/model_runs/{model_name}/qualitative_examples.png)")
            else:
                lines.append("Validation board pending; the saved checkpoint can regenerate it without retraining.")
            selection = load_json(root / "outputs/model_runs" / model_name / "qualitative_selection.json")
            if selection and selection.get("selected"):
                lines.extend(["", "| Type | Image | TP | FP | FN |", "|---|---|---:|---:|---:|"])
                for entry in selection["selected"]:
                    if isinstance(entry, list) and len(entry) == 2:
                        item, kind = entry
                        lines.append(f"| {kind} | {item.get('name', '-')} | {item.get('tp', 0)} | {item.get('fp', 0)} | {item.get('fn', 0)} |")
    lines.extend(["", "**Observed in the displayed validation examples:** Faster R-CNN detects the excavator in 00002.jpg (TP 2, FP 2, FN 0) and the dozer in 00028.jpg (TP 1, FP 1, FN 0), while the crowded or overlapping scenes 04460.jpg, 05513.jpg, and 09385.jpg contain extra predictions and missed objects. RetinaNet detects the isolated dozer in 00028.jpg, but the crowded scenes 07480.jpg and 08970.jpg contain several false boxes and missed objects. RT-DETR produces no correct match for the dump truck in 06108.jpg (11 false positives and one missed object); the crowded scenes 05513.jpg and 02159.jpg also show overlapping extra boxes and missed instances.",
                  "", "**Possible explanations to investigate:** small objects at 640-pixel input resolution may be harder to localize; CNN anchor sizes may not fit the observed box shapes; the class distribution is imbalanced; and overlap or NMS may contribute to duplicate or suppressed detections. These are hypotheses, not causes established by this experiment.",
                  "", "## 8. Comparison and Deployment Discussion", ""])
    if ready:
        fastest = max(models, key=lambda m: m.get("fps", 0))
        most_accurate = max(models, key=lambda m: m.get("mAP50_95", 0) or 0)
        lines.append(f"The fastest measured model was {MODEL_LABELS[fastest['model']]} at {fastest['fps']:.1f} FPS. The highest mAP@0.5:0.95 was from {MODEL_LABELS[most_accurate['model']]} at {most_accurate['mAP50_95']:.3f}. These results make the fastest model a candidate for real-time monitoring and the most accurate model a candidate for offline review, subject to the class-wise results and failure examples.")
    else:
        lines.append("The deployment comparison will be completed after all three models have validation metrics.")
    lines.extend(["", "![Validation speed and accuracy](../outputs/model_runs/speed_accuracy.png)",
                  "", "## 9. Limitations", "",
                  "The experiment uses one random image split and a short training budget. The split is not grouped by scene, so visually similar images may occur in different subsets. The detector families use different training implementations and optimization settings. The reported speed depends on the recorded GPU and inference code. Validation results do not replace evaluation on the held-out test split.",
                  "", "## 10. Conclusion", ""])
    if ready:
        fastest = max(models, key=lambda m: m.get("fps", 0))
        most_accurate = max(models, key=lambda m: m.get("mAP50_95", 0) or 0)
        lines.append(f"On this validation split, {MODEL_LABELS[most_accurate['model']]} had the highest mAP@0.5:0.95 ({most_accurate['mAP50_95']:.3f}) and {MODEL_LABELS[fastest['model']]} had the highest speed ({fastest['fps']:.1f} FPS). The choice between them depends on whether accuracy or throughput is the main requirement. The class-wise scores and example failures should be considered before deployment.")
    else:
        lines.append("The conclusion will be updated after the three model results are available.")
    if missing_outputs:
        lines.extend(["", "## Required outputs still to add", "",
                      "The saved metrics are available, but the submission bundle is not complete until these artifacts are added:", ""])
        lines.extend(f"- {item}" for item in missing_outputs)
        if "Faster R-CNN qualitative_examples.png" in missing_outputs:
            lines.append("- Run the Faster R-CNN examples-only recovery cell in the Colab notebook. It uses the saved checkpoint and preserves the completed metrics.")
    lines.extend(["", "## 11. References", "",
                  "1. Ren et al., 'Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,' NeurIPS, 2015. https://papers.nips.cc/paper_files/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html",
                  "2. Lin et al., 'Focal Loss for Dense Object Detection,' ICCV, 2017. https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_Dense_Object_Detection_ICCV_2017_paper.pdf",
                  "3. Zhao et al., 'DETRs Beat YOLOs on Real-time Object Detection,' CVPR, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html",
                  "4. ACID dataset, link provided in the assignment brief: https://drive.google.com/uc?id=1Qg_X5FygUMBRTcVFPb0s1fQP-20f8n8O",
                  "5. Ultralytics RT-DETR documentation: https://docs.ultralytics.com/models/rtdetr", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


class ReportDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "H1x":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


def build_pdf(root, output_path, dataset, metrics):
    root = Path(root)
    models = metrics.get("models", []) if metrics else []
    by_name = {item["model"]: item for item in models}
    ready = len(models) == 3
    missing_outputs = missing_submission_outputs(root)
    complete = ready and not missing_outputs
    styles = make_styles()
    styles.add(ParagraphStyle(name="TOCHeading1", parent=styles["Bodyx"], leftIndent=0,
                              firstLineIndent=0, spaceBefore=3, spaceAfter=3, leading=13))
    doc = ReportDocTemplate(str(output_path), pagesize=A4, rightMargin=17 * mm,
                            leftMargin=17 * mm, topMargin=16 * mm, bottomMargin=17 * mm,
                            title="CV5570 Assignment 2 Object Detection Report")
    story = []

    # Cover page
    story.append(Spacer(1, 22 * mm))
    para(story, "INDIAN INSTITUTE OF TECHNOLOGY MADRAS", styles["ReportSub"])
    story.append(Spacer(1, 10 * mm))
    para(story, "CV5570: Computer Vision for Construction Engineering", styles["ReportSub"])
    para(story, "Assignment 2 - Report", styles["ReportTitle"])
    story.append(Spacer(1, 12 * mm))
    para(story, "Comparative Evaluation of Faster R-CNN, RetinaNet, and RT-DETR-L for Construction Equipment Detection", styles["ReportSub"])
    story.append(Spacer(1, 25 * mm))
    para(story, "Submitted by:<br/>N P J Vedhaanth<br/>Roll Number: ID26S422", styles["ReportSub"])
    story.append(Spacer(1, 18 * mm))
    para(story, "GitHub Repository: https://github.com/wick1-22/Assignment-2-CV5570", styles["Smallx"])
    para(story, "ACID dataset | Classes: Dozer, Dump Truck, Excavator", styles["Smallx"])
    if not complete:
        story.append(Spacer(1, 10 * mm))
        para(story, "DRAFT - some required saved outputs are not yet included.", styles["Callout"])

    # Contents page
    story.append(PageBreak())
    para(story, "Table of Contents", styles["Heading1"])
    toc = TableOfContents()
    toc.levelStyles = [styles["TOCHeading1"]]
    toc.dotsMinLevel = 0
    story.append(toc)

    story.append(PageBreak())
    para(story, "1. Executive Summary", styles["H1x"])
    if ready:
        fastest = max(models, key=lambda item: item.get("fps", 0))
        most_accurate = max(models, key=lambda item: item.get("mAP50_95", 0) or 0)
        para(story,
             f"This work compares a two-stage detector (Faster R-CNN), a one-stage detector (RetinaNet), and a transformer detector (RT-DETR-L) on the ACID construction-image subset. The experiment uses {dataset['image_count']:,} images and {dataset['instance_count']:,} annotated objects from three classes. The highest validation mAP@0.5:0.95 was {MODEL_LABELS[most_accurate['model']]} ({most_accurate['mAP50_95']:.3f}); the highest measured speed was {MODEL_LABELS[fastest['model']]} ({fastest['fps']:.1f} FPS). The choice between accuracy and throughput is discussed with the per-class scores and example detections.", styles["Bodyx"])
    else:
        para(story, "This report compares Faster R-CNN, RetinaNet, and RT-DETR-L on the three ACID equipment classes. The validation summary will be complete when all model metric files are available.", styles["Bodyx"])

    para(story, "2. Objective and Scope", styles["H1x"])
    para(story, "The objective was to train and compare one detector from each of the two-stage, one-stage, and transformer families. The comparison covers validation AP at IoU 0.5 and 0.50:0.95, class-wise AP, inference speed, parameter count, training time, convergence, and qualitative successes and failures. The test split was held out.", styles["Bodyx"])

    para(story, "3. Dataset Description and Preprocessing", styles["H1x"])
    para(story, f"The selected ACID subset contains {dataset['image_count']:,} images and {dataset['instance_count']:,} object instances. The annotations are COCO JSON, with bounding boxes stored in pixel coordinates as [x, y, width, height]. Image references were checked before processing. The images were shuffled using seed {dataset['split_seed']} and split into 70% training, 20% validation, and 10% test.", styles["Bodyx"])
    split_rows = [["Split", "Images", "Instances", "Dozer", "Dump Truck", "Excavator"]]
    for split_name in ("train", "val", "test"):
        split = dataset["splits"][split_name]
        counts = split["class_instances"]
        split_rows.append([split_name.title(), split["images"], split["annotations"],
                           counts["dozer"], counts["dump_truck"], counts["excavator"]])
    add_table(story, split_rows, [23 * mm, 21 * mm, 23 * mm, 22 * mm, 30 * mm, 29 * mm], font_size=7.5)
    para(story, f"The full class counts are Dozer {dataset['class_instances']['dozer']:,}, Dump Truck {dataset['class_instances']['dump_truck']:,}, and Excavator {dataset['class_instances']['excavator']:,}. Median normalized box area is {dataset['bbox_area_fraction']['median']:.3f}; median width-to-height ratio is {dataset['bbox_aspect_ratio']['median']:.2f}.", styles["Bodyx"])
    add_image(story, root / "outputs/dataset_analysis/class_frequencies.png", width=150 * mm, max_height=67 * mm)

    story.append(PageBreak())
    para(story, "Dataset distributions and examples", styles["H1x"])
    add_image(story, root / "outputs/dataset_analysis/bbox_area_distribution.png", width=150 * mm, max_height=78 * mm)
    add_image(story, root / "outputs/dataset_analysis/bbox_aspect_ratio.png", width=150 * mm, max_height=78 * mm)
    story.append(PageBreak())
    para(story, "Dataset examples", styles["H2x"])
    para(story, "Six dataset images with their ground-truth boxes are shown below. These are dataset examples, not model predictions.", styles["Bodyx"])
    add_image(story, root / "outputs/dataset_analysis/ground_truth_samples.png", width=165 * mm, max_height=230 * mm)

    story.append(PageBreak())
    para(story, "4. Detector Methods and Training Setup", styles["H1x"])
    para(story, "Faster R-CNN uses a region proposal network to identify candidate object regions, followed by a classifier and box-refinement stage. It was selected as a two-stage baseline for equipment at different sizes and against cluttered site backgrounds (Ren et al., 2015). RetinaNet predicts class scores and box offsets densely in one pass; focal loss reduces the weight of easy background examples, a common source of imbalance in detection (Lin et al., 2017). RT-DETR-L uses a transformer encoder/decoder and set-based object predictions. It was included to compare the transformer family on these scenes and measure its speed/accuracy trade-off (Zhao et al., 2024). The predictor used for this run does not return attention maps, so its errors are reviewed using validation predictions and ground truth.", styles["Bodyx"])
    para(story, "Faster R-CNN and RetinaNet use TorchVision COCO-pretrained ResNet-50-FPN v2 weights. RT-DETR-L uses the pretrained `rtdetr-l` checkpoint. The three runs use the same saved split, input size, epoch budget, and disabled augmentations. Their training libraries and optimizer settings are not identical.", styles["Bodyx"])
    setup_rows = [["Model", "Pretrained checkpoint", "Optimizer", "Learning rate", "Batch", "Epochs", "Device"]]
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        model = by_name.get(model_name)
        if model:
            setup_rows.append([MODEL_LABELS[model_name], model.get("pretrained_checkpoint", "-"),
                               model.get("optimizer", "-"), fmt(model.get("learning_rate"), 4),
                               model.get("batch_size", "-"), model.get("epochs", "-"), model.get("device", "-")])
        else:
            setup_rows.append([MODEL_LABELS[model_name], "Pending", "-", "-", "-", "-", "-"])
    add_table(story, setup_rows, [25 * mm, 47 * mm, 21 * mm, 20 * mm, 14 * mm, 14 * mm, 18 * mm], font_size=6.7)
    para(story, "Faster R-CNN used SGD with learning rate 0.005, momentum 0.9, and weight decay 0.0005. RetinaNet used learning rate 0.0005 after the initial 0.005 attempt produced non-finite losses; momentum and weight decay were unchanged. RT-DETR used the optimizer settings recorded by Ultralytics. Training was run on the device recorded in each model's metrics file.", styles["Bodyx"])
    if metrics and metrics.get("shared_setup"):
        setup = metrics["shared_setup"]
        para(story, f"Runtime versions: Python {setup.get('python', 'not recorded')}, PyTorch {setup.get('torch', 'not recorded')}, TorchVision {setup.get('torchvision', 'not recorded')}, Ultralytics {setup.get('ultralytics', 'not recorded')}. GPU: {setup.get('gpu', 'not recorded')}.", styles["Smallx"])

    story.append(PageBreak())
    para(story, "5. Validation Results", styles["H1x"])
    para(story, "AP is calculated with 101-point interpolated precision-recall. FPS is measured on the validation images using the run's inference path and device.", styles["Bodyx"])
    add_table(story, _results_table(metrics), [25 * mm, 24 * mm, 21 * mm, 27 * mm, 15 * mm, 22 * mm, 26 * mm], font_size=7)
    class_rows = [["Model", "Class", "AP@0.5", "AP@0.5:0.95"]]
    for model_name in ("faster_rcnn", "retinanet", "rtdetr"):
        model = by_name.get(model_name)
        for class_name in CLASS_ORDER:
            result = model.get("per_class", {}).get(class_name) if model else None
            class_rows.append([MODEL_LABELS[model_name], class_name.replace("_", " ").title(),
                               fmt(result.get("AP50")) if result else "Pending",
                               fmt(result.get("AP50_95")) if result else "Pending"])
    add_table(story, class_rows, [45 * mm, 43 * mm, 30 * mm, 37 * mm], font_size=7.5)
    add_image(story, root / "outputs/model_runs/speed_accuracy.png", width=145 * mm, max_height=85 * mm)

    curves = [root / "outputs/model_runs" / name / "loss_curves.png"
              for name in ("faster_rcnn", "retinanet", "rtdetr")]
    if any(path.exists() for path in curves):
        story.append(PageBreak())
    para(story, "6. Training and Convergence", styles["H1x"])
    para(story, "The training and validation curves show how the recorded objective changed across epochs. Loss values have model-specific definitions and should be compared as trends within each model, not as directly equivalent scales.", styles["Bodyx"])
    if ready:
        para(story, convergence_summary(root), styles["Bodyx"])
    for model_name, curve in zip(("faster_rcnn", "retinanet", "rtdetr"), curves):
        if curve.exists():
            para(story, MODEL_LABELS[model_name], styles["H2x"])
            add_image(story, curve, width=120 * mm, max_height=48 * mm)

    boards = [name for name in ("faster_rcnn", "retinanet", "rtdetr")
              if (root / "outputs/model_runs" / name / "qualitative_examples.png").exists()]
    if boards:
        story.append(PageBreak())
    para(story, "7. Detection Examples and Failure Analysis", styles["H1x"])
    para(story, "Each board uses validation images. Green boxes are ground truth and red boxes are predictions. Success examples contain a correct match with at least 50% recall and 50% precision at IoU 0.5; the displayed prediction threshold is 0.25. Failure examples use the same rule and are not swapped for easier images.", styles["Bodyx"])
    for model_index, model_name in enumerate(boards):
        board = root / "outputs/model_runs" / model_name / "qualitative_examples.png"
        if model_index > 0:
            story.append(PageBreak())
        para(story, MODEL_LABELS[model_name] + " validation examples", styles["H2x"])
        add_image(story, board, width=165 * mm, max_height=170 * mm)
        selection = load_json(root / "outputs/model_runs" / model_name / "qualitative_selection.json")
        if selection and selection.get("selected"):
            case_rows = [["Type", "Image", "TP", "FP", "FN"]]
            for entry in selection["selected"]:
                if isinstance(entry, list) and len(entry) == 2:
                    item, kind = entry
                    case_rows.append([kind, item.get("name", "-"), item.get("tp", 0), item.get("fp", 0), item.get("fn", 0)])
            if len(case_rows) > 1:
                add_table(story, case_rows, [22 * mm, 94 * mm, 15 * mm, 15 * mm, 15 * mm], font_size=7)

    if boards:
        story.append(PageBreak())
    para(story, "8. Comparison and Deployment Discussion", styles["H1x"])
    if ready:
        fastest = max(models, key=lambda item: item.get("fps", 0))
        most_accurate = max(models, key=lambda item: item.get("mAP50_95", 0) or 0)
        para(story, f"The fastest model in this run was {MODEL_LABELS[fastest['model']]} at {fastest['fps']:.1f} FPS. The highest mAP@0.5:0.95 was from {MODEL_LABELS[most_accurate['model']]} ({most_accurate['mAP50_95']:.3f}). The fastest model is the initial candidate for real-time site monitoring, while the most accurate is the initial candidate for offline batch analysis. This depends on whether its class-wise scores and failure cases meet the use-case requirements.", styles["Bodyx"])
    else:
        para(story, "The speed-accuracy comparison is incomplete until all three model metrics are available.", styles["Callout"])
    para(story, "Observed in the displayed validation examples: Faster R-CNN detects the excavator in 00002.jpg (TP 2, FP 2, FN 0) and the dozer in 00028.jpg (TP 1, FP 1, FN 0), while the crowded or overlapping scenes 04460.jpg, 05513.jpg, and 09385.jpg contain extra predictions and missed objects. RetinaNet detects the isolated dozer in 00028.jpg, but the crowded scenes 07480.jpg and 08970.jpg contain several false boxes and missed objects. RT-DETR produces no correct match for the dump truck in 06108.jpg (11 false positives and one missed object); the crowded scenes 05513.jpg and 02159.jpg also show overlapping extra boxes and missed instances.", styles["Bodyx"])
    para(story, "Possible explanations to investigate: small objects at 640-pixel input resolution may be harder to localize; CNN anchor sizes may not fit the observed box shapes; the class distribution is imbalanced; and overlap or NMS may contribute to duplicate or suppressed detections. These are hypotheses, not causes established by this experiment.", styles["Bodyx"])
    if missing_outputs:
        para(story, "Required outputs still to add", styles["H2x"])
        para(story, "; ".join(missing_outputs) + ".", styles["Callout"])
        if "Faster R-CNN qualitative_examples.png" in missing_outputs:
            para(story, "Run the Faster R-CNN examples-only recovery cell in the Colab notebook. It uses the saved checkpoint and preserves the completed metrics.", styles["Bodyx"])

    para(story, "9. Limitations", styles["H1x"])
    para(story, "The experiment uses a single random image split rather than grouping images by scene, so similar views may appear in more than one subset. The epoch budget is short, and the model families use different training implementations and optimization settings. FPS depends on the GPU, software versions, and inference path. The reported metrics are validation results; the held-out test split was not used for the comparison.", styles["Bodyx"])

    para(story, "10. Conclusion", styles["H1x"])
    if ready:
        fastest = max(models, key=lambda item: item.get("fps", 0))
        most_accurate = max(models, key=lambda item: item.get("mAP50_95", 0) or 0)
        para(story, f"On this validation split, {MODEL_LABELS[most_accurate['model']]} achieved the highest mAP@0.5:0.95 ({most_accurate['mAP50_95']:.3f}), while {MODEL_LABELS[fastest['model']]} had the highest measured speed ({fastest['fps']:.1f} FPS). These measurements give an accuracy-oriented and a speed-oriented option. The selected class-wise AP and failure examples should guide the final choice for a specific monitoring or batch-analysis system.", styles["Bodyx"])
    else:
        para(story, "The final model choice will be written after all three validation runs are available.", styles["Bodyx"])

    para(story, "11. References", styles["H1x"])
    references = [
        "Ren, S. et al. 'Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.' NeurIPS, 2015. https://papers.nips.cc/paper_files/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html",
        "Lin, T.-Y. et al. 'Focal Loss for Dense Object Detection.' ICCV, 2017. https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_Dense_Object_Detection_ICCV_2017_paper.pdf",
        "Zhao, Y. et al. 'DETRs Beat YOLOs on Real-time Object Detection.' CVPR, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html",
        "ACID dataset: link provided in the CV5570 Assignment 2 brief. https://drive.google.com/uc?id=1Qg_X5FygUMBRTcVFPb0s1fQP-20f8n8O",
        "Ultralytics. 'RT-DETR.' https://docs.ultralytics.com/models/rtdetr",
    ]
    for index, reference in enumerate(references, start=1):
        para(story, f"[{index}] {escape(reference)}", styles["Smallx"])

    def page_footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D5DDE5"))
        canvas.line(17 * mm, 12 * mm, A4[0] - 17 * mm, 12 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#657487"))
        canvas.drawString(17 * mm, 7 * mm, "CV5570 Assignment 2 | ACID detector comparison")
        canvas.drawRightString(A4[0] - 17 * mm, 7 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.multiBuild(story, onFirstPage=page_footer, onLaterPages=page_footer)

def main():
    parser = argparse.ArgumentParser(description="Generate the Assignment 2 PDF and editable Markdown report.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root)
    dataset = load_json(root / "outputs/dataset_analysis/dataset_summary.json")
    if dataset is None:
        raise FileNotFoundError("Run src/data_tools.py before building the report.")
    metrics = collect_metrics(root)
    ready = metrics is not None and len(metrics.get("models", [])) == 3
    complete = ready and not missing_submission_outputs(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = "CV5570_Assignment2_Final_Report" if complete else "CV5570_Assignment2_Report_DRAFT"
    build_markdown(dataset, metrics, reports / f"{stem}.md")
    build_pdf(root, reports / f"{stem}.pdf", dataset, metrics)
    print(f"Created {reports / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
