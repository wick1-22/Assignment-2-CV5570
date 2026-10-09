"""Dataset preparation and simple visual analysis for the ACID coursework."""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_ORDER = ["dozer", "dump_truck", "excavator"]
COLORS = {"dozer": (235, 85, 65), "dump_truck": (42, 145, 225), "excavator": (50, 175, 105)}


def load_coco(json_path):
    with open(json_path, "r", encoding="utf-8") as stream:
        coco = json.load(stream)
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    keep = {category_id: name for category_id, name in categories.items() if name.lower() in CLASS_ORDER}
    if len(keep) != 3:
        raise ValueError(f"Expected the three ACID classes {CLASS_ORDER}, got {list(keep.values())}")
    coco["categories"] = [item for item in coco["categories"] if item["id"] in keep]
    coco["annotations"] = [item for item in coco.get("annotations", []) if item["category_id"] in keep]
    return coco, keep


def split_coco(coco, seed=42):
    """Return a repeatable 70/20/10 image split and its annotations."""
    images = list(coco["images"])
    random.Random(seed).shuffle(images)
    n = len(images)
    n_train = int(n * 0.70)
    n_val = int(n * 0.20)
    partitions = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }
    result = {}
    for name, selected in partitions.items():
        image_ids = {image["id"] for image in selected}
        result[name] = {
            "images": selected,
            "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in image_ids],
            "categories": coco["categories"],
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
        }
    return result


def _font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_chart_axes(draw, width, height, title, x_label, y_label):
    left, top, right, bottom = 90, 72, width - 30, height - 72
    draw.text((left, 22), title, fill=(25, 35, 50), font=_font(22))
    draw.line((left, top, left, bottom, right, bottom), fill=(55, 65, 80), width=2)
    draw.text((left, height - 38), x_label, fill=(45, 50, 60), font=_font(15))
    draw.text((8, top - 20), y_label, fill=(45, 50, 60), font=_font(15))
    return left, top, right, bottom


def _bar_chart(values, title, x_label, y_label, path):
    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = _draw_chart_axes(draw, width, height, title, x_label, y_label)
    maximum = max(values.values()) if values else 1
    bar_width = (right - left) / max(len(values), 1) * 0.55
    gap = (right - left) / max(len(values), 1)
    for index, (label, value) in enumerate(values.items()):
        x0 = left + gap * index + (gap - bar_width) / 2
        y0 = bottom - (bottom - top) * value / maximum
        color = COLORS.get(label, (85, 130, 185))
        draw.rectangle((x0, y0, x0 + bar_width, bottom), fill=color)
        draw.text((x0, y0 - 25), str(value), fill=(35, 40, 50), font=_font(15))
        draw.text((x0 - 4, bottom + 9), label.replace("_", " "), fill=(45, 50, 60), font=_font(14))
    image.save(path)


def _histogram(values, title, x_label, path, bins=24, log_scale=False):
    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = _draw_chart_axes(draw, width, height, title, x_label, "Count")
    if not values:
        image.save(path)
        return
    plotted = [math.log10(max(value, 1e-9)) for value in values] if log_scale else values
    low, high = min(plotted), max(plotted)
    if high == low:
        high = low + 1
    counts = [0] * bins
    for value in plotted:
        index = min(bins - 1, int((value - low) / (high - low) * bins))
        counts[index] += 1
    peak = max(counts) or 1
    bar_width = (right - left) / bins
    for index, count in enumerate(counts):
        x0 = left + index * bar_width
        y0 = bottom - (bottom - top) * count / peak
        draw.rectangle((x0 + 1, y0, x0 + bar_width - 1, bottom), fill=(72, 133, 190))
    low_label = 10 ** low if log_scale else low
    high_label = 10 ** high if log_scale else high
    draw.text((left, bottom + 12), f"{low_label:.4g}", fill=(45, 50, 60), font=_font(13))
    draw.text((right - 50, bottom + 12), f"{high_label:.4g}", fill=(45, 50, 60), font=_font(13))
    image.save(path)


def _sample_board(coco, image_root, path, count=6, seed=42):
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    images = [item for item in coco["images"] if annotations[item["id"]]]
    random.Random(seed).shuffle(images)
    images = images[:count]
    cell_w, cell_h, header = 480, 410, 30
    board = Image.new("RGB", (cell_w * 2, (cell_h + header) * 3), "white")
    draw_board = ImageDraw.Draw(board)
    cat_names = {item["id"]: item["name"] for item in coco["categories"]}
    for index, metadata in enumerate(images):
        image_path = image_root / metadata["file_name"]
        with Image.open(image_path) as source:
            source = source.convert("RGB")
            source.thumbnail((cell_w - 12, cell_h - 12))
            tile = Image.new("RGB", (cell_w, cell_h), "#f1f3f5")
            x_offset = (cell_w - source.width) // 2
            y_offset = (cell_h - source.height) // 2
            tile.paste(source, (x_offset, y_offset))
            tile_draw = ImageDraw.Draw(tile)
            scale_x = source.width / metadata["width"]
            scale_y = source.height / metadata["height"]
            for ann in annotations[metadata["id"]]:
                x, y, w, h = ann["bbox"]
                x0 = x_offset + x * scale_x
                y0 = y_offset + y * scale_y
                x1 = x_offset + (x + w) * scale_x
                y1 = y_offset + (y + h) * scale_y
                name = cat_names[ann["category_id"]]
                tile_draw.rectangle((x0, y0, x1, y1), outline=COLORS.get(name, "red"), width=3)
                tile_draw.text((x0, max(0, y0 - 18)), name, fill=COLORS.get(name, "red"), font=_font(14))
            col, row = index % 2, index // 2
            x, y = col * cell_w, row * (cell_h + header)
            draw_board.text((x + 7, y + 5), f"Ground truth: {metadata['file_name']}", fill=(30, 40, 55), font=_font(15))
            board.paste(tile, (x, y + header))
    board.save(path)


def analyze_dataset(json_path, image_root, output_dir, seed=42):
    json_path, image_root, output_dir = Path(json_path), Path(image_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coco, category_map = load_coco(json_path)
    image_by_id = {item["id"]: item for item in coco["images"]}
    missing = [item["file_name"] for item in coco["images"] if not (image_root / item["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image files are missing; first example: {missing[0]}")

    counts = Counter(category_map[ann["category_id"]] for ann in coco["annotations"])
    areas, ratios = [], []
    for ann in coco["annotations"]:
        img = image_by_id[ann["image_id"]]
        _, _, box_w, box_h = ann["bbox"]
        areas.append((box_w * box_h) / (img["width"] * img["height"]))
        ratios.append(box_w / max(box_h, 1e-9))

    splits = split_coco(coco, seed)
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {}
    for split_name, split in splits.items():
        with open(split_dir / f"{split_name}.json", "w", encoding="utf-8") as stream:
            json.dump(split, stream)
        split_counts = Counter(category_map[ann["category_id"]] for ann in split["annotations"])
        split_rows[split_name] = {
            "images": len(split["images"]),
            "annotations": len(split["annotations"]),
            "class_instances": {name: split_counts.get(name, 0) for name in CLASS_ORDER},
        }

    _bar_chart({name: counts.get(name, 0) for name in CLASS_ORDER}, "ACID object instances by class", "Class", "Instances", output_dir / "class_frequencies.png")
    _histogram(areas, "Bounding-box area as fraction of image", "Normalized box area", output_dir / "bbox_area_distribution.png")
    _histogram(ratios, "Bounding-box width to height ratio", "Aspect ratio (log scale)", output_dir / "bbox_aspect_ratio.png", log_scale=True)
    _sample_board(coco, image_root, output_dir / "ground_truth_samples.png", count=6, seed=seed)

    summary = {
        "source_annotation": json_path.name,
        "image_root": "ACID three-class image directory",
        "image_count": len(coco["images"]),
        "instance_count": len(coco["annotations"]),
        "annotation_format": "COCO JSON; bounding boxes stored as [x, y, width, height] pixels",
        "class_instances": {name: counts.get(name, 0) for name in CLASS_ORDER},
        "split_seed": seed,
        "split_proportions": {"train": 0.70, "val": 0.20, "test": 0.10},
        "splits": split_rows,
        "bbox_area_fraction": {"min": min(areas), "median": sorted(areas)[len(areas) // 2], "max": max(areas)},
        "bbox_aspect_ratio": {"min": min(ratios), "median": sorted(ratios)[len(ratios) // 2], "max": max(ratios)},
        "class_id_to_name": {str(key): value for key, value in category_map.items()},
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze ACID COCO annotations and create deterministic splits.")
    parser.add_argument("--annotations", required=True, help="Path to 3classes.json")
    parser.add_argument("--images", required=True, help="Directory containing the image files")
    parser.add_argument("--output", default="outputs/dataset_analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = analyze_dataset(args.annotations, args.images, args.output, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
