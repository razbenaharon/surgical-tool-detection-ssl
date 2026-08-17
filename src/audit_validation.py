"""Audit suspicious YOLO validation annotations without modifying source data.

The script first creates annotated visualizations and a machine-readable audit.
With ``--build-versions`` it also creates two independent dataset snapshots:

* val_original: byte-for-byte copies of the official validation labels/images.
* val_clean: the same snapshot with only explicitly supplied 1-based rows removed.

Existing outputs are never silently overwritten; pass ``--force`` explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2


CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}
CLASS_COLORS = {0: (0, 200, 0), 1: (255, 128, 0), 2: (60, 60, 255)}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_image(images_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for {stem} under {images_dir}")


def read_yolo(label_path: Path) -> list[dict]:
    rows = []
    for row_number, raw in enumerate(label_path.read_text().splitlines(), 1):
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{row_number}: expected 5 fields")
        cls_id = int(parts[0])
        xc, yc, width, height = map(float, parts[1:])
        rows.append(
            {
                "row": row_number,
                "raw": raw,
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "x_center": xc,
                "y_center": yc,
                "width": width,
                "height": height,
                "area": width * height,
                "aspect_ratio": width / height if height else None,
            }
        )
    return rows


def pixel_box(row: dict, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    xc = row["x_center"] * image_width
    yc = row["y_center"] * image_height
    width = row["width"] * image_width
    height = row["height"] * image_height
    return (
        round(xc - width / 2),
        round(yc - height / 2),
        round(xc + width / 2),
        round(yc + height / 2),
    )


def prepare_out_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def draw_audit(image, rows: list[dict], out_dir: Path, stem: str) -> None:
    height, width = image.shape[:2]
    overlay = image.copy()
    for row in rows:
        x1, y1, x2, y2 = row["pixel_box"]
        color = CLASS_COLORS.get(row["class_id"], (255, 255, 255))
        line_width = 8 if row["is_suspicious"] else 5
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, line_width)
        label = f"row {row['row']}: {row['class_name']} {x2-x1}x{y2-y1}px"
        label_y = max(35, y1 - 12)
        cv2.putText(
            overlay,
            label,
            (max(0, x1), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / f"{stem}_all_annotations.jpg"), overlay)

    for row in rows:
        if not row["is_suspicious"]:
            continue
        x1, y1, x2, y2 = row["pixel_box"]
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        pad = max(120, 10 * max(box_w, box_h))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left, top = max(0, cx - pad), max(0, cy - pad)
        right, bottom = min(width, cx + pad), min(height, cy + pad)
        crop = image[top:bottom, left:right].copy()
        local_1 = (x1 - left, y1 - top)
        local_2 = (x2 - left, y2 - top)
        color = CLASS_COLORS.get(row["class_id"], (255, 255, 255))
        cv2.rectangle(crop, local_1, local_2, color, 5)
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        cv2.putText(
            crop,
            f"row {row['row']}: {row['class_name']} ({box_w}x{box_h}px)",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"{stem}_row_{row['row']}_zoom.jpg"), crop)


def copy_validation_snapshot(
    labeled_root: Path,
    version_root: Path,
    removed_rows_by_stem: dict[str, set[int]],
) -> dict:
    src_images = labeled_root / "images" / "val"
    src_labels = labeled_root / "labels" / "val"
    dst_images = version_root / "images" / "val"
    dst_labels = version_root / "labels" / "val"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    removals = []
    for image_path in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        shutil.copy2(image_path, dst_images / image_path.name)
    for label_path in sorted(src_labels.glob("*.txt")):
        raw_lines = label_path.read_text().splitlines()
        remove_rows = removed_rows_by_stem.get(label_path.stem, set())
        if not remove_rows:
            shutil.copy2(label_path, dst_labels / label_path.name)
            continue
        kept_lines = []
        for row_number, raw in enumerate(raw_lines, 1):
            if row_number in remove_rows:
                removals.append(
                    {"file": label_path.name, "row": row_number, "content": raw}
                )
            else:
                kept_lines.append(raw)
        output = "\n".join(kept_lines) + ("\n" if kept_lines else "")
        (dst_labels / label_path.name).write_text(output)

    return {
        "version_root": str(version_root.resolve()),
        "source_root": str(labeled_root.resolve()),
        "image_count": len(list(dst_images.iterdir())),
        "label_count": len(list(dst_labels.glob("*.txt"))),
        "removed_annotations": removals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-root", required=True, type=Path)
    parser.add_argument("--stem", default="ff8c22da-output_0182")
    parser.add_argument("--suspicious-rows", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--out", type=Path, default=Path("reports/validation_audit"))
    parser.add_argument("--versions-root", type=Path, default=Path("datasets/validation_versions"))
    parser.add_argument("--build-versions", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    labeled_root = args.labeled_root.resolve()
    source_label = labeled_root / "labels" / "val" / f"{args.stem}.txt"
    source_image = find_image(labeled_root / "images" / "val", args.stem)
    out_dir = args.out.resolve()
    prepare_out_dir(out_dir, args.force)

    image = cv2.imread(str(source_image))
    if image is None:
        raise RuntimeError(f"Could not read image: {source_image}")
    height, width = image.shape[:2]
    suspicious = set(args.suspicious_rows)
    rows = read_yolo(source_label)
    for row in rows:
        row["pixel_box"] = pixel_box(row, width, height)
        x1, y1, x2, y2 = row["pixel_box"]
        row["pixel_width"] = x2 - x1
        row["pixel_height"] = y2 - y1
        row["is_suspicious"] = row["row"] in suspicious

    draw_audit(image, rows, out_dir, args.stem)
    audit = {
        "source_image": str(source_image),
        "source_image_sha256": sha256(source_image),
        "source_label": str(source_label),
        "source_label_sha256": sha256(source_label),
        "image_width": width,
        "image_height": height,
        "suspicious_rows": sorted(suspicious),
        "annotations": rows,
    }

    if args.build_versions:
        versions_root = args.versions_root.resolve()
        original_root = versions_root / "val_original"
        clean_root = versions_root / "val_clean"
        prepare_out_dir(original_root, args.force)
        prepare_out_dir(clean_root, args.force)
        audit["val_original"] = copy_validation_snapshot(labeled_root, original_root, {})
        audit["val_clean"] = copy_validation_snapshot(
            labeled_root, clean_root, {args.stem: suspicious}
        )
        source_hashes = {
            path.name: sha256(path) for path in sorted((labeled_root / "labels" / "val").glob("*.txt"))
        }
        original_hashes = {
            path.name: sha256(path) for path in sorted((original_root / "labels" / "val").glob("*.txt"))
        }
        audit["val_original"]["labels_byte_identical_to_source"] = source_hashes == original_hashes
        source_image_hashes = {
            path.name: sha256(path)
            for path in sorted((labeled_root / "images" / "val").iterdir())
            if path.suffix.lower() in IMAGE_EXTS
        }
        original_image_hashes = {
            path.name: sha256(path)
            for path in sorted((original_root / "images" / "val").iterdir())
            if path.suffix.lower() in IMAGE_EXTS
        }
        audit["val_original"]["images_byte_identical_to_source"] = (
            source_image_hashes == original_image_hashes
        )

    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
