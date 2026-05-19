"""
Merge multiple COCO-format annotation JSONs into one combined COCO JSON.

For incremental step evaluation.

Fix for duplicate image file names:
 - Each input JSON gets a unique prefix (by default derived from its file stem, e.g. 'cartoon' from 'cartoon_test.json').
 - That prefix is prepended to image['file_name'], so collisions are avoided.

Usage:
  python tools/merge_coco_jsons.py \
      --ann cartoon_test.json sketch_test.json \
      --out combined_test.json \
      --root-prefix /absolute/path/to/ood_coco/
"""

import argparse
import json
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict


def merge_coco(ann_files, out_file, root_prefix=None, keep_info_from_first=True):
    combined = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    catname2id = OrderedDict()
    next_img_id = 1
    next_ann_id = 1

    for i, fpath in enumerate(ann_files):
        fpath = Path(fpath)
        if not fpath.exists():
            raise FileNotFoundError(f"Annotation file not found: {fpath}")

        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if i == 0 and keep_info_from_first:
            combined["info"] = data.get("info", {})
            combined["licenses"] = data.get("licenses", [])

        # Prefix (use file stem like 'cartoon' if available)
        prefix = fpath.stem.split("_")[0]  # e.g. "cartoon_test.json" -> "cartoon"

        # Handle categories
        oldcatid2name = {}
        for c in data.get("categories", []):
            oldcatid2name[c["id"]] = c["name"]
            if c["name"] not in catname2id:
                catname2id[c["name"]] = len(catname2id) + 1

        # Map images
        img_old2new = {}
        for img in data.get("images", []):
            old_id = img["id"]
            new_id = next_img_id
            next_img_id += 1
            img_old2new[old_id] = new_id

            new_img = deepcopy(img)
            new_img["id"] = new_id

            fn = new_img.get("file_name", "")
            # Add domain prefix (cartoon/, sketch/, etc.)
            new_fn = f"{prefix}/{'val2017'}/{fn}"

            if root_prefix:
                new_img["file_name"] = str(Path(root_prefix) / new_fn)
            else:
                new_img["file_name"] = new_fn

            combined["images"].append(new_img)

        # Map annotations
        for ann in data.get("annotations", []):
            new_ann = deepcopy(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = img_old2new[ann["image_id"]]

            cat_name = oldcatid2name[ann["category_id"]]
            new_ann["category_id"] = catname2id[cat_name]

            combined["annotations"].append(new_ann)

    combined["categories"] = [
        {"id": cid, "name": cname} for cname, cid in catname2id.items()
    ]

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fo:
        json.dump(combined, fo)

    print(f"[OK] Wrote combined COCO JSON → {out_path}")
    print(f"Images: {len(combined['images'])}, Annotations: {len(combined['annotations'])}, Categories: {len(combined['categories'])}")


def main():
    parser = argparse.ArgumentParser(description="Merge COCO JSON files into one combined JSON.")
    parser.add_argument('--ann', nargs='+', required=True, help='input COCO annotation json files')
    parser.add_argument('--out', required=True, help='output combined json path')
    parser.add_argument('--root-prefix', default=None,
                        help='optional root path to prepend to each file_name')
    args = parser.parse_args()

    merge_coco(args.ann, args.out, root_prefix=args.root_prefix)


if __name__ == "__main__":
    main()
