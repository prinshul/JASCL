from pathlib import Path
import json, glob
import random
import os
from collections import defaultdict
import math

random.seed(42)

SPLIT_PLAN = {
    "base": {"painting": [ 'bicycle',  'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']},
    "step_1": {"weather": ['person','car','bench']},
    "step_2": {"handmake": ['motorcycle','laptop']}
}

FEWSHOT_RATIOS = {
    "step_1": 0.2,
    "step_2": 0.5
}

SPLIT_RATIOS = {
    "base":   {"train_ratio": 0.6, "val_ratio": 0.2},
    "step_1": {"train_ratio": 0.5, "val_ratio": 0.2},
    "step_2": {"train_ratio": 0.6, "val_ratio": 0.2}
}


def load_ann(ann_file):
    with open(ann_file, "r") as f:
        return json.load(f)

def filter_by_classes(data, keep_classes):
    keep_cat_ids = [c["id"] for c in data["categories"] if c["name"] in keep_classes]
    categories = [c for c in data["categories"] if c["id"] in keep_cat_ids]

    keep_img_ids = set()
    annos = []
    for a in data["annotations"]:
        if a["category_id"] in keep_cat_ids:
            annos.append(a)
            keep_img_ids.add(a["image_id"])

    images = [img for img in data["images"] if img["id"] in keep_img_ids]
    return {"images": images, "annotations": annos, "categories": categories}

def print_split_stats(splits, unlabeled_count_per_class=None):
    """Calculates and prints class-wise image counts for all splits, including unlabeled."""
    print("\n--- Class-wise Split Statistics ---")
    
    if not splits.get("train", {}).get("categories"):
        print("No categories found to print stats."); return
        
    cat_id_to_name = {cat["id"]: cat["name"] for cat in splits["train"]["categories"]}
    
    stats = defaultdict(lambda: defaultdict(set))
    for split_name, data in splits.items():
        if "annotations" not in data: continue
        for ann in data["annotations"]:
            class_name = cat_id_to_name.get(ann["category_id"])
            if class_name:
                stats[split_name][class_name].add(ann["image_id"])

    header = f"{'CLASS':<20} | {'TRAIN':>10} | {'VAL':>10} | {'TEST':>10} | {'UNLABELED':>10} | {'TOTAL':>10}"
    print(header)
    print("-" * len(header))

    all_class_names = {cat['name'] for cat in splits['train']['categories']}

    for class_name in sorted(list(all_class_names)):
        train_count = len(stats["train"].get(class_name, set()))
        val_count = len(stats["val"].get(class_name, set()))
        test_count = len(stats["test"].get(class_name, set()))
        unlabeled_count = unlabeled_count_per_class.get(class_name, 0) if unlabeled_count_per_class else 0
        total_count = train_count + val_count + test_count + unlabeled_count
        print(f"{class_name:<20} | {train_count:>10} | {val_count:>10} | {test_count:>10} | {unlabeled_count:>10} | {total_count:>10}")

def split_train_val_test_stratified(data, train_ratio=0.6, val_ratio=0.2):
    """Performs a stratified split to maintain class proportions across train, val, and test sets."""
    class_to_images = defaultdict(list)
    for ann in data["annotations"]:
        class_to_images[ann["category_id"]].append(ann["image_id"])

    for cat_id, img_ids in class_to_images.items():
        class_to_images[cat_id] = list(set(img_ids))

    train_img_ids, val_img_ids, test_img_ids = set(), set(), set()

    for cat_id, img_ids in class_to_images.items():
        random.shuffle(img_ids)
        n_total = len(img_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_img_ids.update(img_ids[:n_train])
        val_img_ids.update(img_ids[n_train : n_train + n_val])
        test_img_ids.update(img_ids[n_train + n_val :])

    splits = {}
    all_image_map = {i['id']: i for i in data['images']}
    
    for name, ids_set in zip(["train", "val", "test"], [train_img_ids, val_img_ids, test_img_ids]):
        subset_imgs = [all_image_map[i_id] for i_id in ids_set if i_id in all_image_map]
        annos = [a for a in data["annotations"] if a["image_id"] in ids_set]
        splits[name] = {"images": subset_imgs, "annotations": annos, "categories": data["categories"]}
    return splits


def fewshot_split(data, fewshot_ratio=0.2):
    images = data["images"]
    random.shuffle(images)
    n_fs = max(1, int(len(images) * fewshot_ratio))

    labeled_imgs, unlabeled_imgs = images[:n_fs], images[n_fs:]
    
    labeled_ids = {i["id"] for i in labeled_imgs}
    unlabeled_ids = {i["id"] for i in unlabeled_imgs}

    labeled_annos = [a for a in data["annotations"] if a["image_id"] in labeled_ids]
    unlabeled_annos = [a for a in data["annotations"] if a["image_id"] in unlabeled_ids]

    return {
        "labeled": {"images": labeled_imgs, "annotations": labeled_annos, "categories": data["categories"]},
        "unlabeled": {"images": unlabeled_imgs, "annotations": unlabeled_annos, "categories": data["categories"]}
    }
    
def split_step1_custom(data):
    """
    Custom function for step_1:
    - For 'bench', splits into fixed numbers: 30 train, 10 val, 20 test, 41 unlabeled.
    - For other classes, splits into 30 train and distributes the rest by a 10:20:41 ratio.
    """
    print("Executing custom split for step_1 with fixed and proportional logic.")
    
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    id_to_cat_name = {v: k for k, v in cat_name_to_id.items()}
    bench_cat_id = cat_name_to_id.get('bench')

    class_to_images = defaultdict(set)
    for ann in data["annotations"]:
        class_to_images[ann["category_id"]].add(ann["image_id"])

    train_imgs_per_class = defaultdict(set)
    val_imgs_per_class = defaultdict(set)
    test_imgs_per_class = defaultdict(set)
    unlabeled_imgs_per_class = defaultdict(set)

    for cat_id, all_imgs_set in class_to_images.items():
        all_imgs = list(all_imgs_set)
        random.shuffle(all_imgs)
        n_total = len(all_imgs)

        if cat_id == bench_cat_id:
            print(f"Applying manual split for 'bench': 30 train, 10 val, 20 test, 41 unlabeled.")
            n_train, n_val, n_test = 30, 10, 20
        else:
            n_train = 30
            n_rem = n_total - n_train
            ratio_sum = 10 + 20 + 41
            n_val = round(n_rem * (10 / ratio_sum))
            n_test = round(n_rem * (20 / ratio_sum))
            class_name = id_to_cat_name.get(cat_id, 'Unknown')
            print(f"Splitting '{class_name}': {n_train} train, and remainder into val:{n_val}, test:{n_test}, and unlabeled.")

        train_imgs_per_class[cat_id].update(all_imgs[:n_train])
        val_imgs_per_class[cat_id].update(all_imgs[n_train : n_train + n_val])
        test_imgs_per_class[cat_id].update(all_imgs[n_train + n_val : n_train + n_val + n_test])
        unlabeled_imgs_per_class[cat_id].update(all_imgs[n_train + n_val + n_test:])

    master_train_ids, master_val_ids, master_test_ids, master_unlabeled_ids = set(), set(), set(), set()
    for cid in class_to_images:
        master_train_ids.update(train_imgs_per_class[cid])
        master_val_ids.update(val_imgs_per_class[cid])
        master_test_ids.update(test_imgs_per_class[cid])
        master_unlabeled_ids.update(unlabeled_imgs_per_class[cid])

    train_annos = [ann for ann in data['annotations'] if ann['image_id'] in train_imgs_per_class.get(ann['category_id'], set())]
    val_annos = [ann for ann in data['annotations'] if ann['image_id'] in val_imgs_per_class.get(ann['category_id'], set())]
    test_annos = [ann for ann in data['annotations'] if ann['image_id'] in test_imgs_per_class.get(ann['category_id'], set())]
    unlabeled_annos = [ann for ann in data['annotations'] if ann['image_id'] in unlabeled_imgs_per_class.get(ann['category_id'], set())]
    
    splits = {}
    all_image_map = {i['id']: i for i in data['images']}
    
    splits['train'] = {"images": [all_image_map[i] for i in master_train_ids], "annotations": train_annos, "categories": data["categories"]}
    splits['val'] = {"images": [all_image_map[i] for i in master_val_ids], "annotations": val_annos, "categories": data["categories"]}
    splits['test'] = {"images": [all_image_map[i] for i in master_test_ids], "annotations": test_annos, "categories": data["categories"]}
    unlabeled_data = {"images": [all_image_map[i] for i in master_unlabeled_ids], "annotations": unlabeled_annos, "categories": data["categories"]}

    unlabeled_counts = {id_to_cat_name[cid]: len(img_set) for cid, img_set in unlabeled_imgs_per_class.items()}

    return splits, unlabeled_data, unlabeled_counts

def main(root="ood_coco", out_root="ood_coco_splits"):
    os.makedirs(out_root, exist_ok=True)
    print("Data path = ",root)
    for step, domain_info in SPLIT_PLAN.items():
        for domain, n_classes in domain_info.items():
            ann_file = Path(root) / domain / "annotations/instances_val2017.json"
            print(f"\nProcessing: {ann_file}")
            data = load_ann(str(ann_file))
            
            print("Selecting classes ", n_classes)
            filtered = filter_by_classes(data, n_classes)

            step_dir = Path(out_root) / step
            os.makedirs(step_dir, exist_ok=True)

            if step == "base":
                ratios = SPLIT_RATIOS.get(step, {})
                splits = split_train_val_test_stratified(filtered, **ratios)
                print_split_stats(splits)

                for split_name, split_data in splits.items():
                    # FIX: Add original metadata back before saving
                    split_data['info'] = data.get('info', {})
                    split_data['licenses'] = data.get('licenses', [])
                    with open(step_dir / f"{domain}_{split_name}.json", "w") as f:
                        json.dump(split_data, f)
                print(f"[{step}] {domain} -> {len(n_classes)} classes split and saved.\n")

            elif step == "step_1":
                splits, unlabeled_data, unlabeled_counts = split_step1_custom(filtered)
                
                # FIX: Add original metadata back before saving
                unlabeled_data['info'] = data.get('info', {})
                unlabeled_data['licenses'] = data.get('licenses', [])
                with open(step_dir / f"{domain}_unlabeled.json", "w") as f:
                    json.dump(unlabeled_data, f)
                
                print_split_stats(splits, unlabeled_counts)

                for split_name, split_data in splits.items():
                    # FIX: Add original metadata back before saving
                    split_data['info'] = data.get('info', {})
                    split_data['licenses'] = data.get('licenses', [])
                    with open(step_dir / f"{domain}_{split_name}.json", "w") as f:
                        json.dump(split_data, f)
                print(f"\n[{step}] {domain} -> {len(n_classes)} classes (custom fixed/proportional split) saved.\n")
                
            else: # Logic for step_2 or any other future steps
                current_fewshot_ratio = FEWSHOT_RATIOS.get(step)
                if current_fewshot_ratio is None:
                    print(f"Warning: No few-shot ratio found for '{step}'."); continue
                
                sup_unsup = fewshot_split(filtered, fewshot_ratio=current_fewshot_ratio)
                
                # FIX: Add original metadata back before saving
                sup_unsup["unlabeled"]['info'] = data.get('info', {})
                sup_unsup["unlabeled"]['licenses'] = data.get('licenses', [])
                with open(step_dir / f"{domain}_unlabeled.json", "w") as f:
                    json.dump(sup_unsup["unlabeled"], f)
                
                ratios = SPLIT_RATIOS.get(step, {})
                splits = split_train_val_test_stratified(sup_unsup["labeled"], **ratios)
                print_split_stats(splits)
                
                for split_name, split_data in splits.items():
                    # FIX: Add original metadata back before saving
                    split_data['info'] = data.get('info', {})
                    split_data['licenses'] = data.get('licenses', [])
                    with open(step_dir / f"{domain}_{split_name}.json", "w") as f:
                        json.dump(split_data, f)

                print(f"[{step}] {domain} -> {len(n_classes)} classes (few-shot + unlabeled) split and saved.\n")
    
if __name__ == "__main__":
    data_path = "<Path_to_Your_Root_data>/Detection_data/ood_coco/"
    main(root=data_path)