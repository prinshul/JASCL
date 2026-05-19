import argparse
import os
import pickle
import torch
import torch.nn.functional as F
from collections import defaultdict

from mmengine.config import Config
from mmengine.registry import build_from_cfg
from mmengine.runner import load_checkpoint
from mmdet.utils import register_all_modules
from tqdm import tqdm

def save_prototypes(config_path, checkpoint_path, output_path):
    """
    Extracts and saves feature prototypes for each class from a trained
    object detection model using the MMDetection 3.x API.
    """
    register_all_modules()
    print("--- Starting Prototype Generation ---")
    from mmdet.registry import DATASETS, MODELS

    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)
    
    if 'init_cfg' in cfg.model.backbone and cfg.model.backbone.init_cfg is not None:
         if isinstance(cfg.model.backbone.init_cfg, dict) and cfg.model.backbone.init_cfg.get('type') == 'Pretrained':
              cfg.model.backbone.init_cfg = None
         elif isinstance(cfg.model.backbone.init_cfg, list):
             for item in cfg.model.backbone.init_cfg:
                 if item.get('type') == 'Pretrained':
                     item.pop('checkpoint', None)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = build_from_cfg(cfg.model, MODELS)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    roi_extractor = build_from_cfg(cfg.model.roi_head.bbox_roi_extractor, MODELS)
    roi_extractor.to(device)

    train_dataloader_cfg = cfg.train_dataloader
    train_dataset = build_from_cfg(train_dataloader_cfg.dataset, DATASETS)
    
    from mmengine.dataset import default_collate
    
    loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1, 
        sampler=torch.utils.data.SequentialSampler(train_dataset),
        num_workers=2,
        collate_fn=default_collate
    )

    print("Extracting features from the dataset...")
    class_features = defaultdict(list)
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Processing Images")):
            
            processed_data = model.data_preprocessor(data, training=False)
            feature_maps = model.extract_feat(processed_data['inputs'])
            data_sample = processed_data['data_samples'][0]
            
            gt_instances = data_sample.gt_instances
            gt_bboxes = gt_instances.bboxes
            gt_labels = gt_instances.labels

            if len(gt_bboxes) == 0:
                continue

            rois = torch.cat([torch.zeros(gt_bboxes.size(0), 1).to(device), gt_bboxes], dim=1)
            
            pooled_features = roi_extractor(feature_maps[:roi_extractor.num_inputs], rois)
            
            bbox_head = model.roi_head.bbox_head
            bbox_feats_flat = pooled_features.view(pooled_features.size(0), -1)

            instance_features = bbox_feats_flat
            for fc in bbox_head.shared_fcs:
                instance_features = fc(instance_features)

            if i == 0: 
                print("\n--- save_prototypes.py: DEBUG SHAPES ---")
                print(f"Shape after RoI pooling (pooled_features): {pooled_features.shape}")
                print(f"Shape after flattening (bbox_feats_flat): {bbox_feats_flat.shape}")
                print(f"Final instance feature shape (post-FC layers): {instance_features.shape}")

            for inst_idx, label in enumerate(gt_labels):
                class_features[label.item()].append(instance_features[inst_idx].cpu())

    print("Calculating final class prototypes...")
    prototypes = {}
    for class_id, features_list in class_features.items():
        if features_list:
            all_features = torch.stack(features_list, dim=0)
            class_prototype = all_features.mean(dim=0)
            class_prototype = F.normalize(class_prototype, p=2, dim=0)
            prototypes[class_id] = class_prototype

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'wb') as f:
        pickle.dump(prototypes, f)

    print(f" Prototypes successfully saved to: {output_path}")
    print(f"Found prototypes for {len(prototypes)} classes.")

def main():
    parser = argparse.ArgumentParser(description='Save feature prototypes from a detection model.')
    parser.add_argument('config', help='path to the model config file')
    parser.add_argument('checkpoint', help='path to the model checkpoint file')
    parser.add_argument('--out', default='prototypes_base.pkl', help='path to save the output pickle file')
    args = parser.parse_args()

    save_prototypes(args.config, args.checkpoint, args.out)

if __name__ == '__main__':
    main()