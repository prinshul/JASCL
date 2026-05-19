# mmdet/datasets/unlabeled_coco.py
from typing import List, Dict, Any
from mmengine.registry import DATASETS
from .coco import CocoDataset
from mmdet.datasets.api_wrappers import COCO


@DATASETS.register_module()
class UnlabeledCocoDataset(CocoDataset):
    """COCO-style dataset but treats ann_file as images+categories only.
    Accepts JSON that may have empty 'annotations'. Loads images and categories
    and builds data_list but with empty 'instances' for each image so pipelines that
    expect no GT will still work.
    """

    def load_annotations(self, ann_file: str) -> List[Dict[str, Any]]:
        """Load images + categories from COCO ann_file but ignore (or accept empty) annotations."""
        coco = COCO(ann_file)
        img_ids = coco.get_img_ids()
        cats = coco.load_cats(coco.get_cat_ids())
        categories = [{'id': c['id'], 'name': c['name']} for c in cats]

        data_list = []
        for img_id in img_ids:
            img_info = coco.load_imgs([img_id])[0]
            # build entry similar to CocoDataset expectations
            info = {
                'id': img_info['id'],
                'file_name': img_info.get('file_name'),
                'height': img_info.get('height', 0),
                'width': img_info.get('width', 0),
                # no 'annotations' for unlabeled
                'instances': []  # empty list; later pipeline will see no gt
            }
            data_list.append(info)

        # Save categories to metainfo if needed by dataset (CocoDataset already handles categories)
        # return list of image dicts
        return data_list
