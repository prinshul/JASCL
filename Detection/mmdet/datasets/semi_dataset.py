import copy
from .base_det_dataset import BaseDetDataset
from .coco import CocoDataset
from mmdet.registry import DATASETS
import pdb


from typing import Dict, Any, List
#from mmengine.registry import DATASETS
from mmengine.dataset import BaseDataset
from .unlabeled_coco import UnlabeledCocoDataset


@DATASETS.register_module()
class SemiDataset(BaseDataset):
    """
    Wrapper dataset that returns a dict with 'sup' and 'unsup' items per index.

    Config example (inside train_dataloader.dataset):
    dataset=dict(
        type='SemiCocoDataset',
        sup=dict(
            type='CocoDataset',
            ann_file='.../sketch_labeled_fewshot.json',
            data_prefix=dict(img='.../sketch/images/'),
            metainfo=dict(classes=(...)),
            pipeline=[...PackDetInputs...]
        ),
        unsup=dict(
            type='UnlabeledCocoDataset',
            ann_file='.../sketch_unlabeled.json',
            data_prefix=dict(img='.../sketch/images/'),
            metainfo=dict(classes=(...)),
            pipeline=[...PackDetInputs...]
        ),
        sup_repeat=1,
        unsup_repeat=1
    )
    """

    def __init__(self,
                 sup: Dict[str, Any],
                 unsup: Dict[str, Any],
                 sup_repeat: int = 1,
                 unsup_repeat: int = 1,
                 **kwargs):
        # We do not call super().__init__ with ann_file because this wrapper isn't a leaf dataset.
        super().__init__(**kwargs)

        # Build supervised dataset (must be valid CocoDataset config)
        sup_cfg = copy.deepcopy(sup)
        sup_cfg.pop('type', None)
        sup_ann = sup_cfg.pop('ann_file', None)
        sup_data_prefix = sup_cfg.pop('data_prefix', None)
        sup_pipeline = sup_cfg.pop('pipeline', None)
        sup_metainfo = sup_cfg.pop('metainfo', None)

        if sup_pipeline is None:
            raise ValueError("sup.pipeline must be provided and end with PackDetInputs")

        self.sup_dataset = CocoDataset(
            ann_file=sup_ann,
            data_prefix=sup_data_prefix,
            pipeline=sup_pipeline,
            metainfo=sup_metainfo
        )

        # Build unlabeled dataset. Expect UnlabeledCocoDataset or CocoDataset with empty annotations.
        unsup_cfg = copy.deepcopy(unsup)
        unsup_cfg.pop('type', None)
        unsup_ann = unsup_cfg.pop('ann_file', None)
        unsup_data_prefix = unsup_cfg.pop('data_prefix', None)
        unsup_pipeline = unsup_cfg.pop('pipeline', None)
        unsup_metainfo = unsup_cfg.pop('metainfo', None)

        if unsup_pipeline is None:
            raise ValueError("unsup.pipeline must be provided and end with PackDetInputs")

        # instantiate UnlabeledCocoDataset so empty annotations are acceptable
        self.unsup_dataset = UnlabeledCocoDataset(
            ann_file=unsup_ann,
            data_prefix=unsup_data_prefix,
            pipeline=unsup_pipeline,
            metainfo=unsup_metainfo
        )

        self.sup_repeat = max(1, int(sup_repeat))
        self.unsup_repeat = max(1, int(unsup_repeat))

        self._metainfo = copy.deepcopy(self.sup_dataset.metainfo)
        self._length = max(len(self.sup_dataset) * self.sup_repeat,
                           len(self.unsup_dataset) * self.unsup_repeat)

    @property
    def metainfo(self):
        return self._metainfo

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # round-robin sampling
        si = idx % (len(self.sup_dataset) * self.sup_repeat)
        ui = idx % (len(self.unsup_dataset) * self.unsup_repeat)
        sup_item = self.sup_dataset[si % len(self.sup_dataset)]
        unsup_item = self.unsup_dataset[ui % len(self.unsup_dataset)]
        # Each item is expected to be already processed by pipeline -> dict with 'inputs' and 'data_samples'
        return dict(sup=sup_item, unsup=unsup_item)









'''
@DATASETS.register_module()
class SemiDataset(CocoDataset):
    """Semi-supervised dataset wrapper for Mean Teacher.
    Returns both labeled (sup) and unlabeled (unsup) data.
    """

    def __init__(self,
                 sup:dict,
                 unsup:dict,
                 sup_repeat:int = 1,
                 unsup_repeat:int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        
        print('sup dataset')
        print(sup_cfg)
        # Build supervised dataset
        self.sup_dataset = CocoDataset(**sup)
        
        print('unsup_dataset')
        # Build unlabeled dataset
        self.unsup_dataset = CocoDataset(**unsup)
        
        print(sup)
        print(unsup)
        print(pdb.set_trace())
        # Build supervised dataset
        sup_cfg = copy.deepcopy(sup_cfg)
        sup_type = sup_cfg.pop("type")
        #sup_type = sup_type.pop("pipeline")
        if sup_type != "CocoDataset":
            raise ValueError(f"Expected CocoDataset for sup_cfg, got {sup_type}")
        self.sup_dataset = CocoDataset(**sup_cfg)

        # Build unlabeled dataset
        unsup_cfg = copy.deepcopy(unsup_cfg)
        unsup_type = unsup_cfg.pop("type")
        #unsup_type = unsup_type.pop("pipeline")
        if unsup_type != "CocoDataset":
            raise ValueError(f"Expected CocoDataset for unsup_cfg, got {unsup_type}")
        self.unsup_dataset = CocoDataset(**unsup_cfg)
        
        self.sup_repeat = sup_repeat
        self.unsup_repeat = unsup_repeat

        self._metainfo = copy.deepcopy(self.sup_dataset.metainfo)
        self._length = max(
            len(self.sup_dataset) * sup_repeat,
            len(self.unsup_dataset) * unsup_repeat
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        sup_idx = idx % (len(self.sup_dataset) * self.sup_repeat)
        sup_data = self.sup_dataset[sup_idx % len(self.sup_dataset)]

        unsup_idx = idx % (len(self.unsup_dataset) * self.unsup_repeat)
        unsup_data = self.unsup_dataset[unsup_idx % len(self.unsup_dataset)]

        return dict(
            sup=sup_data,     # labeled
            unsup=unsup_data  # unlabeled
        )

    @property
    def metainfo(self):
        return self._metainfo
'''