from __future__ import annotations
from typing import Any, Dict, List, Tuple
import copy

from mmengine.dataset import BaseDataset, force_full_init
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class SemiCocoPairDataset(BaseDataset):
    """
    Pair labeled COCO sample with an unlabeled COCO sample.

    Expects two inner datasets in COCO format:
      - sup: standard CocoDataset (with annotations)
      - unsup: CocoDataset (can have empty annotations)
    Each inner dataset receives its pipeline (we set them via the dataset config).

    Returned dict per index:
      {
        'sup_data': <processed sample dict for student supervised branch>,
        'unsup_data': <processed sample dict for student unsupervised branch>,
        'metainfo': <dataset metainfo>
      }
    """

    def __init__(self,
                 sup: Dict[str, Any],
                 unsup: Dict[str, Any],
                 sup_pipeline: List[dict],
                 unsup_pipeline: List[dict],
                 pair_mode: str = 'cycle',  # 'zip' or 'cycle' (repeat shorter)
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert pair_mode in {'zip', 'cycle'}
        self.pair_mode = pair_mode

        # build coco datasets
        self.sup = CocoDataset(**sup)
        self.unsup = CocoDataset(**unsup)

        # assign pipelines (these are lists of pipeline transforms)
        self.sup.pipeline = self.build_pipeline(sup_pipeline)
        self.unsup.pipeline = self.build_pipeline(unsup_pipeline)

        # metadata
        self._metainfo = self.sup.metainfo

    @force_full_init
    def full_init(self):
        self.sup.full_init()
        self.unsup.full_init()
        if self.pair_mode == 'zip':
            self._length = min(len(self.sup), len(self.unsup))
        else:
            self._length = max(len(self.sup), len(self.unsup))

    def __len__(self) -> int:
        if not self._fully_initialized:
            return 1
        return self._length

    def _index_map(self, idx: int) -> Tuple[int, int]:
        if self.pair_mode == 'zip':
            return idx, idx
        return idx % len(self.sup), idx % len(self.unsup)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sup_idx, unsup_idx = self._index_map(idx)
        sup_data = copy.deepcopy(self.sup[sup_idx])
        unsup_data = copy.deepcopy(self.unsup[unsup_idx])

        # tag branches for downstream clarity
        sup_data['branch'] = 'sup'
        unsup_data['branch'] = 'unsup'

        return {
            'sup_data': sup_data,
            'unsup_data': unsup_data,
            'metainfo': self._metainfo
        }
