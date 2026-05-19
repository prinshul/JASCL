from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader, ConcatDataset

from datasets.lightning_data_module import LightningDataModule
from datasets.mappings import *
from datasets.dataset import Dataset
from datasets.transforms import Transforms
import pickle
import os
import sys
from datetime import datetime


class UrbanScenes(LightningDataModule):
    def __init__(
        self,
        root,
        devices,
        batch_size: int,
        img_size: tuple[int, int],
        num_workers: int,
        num_classes: int,
        num_metrics: int,
        ignore_idx: int,
        curr_step: int,
        scale_range=(0.5, 2.0),
    ) -> None:
        super().__init__(
            root=root,
            devices=devices,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            num_metrics=num_metrics,
            ignore_idx=ignore_idx,
            curr_step=curr_step,
            img_size=img_size,
        )
        print("Urban scenes - curr step ",self.curr_step)

        self.transforms = Transforms(self.img_size, scale_range)

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        '''
        gta5_train_datasets = [
            Dataset(
                zip_path=Path(self.root, f"{i:02}_images.zip"),
                target_zip_path=Path(self.root, f"{i:02}_labels.zip"),
                img_folder_path_in_zip=Path("./images"),
                target_folder_path_in_zip=Path("./labels"),
                img_suffix=".png",
                target_suffix=".png",
                class_mapping=get_cityscapes_mapping(),
                ignore_idx=self.ignore_idx,
                transforms=self.transforms,
            )
            for i in range(1, 11)
        ]
        self.gta5_train_dataset = ConcatDataset(gta5_train_datasets)
        
        
        step0_dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "img_stem_suffix": "images",
            "target_stem_suffix": "",
            "zip_path": Path(self.root, "images.zip"),
            "target_zip_path": Path(self.root, "labels.zip"),
            "class_mapping": get_base_mapping(),
        }
        '''
        
        BDD_datadir = '<Path_to_Your_Root_Data>/Natural_FoSSIL_data/step1/'
        IDD_datadir = '<Path_to_Your_Root_Data>/Natural_FoSSIL_data/step2/'
        step3_datadir = '<Path_to_Your_Root_Data>/Natural_FoSSIL_data/step3/'

        shots_datadir = '<Path_to_Your_Root_Data>/Natural_FoSSIL_data/shots/'
        
        #########################################################################
        ##########       Step 0 
        
        with open(shots_datadir+'gt_train_step1.pkl', 'rb') as file: 
            gt_train=pickle.load(file)
        
        with open(shots_datadir+'gt_val_step1.pkl', 'rb') as file: 
            gt_val=pickle.load(file)
        
        step0_train_dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "img_stem_suffix": "images",
            "target_stem_suffix": "labels",
            "zip_path": Path(BDD_datadir),
            "target_zip_path": Path(BDD_datadir),
            "class_mapping": get_step2_mapping(),
            #"transforms":self.transforms,
        }
        
        
        #Step 0 Full run ->
        
        step0_train_dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "img_stem_suffix": "images",
            "target_stem_suffix": "labels",
            "zip_path": Path(BDD_datadir),
            "target_zip_path": Path(BDD_datadir),
            "class_mapping": get_step2_mapping(),
            #"transforms":self.transforms,
        }
        
        
        self.step0_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path(BDD_datadir, "train/images"),
            target_folder_path_in_zip=Path(BDD_datadir, "train/labels"),
            ignore_idx=self.ignore_idx,
            is_train_file=gt_train,
	    step_data=0,
            **step0_train_dataset_kwargs,
        )
        
        test_data = os.listdir(BDD_datadir+"test/labels/")
        self.step0_val_dataset = Dataset(
            img_folder_path_in_zip=Path(BDD_datadir, "test/images"),
            target_folder_path_in_zip=Path(BDD_datadir, "test/labels"),
            ignore_idx=self.ignore_idx,
            transforms=self.transforms,
            is_train_file=test_data,
	    step_data=0,
            **step0_train_dataset_kwargs,
        )
        
        
        
        ##########################################################################
        ##############          Step 1
        
        nshot = "10"
        print("Few shot of ",nshot," per class.")
        print("\nTaking files from path ",shots_datadir+'nshot_'+nshot)
        with open(shots_datadir+'nshot_'+nshot+'/IDD_train_step2.pkl', 'rb') as file: 
            step1_train_files=pickle.load(file)
            
        
        step1_train_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "images",
            "target_stem_suffix": "labels",
            "zip_path": Path(IDD_datadir),
            "target_zip_path": Path(IDD_datadir),
            "class_mapping": get_step2_mapping(),
            #"transforms":self.transforms,
        }
        
        
        self.step1_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path(IDD_datadir, "train/images"),
            target_folder_path_in_zip=Path(IDD_datadir, "train/labels"),
            ignore_idx=self.ignore_idx,
            is_train_file=step1_train_files,
            step_data=1,
            **step1_train_dataset_kwargs,
        )
        
        
        test_step1_data = os.listdir(IDD_datadir+"test/labels/")
        step1_val_dataset_load = Dataset(
            img_folder_path_in_zip=Path(IDD_datadir, "test/images"),
            target_folder_path_in_zip=Path(IDD_datadir, "test/labels"),
            ignore_idx=self.ignore_idx,
            transforms=self.transforms,
            is_train_file=test_step1_data,
            step_data=1,
            **step1_train_dataset_kwargs,
        )
        
        
        ## step 0 test data
        test_step0_data = os.listdir(BDD_datadir+"test/labels/")
        step0_val_dataset_load = Dataset(
            img_folder_path_in_zip=Path(BDD_datadir, "test/images"),
            target_folder_path_in_zip=Path(BDD_datadir, "test/labels"),
            ignore_idx=self.ignore_idx,
            transforms=self.transforms,
            is_train_file=test_step0_data,
            step_data=0,
            **step0_train_dataset_kwargs,
        )
        
        
        self.step1_val_dataset = ConcatDataset([step0_val_dataset_load,step1_val_dataset_load])
        
        
        ###########################################################################
        ########        Step 2
        
        with open(shots_datadir+'nshot_'+nshot+'/BDD_train_step3.pkl', 'rb') as file: 
            step3_bdd_train_files = pickle.load(file) 
        
        step2_bdd_train_load = Dataset(
            img_folder_path_in_zip=Path(step3_datadir, "train/images"),
            target_folder_path_in_zip=Path(step3_datadir, "train/labels"),
            ignore_idx=self.ignore_idx,
            transforms=self.transforms,
            is_train_file=step3_bdd_train_files,
            step_data=0,
            **step0_train_dataset_kwargs,
        )
        
        
        with open(shots_datadir+'nshot_'+nshot+'/IDD_repeat_train_step3.pkl', 'rb') as file: 
            step3_idd_rep_train_files=pickle.load(file)
        
        step2_idd_train_load = Dataset(
            img_folder_path_in_zip=Path(step3_datadir, "train/images"),
            target_folder_path_in_zip=Path(step3_datadir, "train/labels"),
            ignore_idx=self.ignore_idx,
            transforms=self.transforms,
            is_train_file=step3_idd_rep_train_files,
            step_data=1,
            **step1_train_dataset_kwargs,
        )
        
        print("Total train files step 2, bdd : ",len(step3_bdd_train_files)," ,idd ",len(step3_idd_rep_train_files))
        
        
        self.step2_train_dataset = ConcatDataset([step2_bdd_train_load,step2_idd_train_load])
        
        
        test_step3_all_files = os.listdir(step3_datadir+'test/labels/')
        print("Total files in step 2 test ",len(test_step3_all_files))
        test_step3_idd_files = []
        test_step3_bdd_files = []
        
        
        for files in test_step3_all_files:
            if 'gtFine' in files:
                test_step3_idd_files.append(files)
            else:
                test_step3_bdd_files.append(files)
        
        
        step2_bdd_val_load = Dataset(
            img_folder_path_in_zip=Path(step3_datadir, "test/images"),
            target_folder_path_in_zip=Path(step3_datadir, "test/labels"),
            ignore_idx=self.ignore_idx,
            transforms=self.transforms,
            is_train_file=test_step3_bdd_files,
            step_data=0,
            **step0_train_dataset_kwargs,
        )
        
        
        step2_idd_val_load = Dataset(
            img_folder_path_in_zip=Path(step3_datadir, "test/images"),
            target_folder_path_in_zip=Path(step3_datadir, "test/labels"),
            ignore_idx=self.ignore_idx,
            transforms=self.transforms,
            is_train_file=test_step3_idd_files,
            step_data=1,
            **step1_train_dataset_kwargs,
        )
        
        print("Total test files step 2 : ",len(test_step3_all_files))
        print("Total test files step 2, bdd : ",len(test_step3_bdd_files)," ,idd ",len(test_step3_idd_files))
        
        self.step2_val_dataset = ConcatDataset([step0_val_dataset_load, \
                step1_val_dataset_load, step2_bdd_val_load, step2_idd_val_load])
        
        
        '''
        Step 0 trial ->
        
        step0_train_dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "img_stem_suffix": "images_in",
            "target_stem_suffix": "labels_in",
            "zip_path": Path(self.root, "images_in"),
            "target_zip_path": Path(self.root, "labels_in"),
            "class_mapping": get_base_mapping(),
            #"transforms":self.transforms,
        }
        
        self.step0_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path(self.root, "images_in/train"),
            target_folder_path_in_zip=Path(self.root, "labels_in/train"),
            ignore_idx=self.ignore_idx,
            **step0_train_dataset_kwargs,
        )
        
        self.step0_val_dataset = Dataset(
            img_folder_path_in_zip=Path(self.root, "images_in/val"),
            target_folder_path_in_zip=Path(self.root, "labels_in/val"),
            ignore_idx=self.ignore_idx,
            transforms=self.transforms,
            **step0_train_dataset_kwargs,
        )
        '''
        
        
        
        
        
        
        
        
        
        
        
        '''
        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.root, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.root, "gtFine_trainvaltest.zip"),
            "class_mapping": get_cityscapes_mapping(),
        }
        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            ignore_idx=self.ignore_idx,
            **cityscapes_dataset_kwargs,
        )
        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            ignore_idx=self.ignore_idx,
            **cityscapes_dataset_kwargs,
        )
        '''
        return self

    def val_dataloader(self):
        now_time = datetime.now()

        s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
        # mm/dd/YY H:M:S format
        print("val_dataloader :", s1)
        print("curr_step = ", self.curr_step)
        sys.stdout.flush()
        
        if self.curr_step==2:
            return DataLoader(
                self.step2_val_dataset,
                collate_fn=self.eval_collate,
                **self.dataloader_kwargs,
            )
        
        if self.curr_step==1:
            return DataLoader(
                self.step1_val_dataset,
                collate_fn=self.eval_collate,
                **self.dataloader_kwargs,
            )
        
        if self.curr_step==0:
            return DataLoader(
                self.step0_val_dataset,
                collate_fn=self.eval_collate,
                **self.dataloader_kwargs,
            )
        
