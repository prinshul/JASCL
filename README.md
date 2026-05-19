<div align="center">

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-orange?logo=huggingface&style=for-the-badge)](https://huggingface.co/anony34/FoSSIL)

</div>

<br>


This is the official implementation of Continual Segmentation under Joint Nonstationarity accepted at ICML'26.

Continual semantic segmentation remains an underexplored area in both 2D and 3D domains. The problem becomes particularly challenging when classes and domains evolve over time, with incremental classes having only a few labeled samples. In this setting, the model must simultaneously address catastrophic forgetting of old classes, overfitting due to the limited labeled data of new classes, and domain shifts arising from changes in data distribution. Existing methods fail to simultaneously address these real-world constraints. We introduce the JASCL framework, which integrates gradient-adaptive stabilization and prototype anchored supervision (PAS) to enhance continual learning across class-incremental (CIL), domain-incremental (DIL), and few-shot scenarios. Gradient-adaptive stabilization perturbs parameters with overfitted or saturated gradients more strongly, while perturbing parameters with highly changing or large gradients less, preserving critical weights, allowing less critical parameters to explore alternative solutions in the parameter space, mitigating forgetting, reducing overfitting, and improving robustness to domain shifts. For incremental classes with unlabeled data, PAS enables semi-supervised learning by refining pseudo-labels and filtering out incorrect high-confidence predictions, ensuring reliable supervision for incremental classes. Together, these components work synergistically to enhance stability, generalization, and continual learning across all learning regimes.

#
⚙️ Pretrained models are available on [Hugging Face](https://huggingface.co/anony34/FoSSIL).

##
## 🌟 JASCL Datasets 

We have prepared and processed the datasets to be directly usable for all experiments.

+ Download **Med JASCL-Disjoint** data from [here](https://zenodo.org/records/17218309?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQ2OWQ3MjNjLWYxZTItNDNkNi04NjdmLTI3ZDVlZDJkYTUzYSIsImRhdGEiOnt9LCJyYW5kb20iOiJhYmM5ZWIwZjhlYmU3ZjE5NTFmYmEyYTlhNTY0MWJmOCJ9.n1mVFQw092WMjOVF2tm45v3DA2cR4PCZxuKXmX0DzCu5Jrz50fch73vnDqqrMMGpQBSfE1pvzMi7qiDxe4-beA). This setting involves disjoint classes and medical domains (e.g., organs and tumors) across sessions. We evaluated relevant existing few-shot, class-incremental, and domain-incremental methods under this setup using a 3D U-Net backbone. Additionally, we tested medically relevant continual learning approaches to provide a comprehensive comparison.
+ Download **Med JASCL-Mixed** data from [here](https://zenodo.org/records/17297404). This is a variant of **Med JASCL-Disjoint**, where classes and medical domains may reappear across sessions, and multiple domains can be present within the same session. In this setting, we evaluated the robustness of various backbones, including pre-trained models such as CLIP-driven architectures and transformer-based backbones like MedFormer and SwinUNetr.
+ Download **Med Semi-Supervised-JASCL** data from [here](https://zenodo.org/records/17218309?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQ2OWQ3MjNjLWYxZTItNDNkNi04NjdmLTI3ZDVlZDJkYTUzYSIsImRhdGEiOnt9LCJyYW5kb20iOiJhYmM5ZWIwZjhlYmU3ZjE5NTFmYmEyYTlhNTY0MWJmOCJ9.n1mVFQw092WMjOVF2tm45v3DA2cR4PCZxuKXmX0DzCu5Jrz50fch73vnDqqrMMGpQBSfE1pvzMi7qiDxe4-beA). This is a variant of Med JASCL-Disjoint, where each incremental class is accompanied by unlabeled data. In this setting, we evaluated relevant semi-supervised approaches that can assist in multi-constraint continual semantic segmentation.
+ Download **Natural-JASCL** data from [here](https://zenodo.org/records/17255889?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjFjOTk5M2RhLTZiOTctNDFhMi04ZjA3LThmYzZmOWM0ZjllMCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MTFkYjM2NzgyOGQ2NzQyOTk3YzQ2MmM4NTg0OWQwMSJ9.aUa1gFihHpXF4ixrywaSZI59q1pEYy3z4yXxnVwNQm22Pf9-ZMACj7tu-Q_O8yAgvq35U5t0yCepZWf6iRjCDw). This setting comprises multiple autonomous-driving domains, where classes may reappear across sessions. Within this setup, we evaluated class-incremental-only, domain- & class-incremental, and few-shot class-incremental learning (FSCIL) methods for semantic segmentation, adapted from the natural 2D domain. We also assessed the robustness of various backbones, including SAM, and achieved improvements over its original performance.
+ Download **Semi-Supervised Natural-JASCL** data from [here](https://zenodo.org/records/17255889?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjFjOTk5M2RhLTZiOTctNDFhMi04ZjA3LThmYzZmOWM0ZjllMCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MTFkYjM2NzgyOGQ2NzQyOTk3YzQ2MmM4NTg0OWQwMSJ9.aUa1gFihHpXF4ixrywaSZI59q1pEYy3z4yXxnVwNQm22Pf9-ZMACj7tu-Q_O8yAgvq35U5t0yCepZWf6iRjCDw). This setting includes domains from natural driving scenes, where incremental classes are introduced with access to unlabeled data. In this scenario, we evaluated semi-supervised methods that are relevant for multi-constraint continual semantic segmentation.
+ Download the **Detection** data from [here](https://drive.google.com/file/d/1aBfIJN0zo_i80Hv4p7Ch7M8pRzO37qbq/view?usp=drive_link). Rename the folder to ``Detection_data``.
Detection data structure:
```
Detection_data/ood_coco
|-- sketch
    |-- val2017
        |-- xxx.jpg
        |-- ...
    |-- annotations
        |-- instances_val2017.json
|-- painting
    |-- val2017
        |-- xxx.jpg
        |-- ...
    |-- annotations
        |-- instances_val2017.json
|-- weather
    |-- val2017
        |-- xxx.jpg
        |-- ...
    |-- annotations
        |-- instances_val2017.json
|-- cartoon
    ...
```

Note: The **Med JASCL-Disjoint** data provides the full data required for **Med Semi-Supervised-JASCL** experiments. The **Med Semi-Supervised-JASCL** data, on the other hand, contains only the unlabeled portion of the **Med JASCL-Disjoint** dataset. We can run both **Med JASCL-Disjoint** and **Med Semi-Supervised-JASCL** experiments with **Med JASCL-Disjoint** data.

We have used the following datasets (domains) to create medical benchmark datasets: [TS](https://github.com/wasserth/TotalSegmentator), [AMOS](https://proceedings.neurips.cc/paper_files/paper/2022/file/ee604e1bedbd069d9fc9328b7b9584be-Paper-Datasets_and_Benchmarks.pdf), [BCV](https://www.synapse.org/Synapse:syn3193805/wiki/217789), [BraTS](https://www.med.upenn.edu/cbica/brats2020/data.html), [MOTS](https://github.com/jianpengz/DoDNet), [VerSe](https://github.com/anjany/verse). \
We have used the following datasets (domains) to create natural benchmark datasets: [BDD](https://bair.berkeley.edu/blog/2018/05/30/bdd/), [IDD](https://idd.insaan.iiit.ac.in/), [Cityscapes](https://www.cityscapes-dataset.com/). \
For detection experiments, we have used the multi-domain [COCO-O](https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_COCO-O_A_Benchmark_for_Object_Detectors_under_Natural_Distribution_Shifts_ICCV_2023_paper.pdf) dataset.

The following table shows the number of incremental classes and their domains in each session for all [settings](https://github.com/anony34/FoSSIL/blob/main/class_info.md) (SS stands for Semi-Supervised):

| Setting              | Session 0 (Base)       | Session 1             | Session 2           | Session 3           | Session 4           | Session 5           |
|------------------------|-----------------------|----------------------|--------------------|--------------------|--------------------|--------------------|
| Med JASCL-Disjoint | 15 (TS)               | 5 (AMOS)             | 6 (BCV)            | 4 (MOTS)           | 3 (BraTS)          | 4 (VerSe)          |
| Med JASCL-Mixed   | 10 (AMOS)             | 8 (BCV, MOTS)        | 6 (TS, AMOS)       | 4 (MOTS, TS)       | 7 (BraTS, VerSe)   | --                 |
| Med SS-JASCL       | 15 (TS)               | 5 (AMOS)             | 6 (BCV)            | 4 (MOTS)           | 3 (BraTS)          | 4 (VerSe)          |
| Natural-JASCL      | 10 (BDD)              | 5 (IDD)              | 5 (BDD, IDD)       | --                 | --                 | --                 |
| SS Natural-JASCL   | 10 (BDD)              | 2 (Cityscapes)       | 2 (IDD)            | 3 (IDD)            | --                 | --                 |

Base has a large number of labeled samples, while incremental sessions (1-5) have fixed *K*-shot labeled samples with additional unlabeled data. In the segmentation results, for any session, we report the Dice coefficient and IoU, averaged over all classes in the current session as well as those from preceding sessions that test both forgetting on old-classes and overfitting on novel classes from different domains. For Detection, we report the mAP (mean Average Precision) for each session individually. **Example**: In the base training session, 10 classes are trained with 2000 labeled examples with 100 validation and 200 test examples. In the next 5-shot incremental session training with 5 new classes, 50 validation examples, and 100 test examples, each class has 5 *fixed* labeled examples throughout training. The model is tested on (current 5 classes + previous 10 classes) in this session. In a semi-supervised setting, everything remains the same as described previously, except that additional unlabeled examples (e.g., 500) may be provided per class or per session.

##
## 🎯 Environment Setup
```
git clone https://github.com/prinshul/JASCL.git
cd JASCL

** Medical ** 
conda create -n med_fossil python=3.9.19
conda activate med_fossil
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements_med.txt
pip install numpy==1.23.5
pip install SimpleITK==2.0.2
(Common for all medical experiments) 

** Natural **
conda create -n natural_foSSIL python=3.9.19
conda activate natural_foSSIL
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_nat.txt
(Common for all natural experiments) 

** Detection **
conda create -n det_fossil python=3.9.23
conda activate det_fossil
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_det.txt
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install yapf==0.32
pip install "numpy<2"
```
#
## 🔥 Baselines and Backbones
The most relevant and abundant baselines related to our proposed framework are few-shot class-incremental learning (FSCIL) methods, which jointly integrate the class-incremental and few-shot learning paradigms. For natural scene datasets, we re-implemented and adapted several popular class-incremental-only, domain- & class-incremental, and FSCIL methods for semantic segmentation, using their publicly available implementations. In contrast, medical-domain baselines for semantic segmentation are considerably scarcer. Therefore, we re-implemented and adapted general approaches that are most closely aligned with our framework, leveraging their publicly available source codes. We further implemented a variety of semi-supervised, representation learning, few-shot learning, meta-learning, and active learning approaches, as well as methods designed to address domain shifts. Detailed information on the implemented baselines is provided [here](https://github.com/anony34/FoSSIL/blob/main/baselines.md). We include several popular and recent backbones, such as 3D-UNet, DeepLabv3+, Faster R-CNN, and transformer-based architectures, including MedFormer, SwinUNetr, and SAM.


#
In our experiments, we tried to pick the best models based on different criteria, like picking the last trained model or picking the model with the best performance on validation data. Using validation data is optional, but it can be used to tune hyperparameters or to select the best model. We applied the same criteria consistently to the baselines. We have used *fixed* *K*-shot labeled samples per class throughout the entire training process in incremental sessions. In our medical experiments, we fixed the seed (1024) for all methods to select 5 (*K*=5) labeled examples per class from the benchmark datasets, which were kept fixed throughout training in incremental sessions. For semi-supervised segmentation experiments, we implemented a simple [mean-teacher](https://arxiv.org/abs/1703.01780) model. For detection, we used [SoftTeacher](https://github.com/microsoft/SoftTeacher). It is recommended to ``export CUDA_VISIBLE_DEVICES=0`` before running experiments. We conducted all experiments on a single NVIDIA A100 GPU (40GB). Within the JASCL framework, we employed both [stochastic](https://github.com/anony34/JASCL/blob/f76ba0f2d1ac0a14f8ac4f6e0a64752dfc157e38/Med_FoSSIL-Mixed/base/Medformer_base/networks/med_former/medformer.py) and [deterministic](https://github.com/anony34/FoSSIL/blob/main/Natural-FoSSIL/base/benchmark-vfm-ss/models/linear_decoder.py) classifier heads for the models used in the base session. Improvements are consistently observed for both formulations over the incremental sessions. We do not compare JASCL with other methods in the base session, as most existing methods introduce their novel components during the incremental sessions.
#
## 🏁 Semi-Supervised Natural-JASCL

###
Download the data corresponding to **Semi-Supervised Natural-JASCL** and put the paths to the data in ``share_quant.py`` in the *base* and *inc* (incremental) folders.
Run the following base command in the *base* folder: 
```
cd FoSSIL/Semi-Supervised_Natural-FoSSIL/base/vanilla_deeplab_final
python train_step1.py --eval-type train
python train_step1.py --eval-type test
```
Run the incremental sessions in the *inc* folder using:
```
cd FoSSIL/Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT
python train_step2.py --eval-type train --nshot 10 --pseudo_label 5 --proto_use 25 --batch-size 2
python train_step2.py --eval-type test --nshot 10
python train_step3.py --eval-type train --nshot 10 --pseudo_label 5 --proto_use 25 --batch-size 2
python train_step3.py --eval-type test --nshot 10
python train_step4.py --eval-type train --nshot 10 --pseudo_label 5 --proto_use 25 --batch-size 2
python train_step4.py --eval-type test --nshot 10
```
We have introduced unlabeled data at the 25th epoch and repeat it after every 5 epochs. 10-shot learning uses 10 fixed labeled samples per class throughout the entire training process in incremental sessions.

##
## 🏁 Natural-JASCL (with SAM)
###
Download the data corresponding to **Natural-JASCL** and place the paths to the data in ``datasets/urbanscenes.py`` (around line 73) in *base* and *inc* folders.
Run the following base session in the *base* folder: 

```
cd FoSSIL/Natural-FoSSIL/base/benchmark-vfm-ss
python main.py fit -c configs/step0.yaml --root results/step0 --model.network.encoder_name samvit_base_patch16.sa1b 
```
Run the incremental sessions in the *inc* folder using:
```
cd FoSSIL/Natural-FoSSIL/inc/benchmark-vfm-ss_new

python main.py fit -c configs/step1.yaml \
  --root results/step1 \
  --model.network.encoder_name samvit_base_patch16.sa1b \
  --model.network.ckpt_path <CKPT_PATH_from_base_folder> \
  --model.freeze_encoder True

python main.py fit -c configs/step2.yaml \
  --root results/step2 \
  --model.network.encoder_name samvit_base_patch16.sa1b \
  --model.network.ckpt_path <CKPT_PATH_from_step1> \
  --model.freeze_encoder True
```
``CKPT_PATH_from_base_folder`` is the base checkpoint inside ``base/benchmark-vfm-ss/results/step0/lightning_logs/version_{i}/checkpoints/``  where ``i`` is the current version. ``CKPT_PATH_from_step1`` is the incremental checkpoint inside ``inc/benchmark-vfm-ss/results/step1/lightning_logs/version_{i}/checkpoints/``  where ``i`` is the current version.

##
## 🏁 Med JASCL-Disjoint
###
Download the data corresponding to **Med JASCL-Disjoint** and place the paths to the data in ``argparser.py`` : (``train_root_path``, ``val_root_path``, ``list_dir``) in *base* and *inc* folders.
Run the following base session in the *base* folder: 
```
cd FoSSIL/Med_FoSSIL-Disjoint/base/codu_run/codu
sh run/merged-ms.sh
```
(add ``--test`` to the command in ``run/merged-ms.sh`` to test)    
for example, in ``run/merged-ms.sh``

```
train command: exp --method FT --name FT --lr ${lr} ${gen_par} --num_classes 16 --step 0 --debug --batch_size ${bs} 
test command: exp --method FT --name FT --lr ${lr} ${gen_par} --num_classes 16 --step 0 --debug --batch_size ${bs} --test
```
Run the incremental sessions in the *inc* folder using:
```
cd FoSSIL/Med_FoSSIL-Disjoint/inc/codu_perturbation/codu
sh run/merged-ms.sh
```
(Use the corresponding session command to train the incremental sessions accordingly.) 

Note: **Med JASCL-Mixed** and **Med Semi-Supervised-JASCL** can be setup similarly and run. For the **Med JASCL-Mixed** and **Med Semi-Supervised-JASCL** experiments, the checkpoints and prototypes from the *base* should be copied into their respective folders within the *inc* folder.

##
## 🏁 Detection
###
Download the data corresponding to Detection (change folder name to ``Detection_data``) and place the path to the data in ``tools/prepare_splits.py`` : (``data_path`` variable).

Run the following command to prepare the required data splits: 
```
cd FoSSIL/Detection/
python tools/prepare_split.py 
```

Run session 1 (train) after updating the [base](https://drive.google.com/file/d/1aMP9uh_Te21BhijG7bqHgD4hHFHfFA-k/view?usp=drive_link) checkpoint, updating your data and code root paths  in ``configs/soft_incremental/step1.py`` at lines 18,6 and 7 respectively:
```
python tools/train.py configs/soft_incremental/step1.py
```
To test the session 1 model:
```
python tools/eval_incremetal.py --step1_ckpt work_dirs/ssl_coco_o_step1/best_coco_bbox_mAP_epoch_{i}.pth
```
where ``i`` is the epoch corresponding to the best checkpoint.

#
## 🏆 Results

✅ Performance of baselines on Med JASCL-Disjoint benchmark (3-sessions). Results reported as Dice coefficients (0-1). PD (Performance drop rate) = ((Session 0 − Session 2) / Session 0) × 100.
| **Method**         | **Session 0** | **Session 1** | **Session 2** | **PD (↓)** |
| :----------------- | :-----------: | :-----------: | :-----------: | :----------: |
| PIFS               |     0.700     |     0.129     |     0.078     |     88.9     |
| NC-FSCIL           |     0.394     |     0.077     |     0.081     |     79.4     |
| CLIP-CT            |     0.475     |     0.186     |     0.141     |     70.3     |
| MiB                |     0.700     |     0.271     |     0.096     |     86.3     |
| MDIL               |     0.779     |     0.115     |     0.097     |     87.6     |
| C-FSCIL            |     0.787     |     0.334     |     0.297     |     62.3     |
| SoftNet            |     0.820     |     0.305     |     0.146     |     82.2     |
| GAPS               |     0.700     |     0.334     |     0.253     |     63.9     |
| FSCIL-SS           |     0.700     |     0.115     |     0.089     |     87.3     |
| Subspace           |     0.257     |     0.054     |     0.040     |     84.4     |
| Gen-Replay         |     0.700     |     0.076     |     0.102     |     85.4     |
| FeCAM              |     0.700     |     0.048     |     0.042     |     94.0     |
| FACT               |     0.357     |     0.071     |     0.028     |     92.2     |
| MAML               |     0.700     |     0.001     |     0.059     |     91.6     |
| MAML + Reg.        |     0.700     |     0.001     |     0.062     |     91.1     |
| MTL                |     0.700     |     0.079     |     0.088     |     87.4     |
| UnSupCL            |     0.700     |     0.039     |     0.088     |     87.4     |
| SupCL              |     0.700     |     0.058     |     0.042     |     94.0     |
| UnSupCL-HNM        |     0.700     |     0.035     |     0.068     |     90.3     |
| **JASCL (U-Net)** |   0.736   |   **0.460**   |   **0.398**   |   **45.9**   |

#
✅ Performance on Natural-JASCL benchmark. All values are reported as mIoU (0–100).  
PD (Performance Drop rate) = ((Session 0 − Session 2) / Session 0) × 100.

| Method           | Session 0 | Session 1 | Session 2 | PD (↓) |
|----------------- |:---------:|:---------:|:---------:|:------:|
| DeepLab Vanilla  | 47.76     | 2.18      | 3.86      | 91.9   |
| GAPS             | 47.76     | 23.42     | 16.68     | 65.1   |
| MiB              | 47.76     | 2.50      | 2.37      | 95.0   |
| MDIL             | 48.54     | 1.59      | 3.02      | 93.8   |
|----------------- |-----------|-----------|-----------|--------|
| SAM Vanilla      | 66.0      | 32.6      | 30.81     | 53.3   |
| **JASCL (SAM)** | 66.0      | **33.2**  | **31.22** | **52.7** |



#
✅ Performance of recent prototype replay-based and semi-supervised methods on **Med JASCL-Disjoint** and **Med Semi-Supervised-JASCL** benchmark datasets with U-Net and MedFormer backbones, respectively. Results reported as Dice coefficients (0-1).
| **Benchmark dataset / Method**                         | **Session 0** | **Session 1** |
|------------------------------------------------|---------------|---------------|
| **Med JASCL-Disjoint (U-Net)**                |               |               |
| [Saving100x](https://openreview.net/pdf?id=Ct0zPIe3xs) | 0.700         | 0.072         |
| [Adaptive Prototype](https://ojs.aaai.org/index.php/AAAI/article/view/33188) | 0.700         | 0.044         |
| **JASCL**                                     | 0.736     |  **0.460**|
| **Med Semi-Supervised-JASCL (MedFormer)**     |               |               |
| [CSL](https://iccv.thecvf.com/virtual/2025/poster/1319) | 0.659         | 0.040         |
| **JASCL**                                     | 0.640     |**0.431** |

#
✅ Performance of JASCL on the Med JASCL-Disjoint benchmark and its variant, Med Semi-Supervised-JASCL (which includes additional unlabeled data), evaluated across incremental sessions (TS (Base) → AMOS → BCV → MOTS → BraTS → VerSe).

**Seen** refers to the average performance on classes the model has encountered in previous sessions, while **New** refers to the average performance on classes introduced in the current session. Results are reported as Dice coefficients (0–1). The reported results confirm that unlabeled data helps to boost the performance.  
SS refers to Semi-Supervised, and HM refers to Harmonic Mean. SS JASCL denotes the performance of JASCL on the Med Semi-Supervised-JASCL benchmark.

| **Method** | **AMOS Seen** | **AMOS New** | **AMOS HM** | **BCV Seen** | **BCV New** | **BCV HM** | **MOTS Seen** | **MOTS New** | **MOTS HM** | **BraTS Seen** | **BraTS New** | **BraTS HM** | **VerSe Seen** | **VerSe New** | **VerSe HM** |
|:-----------:|:-------------:|:------------:|:-----------:|:------------:|:------------:|:-----------:|:-------------:|:------------:|:-----------:|:--------------:|:------------:|:-----------:|:--------------:|:------------:|:-----------:|
| **JASCL** | 0.610 | 0.074 | 0.132 | 0.477 | **0.069** | **0.120** | 0.382 | 0.180 | 0.245 | **0.043** | 0.198 | **0.071** | 0.367 | 0.119 | 0.180 |
| **SS JASCL** | **0.706** | **0.099** | **0.174** | **0.561** | 0.065 | 0.116 | **0.444** | **0.218** | **0.292** | 0.042 | **0.216** | 0.070 | **0.391** | **0.182** | **0.248** |



#
✅ Performance of JASCL on the Semi-Supervised Natural-JASCL benchmark (which includes additional unlabeled data), evaluated across incremental sessions (BDD100K (Base) → Cityscapes → IDD → IDD (with different classes from the previous session)).

**Seen** refers to the average performance on classes the model has encountered in previous sessions, while **New** refers to the average performance on classes introduced in the current session. Results are reported as mIoU (0–100). The results show that pseudo-label refinement (PRL) enhances the performance of JASCL compared to JASCL without pseudo-label refinement (w/o PRL). **HM** refers to Harmonic Mean.  

| **Method** | **Cityscapes Seen** | **Cityscapes New** | **Cityscapes HM** | **IDD Seen** | **IDD New** | **IDD HM** | **IDD Seen** | **IDD New** | **IDD HM** |
|:-----------:|:-------------------:|:------------------:|:----------------:|:-------------:|:------------:|:-----------:|:-------------:|:------------:|:-----------:|
| **JASCL + GAPS** | **27.33** | **29.98** | **28.59** | **26.00** | **49.82** | **34.17** | **28.88** | **24.23** | **26.35** |
| **JASCL + GAPS w/o PAS** | 25.62 | 27.88 | 26.70 | 23.46 | 41.50 | 29.97 | 27.54 | 15.74 | 20.03 |


#
✅ Performance of JASCL (MedFormer) on the Med JASCL-Mixed benchmark evaluated across incremental sessions (**Session 0 (Base) → Session 1 → Session 2 → Session 3 → Session 4**).

**Seen** refers to the average performance on classes the model has encountered in previous sessions, while **New** refers to the average performance on classes introduced in the current session.  Results are reported as **Dice coefficients (0–1)**.  
The results show that **gradient-adaptive stabilization (GAS)** enhances the performance of JASCL compared to JASCL without gradient-adaptive stabilization (**w/o GAS**). **HM** refers to Harmonic Mean.  

| **Method** | **Session&nbsp;1<br>Seen** |**Session&nbsp;1<br>New** | **Session&nbsp;1<br>HM** | **Session&nbsp;2<br>Seen** |  **Session&nbsp;2<br>New** |  **Session&nbsp;2<br>HM** | **Session&nbsp;3<br>Seen** | **Session&nbsp;3<br>New**  | **Session&nbsp;3<br>HM** | **Session&nbsp;4<br>Seen** | **Session&nbsp;4<br>New** | **Session&nbsp;4<br>HM** |
|:-----------:|:----------------:|:----:|:----:|:----------------:|:----:|:----:|:----------------:|:----:|:----:|:----------------:|:----:|:----:|
| **JASCL** | **0.534**&nbsp;&nbsp; | **0.159**&nbsp;&nbsp; | **0.245**&nbsp;&nbsp; | **0.398**&nbsp;&nbsp; | **0.076**&nbsp;&nbsp; | **0.128**&nbsp;&nbsp; | **0.329**&nbsp;&nbsp; | **0.046**&nbsp;&nbsp; | **0.081**&nbsp;&nbsp; | **0.289**&nbsp;&nbsp; | **0.038**&nbsp;&nbsp; | **0.067**&nbsp;&nbsp; |
| JASCL w/o GAS | 0.105&nbsp;&nbsp; | 0.010&nbsp;&nbsp; | 0.018&nbsp;&nbsp; | 0.069&nbsp;&nbsp; | 0.015&nbsp;&nbsp; | 0.025&nbsp;&nbsp; | 0.061&nbsp;&nbsp; | 0.034&nbsp;&nbsp; | 0.043&nbsp;&nbsp; | 0.099&nbsp;&nbsp; | 0.008&nbsp;&nbsp; | 0.014&nbsp;&nbsp; |

#
## 📊 Cost Analysis
| **Setting**                     | **Parameters**       |**Parameters**            | **FLOPs**          |  **FLOPs**           | **Training Time**   |  **Training Time**          |
|---------------------------------|--------------------|-----------|------------------|-----------|------------------|-----------|
|                                 | JASCL             | w/o JASCL | JASCL           | w/o JASCL | JASCL           | w/o JASCL |
| **Med_JASCL-Disjoint**         | 16.27M             | 16.27M     | 0.52T            | 0.52T      | 4hrs 6mins       | 4hrs 5mins |
| **Med_Semi-Supervised-JASCL** | 39.59M             | 39.59M     | 1.1T             | 1.1T       | 5hrs 18mins      | 5hrs 8mins |
| **Natural-JASCL (SAM vit-b)**  | 88.9M              | 88.9M      | 0.37T            | 0.37T      | 1hrs 43mins      | 1hrs 35mins |

We report the computational analysis for Session 1. The results show that JASCL does not incur any additional computational cost while significantly improving performance. To store the prototypes, we consume 𝑂(𝑁𝐷) memory, where 𝑁 is the number of classes and 𝐷 is the feature dimension, which is significantly smaller than the memory required to store images.

#
## 🧩 Pseudo Code (JASCL as plug-and-play)
### 🛠️ Gradient-adaptive stabilization
```bash
class Probabilistic_Classifier:
    # The classifier is initialized with weights (W) and a parameter
    # to track gradient information ('grad_update').
    function initialize(input_channels, num_classes, kernel_size):
        # Let 'W' be the weights of a standard convolutional layer.
        self.W = Conv2D(in_channels=input_channels, out_channels=num_classes, kernel_size=kernel_size)

        # 'grad_update' is a trainable parameter that will store gradients
        # information to estimate uncertainty for each weight. Initialize as zeros.
        self.grad_update = Parameter(shape=self.W.weight.shape, initial_value=zeros)

        # 'temp' is a temperature for scaling logits.
        self.temp = 10.0

    # --- Forward Pass ---
    # Defines how the input features are processed.
    function forward(input_features, stochastic_mode=True):
        epsilon = 1e-8
        inverse_grad = 1 / (self.grad_update + epsilon)
        # This factor determines the magnitude of noise to be added.


        # Normalize for stability.
        min_val = min(inverse_grad)
        max_val = max(inverse_grad)
        inverse_grad_normalized = (1 + inverse_grad - min_val) / (1 + (max_val - min_val))

        if stochastic_mode is True:
            # Generate random noise from a standard normal distribution.
            noise = sample_from_standard_normal(shape=self.W.weight.shape)

            # Calculate new weights by adding scaled noise to the weights W.
            sampled_weights = self.W.weight + inverse_grad_normalized * noise
        else:
            # In deterministic (evaluation) mode, use the weights W.
            sampled_weights = self.W.weight

        normalized_weights = L2_normalize(sampled_weights, dimension=1)
        normalized_input = L2_normalize(input_features, dimension=1)

        scores = convolution(normalized_input, normalized_weights)
        final_scores = scores * self.temp

        return final_scores
```
###
During incremental learning, in your model's definition, you replace your final classification layer (``conv`` or ``linear``) with the ``Probabilistic_Classifier``. After calculating the gradients during backpropagation (after ``loss.backward()``), copy the gradients of the weights (``W.weight``) into the ``grad_update`` parameter. 

##
### 🛠️ Prototype Anchored Supervision (PAS)
```bash
function filter_pseudo_labels(logits, features, class_prototypes, confidence_threshold, similarity_threshold):

    # Apply softmax to get probabilities.
    probabilities = softmax(logits)
    # Get the predicted class (pseudo-label) and its confidence (max probability).
    confidence_scores, initial_labels = get_max_and_argmax(probabilities)

    # Reshape features, labels, and confidences into 1D arrays.
    flat_features = flatten(features)           # Shape: [N_pixels, feature_dim]
    flat_labels = flatten(initial_labels)       # Shape: [N_pixels]
    flat_confidence = flatten(confidence_scores)  # Shape: [N_pixels]

    # Normalize features and prototypes to compute cosine similarity.
    normalized_features = L2_normalize(flat_features)
    normalized_prototypes = L2_normalize(class_prototypes)

    # For each pixel, calculate the similarity between its feature vector and the
    # prototype corresponding to its predicted class.
    # similarity = cos(feature_pixel_i, prototype_class_k) where class_k is the predicted label for pixel_i.
    similarity_scores = cosine_similarity(normalized_features, normalized_prototypes[flat_labels])

    # A pseudo-label is considered reliable only if BOTH conditions are met.
    is_confident = flat_confidence > confidence_threshold
    is_similar = similarity_scores > similarity_threshold
    keep_mask = is_confident AND is_similar

    # Create a new label map, initially identical to the predicted labels.
    filtered_labels = flat_labels.clone()
    # Mark unreliable pixels with an ignore index (e.g., -1 or 255).
    filtered_labels[NOT keep_mask] = IGNORE_INDEX

    return filtered_labels
```
```bash
# --- Inside your training loop for unlabeled data ---

# Get predictions from both student and teacher models
student_logits, student_features = student_model(unlabeled_images)
teacher_logits, teacher_features = teacher_model(unlabeled_images)

# Refine the pseudo-labels from both models
student_filtered_labels = filter_pseudo_labels(student_logits, student_features, prototypes)
teacher_filtered_labels = filter_pseudo_labels(teacher_logits, teacher_features, prototypes)

# Use the refined labels in a consistency loss
# The loss is now calculated only on the reliable, filtered pixels.
consistency_loss = MSE_loss(student_filtered_labels, teacher_filtered_labels)

# Backpropagate the loss to train the student model
consistency_loss.backward()
optimizer.step()
```

###
Generate Prototypes from the labeled data. Perform a forward pass to get the logits and deep features from the model. Call the ``filter_pseudo_labels`` function with the model outputs and the pre-computed prototypes. The resulting ``filtered_labels`` can now be used as a high-quality target for the consistency loss.

###
## 🤝 Acknowledgement

For baselines implementation ([med_disjoint](https://github.com/anony34/FoSSIL/tree/main/baselines_Med_disjoint), [med_mixed](https://github.com/anony34/FoSSIL/tree/main/baselines_Med_mixed), [natural](https://github.com/anony34/FoSSIL/tree/main/baselines_Natural), [med_Semi-Supervised](https://github.com/anony34/FoSSIL/tree/main/baselines_semi-supervised_medical), [natural_Semi-Supervised](https://github.com/anony34/FoSSIL/tree/main/baselines_semi-supervised_natural), [ablation_constraints_cl](https://github.com/anony34/FoSSIL/tree/main/ablations_constraints_CL)) we have used the following repositories: 

**Backbones**: [U-Net](https://github.com/yhygao/CBIM-Medical-Image-Segmentation/blob/main/model/dim3/unet.py) (Medical 3D volumes), [Faster R-CNN](https://github.com/microsoft/SoftTeacher) (Detection), [DeepLabV3+](https://github.com/f1recracker/pytorch-deeplab-v3-plus) (Natural scenes), [MedFormer](https://github.com/yhygao/CBIM-Medical-Image-Segmentation) (Medical 3D volumes), [SwinUNetr](https://github.com/yhygao/CBIM-Medical-Image-Segmentation/blob/main/model/dim3/swin_unetr.py) (Medical 3D volumes), [SAM](https://github.com/tue-mps/benchmark-vfm-ss) (Natural scenes).

**Class-Incremental Semantic Segmentation**: [MiB](https://github.com/fcdl94/MiB), [CLIP-CT](https://github.com/MrGiovanni/ContinualLearning), [Saving100x](https://github.com/jinpeng0528/STAR), [Adaptive Prototype](https://github.com/zhu-gl-ux/Adapter).

**Domain-Incremental Learning**: [MDIL](https://github.com/prachigarg23/MDIL-SS).

**Few-shot Class-Incremental Learning**: [PIFS](https://github.com/fcdl94/FSS), [Subspace](https://github.com/feyzaakyurek/subspace-reg), [C-FSCIL](https://github.com/IBM/constrained-FSCIL), [FACT](https://github.com/LAMDA-CL/CVPR22-Fact), [NC-FSCIL](https://github.com/NeuralCollapseApplications/FSCIL), [Gen-Replay](https://github.com/mobaidoctor/med-ddpm), [GAPS](https://github.com/RogerQi/GAPS), [SoftNet](https://github.com/ihaeyong/SoftNet-FSCIL), [FSCIL-SS](https://github.com/ChasonJiang/FSCILSS), [FeCAM](https://github.com/dipamgoswami/FeCAM).

**Semi-Supervised Learning**: [RETRIEVE](https://github.com/decile-team/cords), [NNCSL](https://github.com/kangzhiq/NNCSL), [UaD-CE](https://github.com/yawencui/UaD-ClE), [CSL](https://github.com/PanLiuCSU/CSL).

**Others (Representation Learning, Meta-Learning, Active Learning, Few-shot Learning and Domain shift)**: [SupCL](https://github.com/google-research/google-research/tree/master/supcon), [UnSupCL](https://github.com/google-research/simclr), [UnSupCL-HNM](https://github.com/joshr17/HCL), [MTL](https://github.com/AI-secure/multi-task-learning), [MAML](https://github.com/CEA-LIST/MetaMTReg), [CLIP-driven](https://github.com/ljwztc/CLIP-Driven-Universal-Model), [HALO](https://github.com/paolomandica/HALO).

###
## 📖 License

[![MIT License](https://img.shields.io/badge/MIT-License-yellow?logo=open-source)](https://opensource.org/licenses/MIT) (code)  
[![CC BY-NC 4.0](https://img.shields.io/badge/CC_BY--NC-4.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/) (models & datasets) 

© 2025 Anony34. Use code under MIT License, models and datasets under CC BY-NC 4.0.






































































































































