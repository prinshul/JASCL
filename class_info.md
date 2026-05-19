# Natural-FoSSIL

Session 0 is the base session, followed by incremental sessions. 
The following datasets are used :

- BDD100k: https://www.bdd100k.com/
- IDD: https://idd.insaan.iiit.ac.in/

Session 0 : BDD100k (10 Classes)
```
road - 0
sidewalk - 1
building - 2
wall - 3
fence - 4
traffic light - 5
traffic sign - 6
person - 7
car - 8
truck - 9
```
Session 1: IDD (5 Classes: 5 New)
```
Bridge/tunnel - 10
Parking - 11
rail track - 12 
Autorickshaw - 13
pole - 14
```
Session 2 : IDD (2 classes from BDD100K reappear from the IDD domain) + BDD100k (3 New)
```
Bus- 15 (from BDD100k)
Sky- 16(from BDD100k)
bicycle - 17 (from BDD100k)
person - 7 (repeat - from IDD)
car - 8 (repeat - from IDD) 
```
# Med_FoSSIL-Disjoint

The Med_FoSSIL-Disjoint dataset comprises six domains, each containing entirely distinct classes that represent different organs and tumours.

Session 0 (Total Segmentator - TS)
```
Background - 0
Total Segmentator:
Sacrum - 1
Stomach - 2
Lung upper lobe left - 3
Lung lower lobe left - 4
Brain - 5
Atrium (left) - 6
Ventricle (left) - 7
Pulmonary artery - 8
Aorta - 9
Gallbladder - 10
Trachea - 11
Rib left1 - 12
Rib right1 - 13
Rib left2 - 14
Rib right2 - 15
```
Session 1 (AMOS)
```
Pancreas - 16
Duodenum - 17
Bladder - 18	
Prostate/uterus - 19
Postcava- 20
```
Session 2 (BCV)
```
Spleen - 21
Liver - 22
Left kidney - 23
Right kidney - 24
Left adrenal - 25
Right adrenal - 26
```
Session 3 (MOTS)
```
Hepatic Vessel + Tumor - 27,28
Colon Tumor - 29
Lung Tumor - 30
```
Session 4 (Brats)
```
NCR — label 1	- 31
ED — label 2	- 32
ET — label 4	- 33
```
Session 5 (Verse)
```
C3(3)	- 34
C4(4)	- 35
T1(8)	- 36
T2(9)	- 37
```
# Med_FoSSIL-Mixed

In Med_FoSSIL-Mixed, we have shuffled the order of domains and introduced new organs from the existing domains. Also, we have mixed organs from different domains. For example, in session 2, we introduced new organs from the same domain (Amos - 1 class Bladder). Also, we have mixed up different domains, such as BCV and MOTS, in session 1. 

Session 0
Amos
```
Background - 0
Amos : 
Spleen - 1
right kidney - 2 
left kidney - 3 
Gall bladder - 4
Esophagus - 5
prostate/uterus - 6
Stomach - 7
Arota / aorta - 8
Duodenum - 9
Postcava - 10
```
Session 1
BCV
```
Pancreas - 11
inferior vena cava - 12
portal vein and splenic vein - 13
Left adrenal - 14
Right adrenal - 15
Liver - 16
MOTS Hepatic Vessel + Tumor - 17 and 18
```
Session 2
TS
```
small_bowel - 19
Hip_left - 20
Hip_right - 21
Lung upper lobe left - 22
Lung lower lobe left - 23
TS repeat organs from Amos (Session 0)
Kidney right - 2
Kidney left - 3
Stomach - 7
Amos - 1 new class
Bladder - 24
```
Session 3 
MOTS
```
Colon Tumor - 25
Lung Tumor - 26
TS - 2 new classes
Femur_left - 27
Femur_right - 28
```
Session 4
Brats + Verse (3 + 4)
```
Brats
T1	- 29
T2	- 30
Flair	- 31
Verse
C3(3)	- 32
C4(4)	- 33
T1(8)	- 34
T2(9)	- 35
```
# Med_Semi-Supervised-FoSSIL
This benchmark is a variant of Med_FoSSIL-Disjoint with unalabeled data in incremental sessions (8 to 30 3D volumes) per session.

Session 0 (TS)
```
Background - 0
Total Seg:
Sacrum - 1
Stomach - 2
Lung upper lobe left - 3
Lung lower lobe left - 4
Brain - 5
Atrium (left) - 6
Ventricle (left) - 7
Pulmonary artery - 8
Aorta - 9
Gallbladder - 10
Trachea - 11
Rib left1 - 12
Rib right1 - 13
Rib left2 - 14
Rib right2 - 15
```
Session 1 (AMOS)
```
Pancreas - 16
Duodenum - 17
Bladder - 18	
Prostate/uterus - 19
Postcava- 20
```
Session 2 (BCV)
```
Spleen - 21
Liver - 22
Left kidney - 23
Right kidney - 24
Left adrenal - 25
Right adrenal - 26
```
Session 3 (MOTS)
```
Hepatic Vessel + Tumor - 27,28
Colon Tumor - 29
Lung Tumor - 30
```
Session 4 (Brats)
```
NCR — label 1	- 31
ED — label 2	- 32
ET — label 4	- 33
```
Session 5 (Verse)
```
C3(3)	- 34
C4(4)	- 35
T1(8)	- 36
T2(9)	- 37
```

# Semi-Supervised_Natural-FoSSIL
The following datasets are used :

- BDD100k: https://www.bdd100k.com/
- Cityscapes: https://www.cityscapes-dataset.com/
- IDD: https://idd.insaan.iiit.ac.in/

In this benchmark, each incremental session includes 400 unlabeled samples per class.

Session 0 : BDD100k (10 Classes) 
```
road - 0
sidewalk - 1
building - 2
wall - 3
fence - 4
traffic light - 5
traffic sign - 6
person - 7
car - 8
truck - 9
```
Session 1 : Cityscapes (2 classes) 
```
parking - 10
vegetation - 11
```
Session 2 : IDD (2 classes)
```
rail track - 12
sky - 13
```
Session 3 : IDD (3 classes)
```
motorcycle - 14
autorickshaw - 15
bridge/tunnel - 16
```
# Detection

- [COCO-O](https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_COCO-O_A_Benchmark_for_Object_Detectors_under_Natural_Distribution_Shifts_ICCV_2023_paper.pdf)
  
Session 0 :
```
77 classes from the Painting domain
```

Session 1 :
```
Three disjoint classes (person, car, bench) from the weather domain
```
