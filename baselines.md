
# **Class-Incremental Semantic Segmentation**: 

[MiB](https://github.com/fcdl94/MiB): A class-incremental learning method for semantic segmentation in natural images.

[CLIP-CT](https://github.com/MrGiovanni/ContinualLearning): A Continual learning method for abdominal *multi-organ* and *tumour segmentation*.

[Saving100x](https://github.com/jinpeng0528/STAR): A class-incremental learning method for semantic segmentation in natural images that leverages *prototype replay*.

[Adaptive Prototype](https://github.com/zhu-gl-ux/Adapter): A class-incremental learning method for semantic segmentation in natural images that adaptively updates *prototypes*.

These methods cannot handle few-shot learning and domain shifts.

# **Domain-Incremental Learning**: 

[MDIL](https://github.com/prachigarg23/MDIL-SS): A class- & domain-incremental learning method for semantic segmentation in natural images that employs multiple decoders for different domains with a shared encoder.

MDIL uses a separate decoder per domain, which limits scalability as the number of domains increases. It cannot handle few-shot incremental classes.

# **Few-shot Class-Incremental Learning**:

[PIFS](https://github.com/fcdl94/FSS): A *prototype-based* method for few-shot class-incremental semantic segmentation.

[Subspace](https://github.com/feyzaakyurek/subspace-reg): A *regularization-based* method for few-shot class-incremental learning.

[C-FSCIL](https://github.com/IBM/constrained-FSCIL): A meta-learning method for few-shot class-incremental learning.

[FACT](https://github.com/LAMDA-CL/CVPR22-Fact): A few-shot class-incremental learning method that reserves the embedding space for new classes in the base session for future possible extensions. 

[NC-FSCIL](https://github.com/NeuralCollapseApplications/FSCIL): A few-shot class-incremental learning method that proposes to fix a learnable classifier as a geometric structure instructed by neural collapse. [Zhong et al.](https://arxiv.org/pdf/2301.01100) showed that a semantic segmentation model with a classifier fixed as a simplex equiangular tight frame (ETF) performs significantly worse than a model with a learnable classifier.

[Gen-Replay](https://github.com/mobaidoctor/med-ddpm): A *data-free replay* method that implements [Liu et al.](https://arxiv.org/abs/2207.11213) with a diffusion model to generate 3D medical volumes.

[GAPS](https://github.com/RogerQi/GAPS):  A few-shot class incremental learning method that utilizes guided copy-paste augmentation to synthesize diverse training data in semantic segmentation.

[SoftNet](https://github.com/ihaeyong/SoftNet-FSCIL): A few-shot class incremental learning method that jointly learns the model weights and adaptive soft masks to minimize *catastrophic forgetting* and to avoid *overfitting* novel few samples.

[FSCIL-SS](https://github.com/ChasonJiang/FSCILSS): A few-shot class incremental learning method that uses pseudo-labeling to augment novel classes and leverages knowledge distillation to prevent *forgetting*.

[FeCAM](https://github.com/dipamgoswami/FeCAM):  A few-shot class incremental learning method that investigates methods to enhance the representation of class prototypes in CIL, aiming to
improve plasticity within the stability-favoring classifier-incremental setting. The method evaluates domain incremental learning as well, separately.

Although effective for class-incremental and few-shot learning, these methods are not robust to domain shifts.

# **Semi-Supervised based methods**:

These methods can handle unlabeled data.

[RETRIEVE](https://github.com/decile-team/cords): A coreset selection framework for efficient and *robust semi-supervised learning*. 

[NNCSL](https://github.com/kangzhiq/NNCSL): A nearest-neighbor-based *continual semi-supervised* method.

[UaD-CE](https://github.com/yawencui/UaD-ClE): It uses a Class Equilibrium module to address overfitting and an Uncertainty aware Distillation module to distill reliable knowledge
for memorizing previous categories and eliminate the ambiguity between previous and novel categories.

[CSL](https://github.com/PanLiuCSU/CSL): CSL proposes *pseudo-label selection* as a convex optimization problem within the confidence distribution. By doing so, CSL aims to overcome the challenges of overconfidence and context loss, ultimately enhancing the performance of semi-supervised semantic segmentation models.

These methods cannot jointly address domain shifts, class-incremental, and few-shot learning.

# **Others (Representation Learning, Few-shot Learning, Meta-Learning, Active Learning and Domain-shift)**: 

[SupCL](https://github.com/google-research/google-research/tree/master/supcon), [UnSupCL](https://github.com/google-research/simclr), and [UnSupCL-HNM](https://github.com/joshr17/HCL) are contrastive learning methods designed for supervised, unsupervised, and hard negative mining settings, respectively.

[MTL](https://github.com/AI-secure/multi-task-learning): MTL enables fast adaptation to unseen tasks through efficient training. It bridges gradient-based meta-learning and multi-task learning, making it particularly suitable for *few-shot learning* scenarios.

[MAML](https://github.com/CEA-LIST/MetaMTReg): A multi-task representation learning framework that improves meta-learning via spectral-based regularization, ideal for *few-shot learning*.

[CLIP-driven](https://github.com/ljwztc/CLIP-Driven-Universal-Model): A universal and extensible Language-Vision model for *organ segmentation* and tumor detection from abdominal computed tomography. It is a  large pre-trained CLIP-driven U-Net model, which has already been exposed to most of the 35 classes in the **Med FoSSIL-Mixed** setting.

[HALO](https://github.com/paolomandica/HALO): A hyperbolic neural network approach to pixel-level active learning for semantic segmentation under *domain shift*. 

