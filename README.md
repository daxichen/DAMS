# DAMS
# Decoupling and Aligning Modality-Shared Semantics for Single-Domain Generalization in Hyperspectral Image Classification
![DAMS Framework](figure/DAMS.png)
DSPLTnet comprises three main components designed for robust cross-scene HSI classification. **1.** The spectral patch low-frequency transformation network minimizes the similarity between original spectral patch features and their low-frequency perturbed counterparts, thereby encouraging the model to focus on class-essential semantic information within the low-frequency components of SD HSIs. **2.** A novel domain-agnostic DAG is constructed to enable the model to identify domain-invariant features through causal structure learning, concurrently pruning domain-specific and spurious features detrimental to model generalization. **3.** A progressive contrastive learning framework is employed to obtain more representative class-conditional prototypes, thereby effectively searching for the domain-agnostic DAG from them. Notably, during inference, the frequency domain filtering branch, the learned domain-agnostic DAG, and the classifier are utilized for cross-scene HSI classification.

# Requirements：
```
1. torch==1.11.0+cu113
2. python==3.8.3
3. ptflops==0.6.9
4. timm==0.5.4
```
# Dataset:
The dataset can be downloaded from here: [HSI datasets](https://github.com/YuxiangZhang-BIT/Data-CSHSI). We greatly appreciate their outstanding contributions.

The dataset directory should look like this:
```
datasets
  Houston
  ├── Houston13.mat
  ├── Houston13_7gt.mat
  ├── Houston18.mat
  └── Houston18_7gt.mat
```

# Usage:
Houston datasets:
```
python inference.py --save_path ./results/ --data_path ./datasets/Houston/ --target_name Houston18 --patch_size 8
```
