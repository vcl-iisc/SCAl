<div align="center">

## 🚀 FedSCAl: Leveraging Server and Client Alignment for Unsupervised Federated Source-Free Domain Adaptation (WACV-26 🎉)<br> [webpage](https://vcl-iisc.github.io/FedSCAl/) | [paper](https://www.arxiv.org/pdf/2512.06738) | [video](https://vcl-iisc.github.io/FedSCAl/)<br><br> <p align="left">🎯 Overview</p>
</div>

An FL framework leveraging our proposed Server-Client Alignment (SCAl) mechanism to regularize client updates by aligning the clients’ and server model’s predictions.

This repository supports:

* Source training
* SCAl: Federated source-Free Domain Adaptation (FFreeDA) 
* SCAL with adaptive or fixed thresholds
* Baseline (no SCAl)
* BMD (as base sfda)
* ViT-Small and ViT-Base backbones
* OfficeHome and DomainNet-Small dataset experiments

## 📂 Directory Structure

```
./data/
    └── <dataset_folder>
        ├── OfficeHome
        ├── DomainNetS
        └── ...
```

Ensure all datasets are downloaded inside `./data/<dataset_folder>`.

---

##  Step 1: Train Server (Source) Models

```bash
python train_centralDA_source.py \
    --domain_s 'art' --domain_u 'clipart' \
    --model_name VITs --d_mode new --device cuda:0 \
    --var_lr 0.03 --scheduler_name 'CosineAnnealingLR' --cycles 50 \
    --control_name "1_sup-ft-fix_15_0.3_iid_5-5_0.07_1" \
    --resume_mode 1 --init_seed 2020 --backbone_arch vit-small
```

```bash
python train_centralDA_sourceOC.py \
    --data_name 'DomainNetS' --domain_s 'clipart' \
    --data_name_unsup 'DomainNetS' --domain_u 'infograph' \
    --model_name VITs --d_mode new --device cuda:0 \
    --var_lr 0.1 --scheduler_name 'CosineAnnealingLR' --cycles 50 \
    --control_name "1_sup-ft-fix_8_0.5_iid_5-5_0.07_1" \
    --resume_mode 1 --init_seed 2020 --backbone_arch vit-b
```

---

## Step 2: Run SCAl

### 🔵 SCAl on OfficeHome

```bash
python train_classifier_ssDA_target.py \
    --domain_s 'art' --unsup_doms 'product-clipart-realworld' \
    --model_name VITs --d_mode new --device cuda:1 \
    --var_lr 0.03 --scheduler_name 'CosineAnnealingLR' --cycles 60 \
    --control_name "1110022_sup-ft-fix_15_0.3_iid_5-5_0.07_1" \
    --resume_mode 1 --init_seed 2025 --run_shot 1 --par 1 \
    --tag_ "2020_art_0.03_VITs_1_sup-ft-fix" \
    --client_test 1 --pick 'checkpoint' --add_fix 1 \
    --backbone_arch vit-small --lambda 3 --g_lambda 0.3 \
    --global_reg 1 --adpt_thr 1
```

### 🔵 SCAL on DomainNet-Small

```bash
python train_classifier_ssDA_target_DN.py \
    --data_name 'DomainNetS' --data_name_unsup 'DomainNetS' \
    --domain_s 'sketch' \
    --unsup_doms 'clipart-quickdraw-real-painting-infograph' \
    --model_name VITs --d_mode new --device cuda:0 \
    --var_lr 0.03 --scheduler_name 'CosineAnnealingLR' --cycles 50 \
    --control_name "99001_sup-ft-fix_10_0.3_iid_5-5_0.07_1" \
    --resume_mode 1 --init_seed 2020 --run_shot 1 --par 1 \
    --tag_ "2020_sketch_0.1_VITs_991_sup-ft-fix" \
    --client_test 1 --pick 'checkpoint' --backbone_arch vit-small
```


### ⭐ SCAL Variants

| SCAL Variants | added arguments |
|--------|------------|
| Fixed Threshold Instead of Adaptive   | `--adpt_thr 0 --threshold <set_threshold>` |
| Client Alignment Only (No Global Regularizer)   | `--global_reg 0` |
| Baseline (No SCAl) | `--add_fix 0` |
| BMD(as base sfda) | `--run_shot 0 --add_fix 1 --global_reg 1 --adpt_thr 1` |
| ViT-Base Backbone | `--backbone_arch vit-b` |

---

## 🛠️ Tips for Reproducibility
* Keep `init_seed` identical across runs for consistent results.
* Use the same `control_name` when resuming.
* Verify dataset folder structure before training.
* Ensure GPU selection (`cuda:0, cuda:1, ...`) matches your hardware.


## ✏️ Citation
If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```bibtex
@InProceedings{yashwanth_2026_WACV,
        author    = {Yashwanth, M and Koti, Sampath and Singh, Arunabh and Marjit, Shyam and Chakraborty, Anirban},
        title     = {FedSCAl: Leveraging Server and Client Alignment for Unsupervised Federated Source-Free Domain Adaptation},
        booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
        year      = {2026},
    }
```
## ✉️ Contact

M. Yashwanth: yashwanth06904@gmail.com or yashwanthm@iisc.ac.in
