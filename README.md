

# CARE

## A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis



<div align="center">

[![a](https://img.shields.io/badge/Download_Model-huggingface-blue)](https://huddingface/)
[![arXiv](https://img.shields.io/badge/Arxiv-2602.21637-red
)](https://arxiv.org/abs/2602.21637)

</div>

## Abstract
Foundation models have recently achieved impressive success in computational pathology, demonstrating strong generalization across diverse histopathology tasks. However, existing models overlook the heterogeneous and non-uniform organization of pathological regions of interest (ROIs) because they rely on natural image backbones not tailored for tissue morphology. Consequently, they often fail to capture the coherent tissue architecture beyond isolated patches, limiting interpretability and clinical relevance. To address these challenges, we present Cross-modal Adaptive Region Encoder (CARE), a foundation model for pathology that automatically partitions WSIs into several morphologically relevant regions. Specifically, CARE employs a two-stage pretraining strategy: (1) a self-supervised unimodal pretraining stage that learns morphological representations from 34,277 whole-slide images (WSIs) without segmentation annotations, and (2) a cross-modal alignment stage that leverages RNA and protein profiles to refine the construction and representation of adaptive regions. This molecular guidance enables CARE to identify biologically relevant patterns and generate irregular yet coherent tissue regions, selecting the most representative area as ROI. CARE supports a broad range of pathology-related tasks, using either the ROI feature or the slide-level feature obtained by aggregating adaptive regions. Based on only one-tenth of the pretraining data typically used by mainstream foundation models, CARE achieves superior average performance across 33 downstream benchmarks, including morphological classification, molecular prediction, and survival analysis, and outperforms other foundation model baselines overall.

## 🔥 Update
- [2026.02.21] Our paper was accepted by CVPR2026.

## Data Preprocess
we follow the CLAM's WSI preprocessing solution. To satisfy CARE’s requirement for patch coordinate information, we store both the extracted features and their corresponding coordinates in a single `.npy` file.
The saving format is as follows:

```python
features_list = []
indexs_list = []
inst_labels_list = []
# Get patch feature
features = features.cpu().numpy().astype(np.float32)
features_list.append(features)
indexs_list += coords
inst_labels_list += inst_labels
asset_dict = {
    "feature": features_list,
    "index": indexs_list,
    "inst_label": inst_labels_list,
}
np.save(output_path, asset_dict)
```


The data can be loaded with the following format:

```python
fea = np.load("./data/MUT/conch_v1_5/19579_0_1024.npy", allow_pickle=True)

features = fea[()]["feature"]
cor = fea[()]["index"]
# Parse patch coordinates from the index strings
coords = np.array(
    [filename.split("_")[:2] for filename in cor],
    dtype=int,
)
# Convert features and coordinates to tensors
patch_embedding = torch.from_numpy(features).unsqueeze(0).to(device)
coords = torch.from_numpy(coords).unsqueeze(0)
```

## CARE WSI Feature Extraction
We provide an example of how to extract WSI features using CARE in the file `care_wsi_encoder_api.py`. The pretrained CARE weights will be released on Hugging Face. The model weights will be released in the coming days.


Moreover, we can obtain ROI features using the following code, as proposed in this paper:
```python
roi_embedding = model.get_roi_features(patch_embedding, N_values, coords)
```



## Linear Probing Evaluation

We provide code for linear probing with two classifiers, **KNN** and **logistic_regression**, in `train_wsi_model.py`.

For the logistic regression setting, you can run:

```bash
python -u train_wsi_model.py \
  --gpu 1 \
  --task t1_gene \
  --dataset MUT \
  --experiment_target BAP1 \
  --model_name CARE \
  --model_type logistic_regression \
  --num_region 8 \
  --data_root_dir ./data \
```

For the KNN setting, you can run:

```bash
python -u train_wsi_model.py \
  --gpu 1 \
  --task t1_gene \
  --dataset MUT \
  --experiment_target BAP1 \
  --model_name CARE \
  --model_type KNN \
  --num_region 8 \
  --data_root_dir ./data \
```

**Note:** Before running the linear probing evaluation, please first extract CONCH v1.5 patch features using the CLAM codebase, and save the features together with their coordinates in `.npy` format.

If you found this work useful, please consider citing:
```bibtex
@misc{zhang2026care,
      title={CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis}, 
      author={Di Zhang and Zhangpeng Gong and Xiaobo Pang and Jiashuai Liu and Junbo Lu and Hao Cui and Jiusong Ge and Zhi Zeng and Kai Yi and Yinghua Li and Si Liu and Tingsong Yu and Haoran Wang and Mireia Crispin-Ortuzar and eimiao Yu and Chen Li and Zeyu Gao},
      year={2026},
      eprint={2602.21637},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.21637}, 
}
```




