

# CARE

## A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis

<div align="center">

[![a](https://img.shields.io/badge/Model-huggingface-blue)](https://huggingface.co/Zipper-1/CARE)
[![arXiv](https://img.shields.io/badge/Arxiv-2602.21637-red
)](https://arxiv.org/abs/2602.21637)

</div>

## 🔥 Update
- [2026.03.11] The model weight was released in [Hugging Face](https://huggingface.co/Zipper-1/CARE).
- [2026.02.21] Our paper was accepted by CVPR 2026.


## Data Preprocessing
We follow the CLAM WSI preprocessing pipeline. CARE builds a feature grid from CONCH v1.5 patch features using patch coordinates. We store both the extracted features and their corresponding coordinates in a single `.npy` file.
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
coords = torch.from_numpy(coords).unsqueeze(0).to(device)
```

More specifically, if you use the CLAM `.h5` file format, please follow the usage instructions on [Hugging Face](https://huggingface.co/Zipper-1/CARE).




## CARE WSI Feature Extraction
We provide an example of how to extract WSI features using CARE in the file `care_wsi_encoder_api.py`. The pretrained CARE weight is released on Hugging Face.



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


## Result

**Bold** indicates the best result, <u>underline</u> indicates the second-best result, and all reported results are obtained with logistic regression.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">EBRAINS <br> 30 classes</th>
      <th colspan="2">Combine-Lung <br>subtype 2 classes</th>
      <th colspan="2">Cross-LUNG-fine <br> 3 classes</th>
      <th colspan="2">MUT-BAP1 <br> 2 classes</th>
      <th colspan="2">EBRAINS-IDH <br> 2 classes</th>
      <th colspan="2">CCRCC <br> OS</th>
    </tr>
    <tr>
      <th>ACC</th>
      <th>F1</th>
      <th>ACC</th>
      <th>AUC</th>
      <th>ACC</th>
      <th>F1</th>
      <th>ACC</th>
      <th>AUC</th>
      <th>ACC</th>
      <th>AUC</th>
      <th>C-index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mean-pool</td>
      <td>65.8</td>
      <td>71.4</td>
      <td>88.1</td>
      <td>95.4</td>
      <td>63.6</td>
      <td>73.7</td>
      <td>61.0</td>
      <td>84.8</td>
      <td>87.8</td>
      <td>94.2</td>
      <td>48.4</td>
    </tr>
    <tr>
      <td>CHIEF</td>
      <td>60.6</td>
      <td>68.4</td>
      <td>87.6</td>
      <td>95.6</td>
      <td><strong>65.7</strong></td>
      <td><u>74.2</u></td>
      <td>57.2</td>
      <td>85.9</td>
      <td>88.9</td>
      <td>95.6</td>
      <td>53.2</td>
    </tr>
    <tr>
      <td>PRISM</td>
      <td>59.5</td>
      <td>65.8</td>
      <td>86.2</td>
      <td>95.2</td>
      <td>58.3</td>
      <td>68.0</td>
      <td>57.5</td>
      <td>86.5</td>
      <td>88.7</td>
      <td>94.9</td>
      <td>38.0</td>
    </tr>
    <tr>
      <td>GigaPath</td>
      <td>64.7</td>
      <td>71.6</td>
      <td>88.4</td>
      <td>95.6</td>
      <td>54.2</td>
      <td>62.2</td>
      <td><u>61.4</u></td>
      <td>86.6</td>
      <td>88.2</td>
      <td>94.7</td>
      <td>55.8</td>
    </tr>
    <tr>
      <td>TANGLE</td>
      <td>64.5</td>
      <td>71.2</td>
      <td>86.3</td>
      <td>94.7</td>
      <td>60.3</td>
      <td>70.1</td>
      <td><strong>63.6</strong></td>
      <td>86.8</td>
      <td>89.0</td>
      <td>95.2</td>
      <td>46.6</td>
    </tr>
    <tr>
      <td>FEATHER</td>
      <td>68.2</td>
      <td>73.1</td>
      <td>85.5</td>
      <td>94.1</td>
      <td>56.2</td>
      <td>68.5</td>
      <td>54.7</td>
      <td>84.3</td>
      <td>88.5</td>
      <td>95.9</td>
      <td><u>56.6</u></td>
    </tr>
    <tr>
      <td>TITAN</td>
      <td><strong>74.8</strong></td>
      <td><strong>78.8</strong></td>
      <td><strong>89.2</strong></td>
      <td><u>96.6</u></td>
      <td>63.8</td>
      <td><u>74.2</u></td>
      <td>59.8</td>
      <td>86.0</td>
      <td><u>91.4</u></td>
      <td><strong>96.7</strong></td>
      <td>40.5</td>
    </tr>
    <tr>
      <td><strong>CARE</strong></td>
      <td><u>74.0</u></td>
      <td><u>78.7</u></td>
      <td><u>89.0</u></td>
      <td><strong>96.8</strong></td>
      <td><u>65.5</u></td>
      <td><strong>74.4</strong></td>
      <td><u>61.4</u></td>
      <td><strong>88.9</strong></td>
      <td><strong>91.5</strong></td>
      <td><u>96.6</u></td>
      <td><strong>63.0</td>
    </tr>
  </tbody>
</table>


## Citation
If you find our work useful in your research, please consider citing CARE and CONCH v1.5:

```
@article{zhang2026care,
  title={CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis},
  author={Zhang, Di and Gong, Zhangpeng and Pang, Xiaobo and Liu, Jiashuai and Lu, Junbo and Cui, Hao and Ge, Jiusong and Zeng, Zhi and Yi, Kai and Li, Yinghua and others},
  journal={arXiv preprint arXiv:2602.21637},
  year={2026}
}
```
```
@article{ding2025multimodal,
  title={A multimodal whole-slide foundation model for pathology},
  author={Ding, Tong and Wagner, Sophia J and Song, Andrew H and Chen, Richard J and Lu, Ming Y and Zhang, Andrew and Vaidya, Anurag J and Jaume, Guillaume and Shaban, Muhammad and Kim, Ahrong and others},
  journal={Nature Medicine},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
