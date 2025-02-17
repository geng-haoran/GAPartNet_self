<h2 align="center">
  <b>GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts</b>

  <b><i>CVPR 2023 Highlight</i></b>


<div align="center">
    <a href="https://cvpr.thecvf.com/virtual/2023/poster/22552" target="_blank">
    <img src="https://img.shields.io/badge/CVPR 2023-Highlight-red"></a>
    <a href="https://arxiv.org/abs/2211.05272" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://pku-epic.github.io/GAPartNet/" target="_blank">
    <img src="https://img.shields.io/badge/Page-GAPartNet-blue" alt="Project Page"/></a>
</div>
</h2>

This is the official repository of [**GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts**](https://arxiv.org/abs/2211.05272).

For more information, please visit our [**project page**](https://pku-epic.github.io/GAPartNet/).

## GAPartNet Dataset

GAPartNet Dataset (PartNet-Mobility part, 1045/1166 objects) has been released.

The AKB-48 part (121/1166 objects) will be released in recent days.

To obtain our dataset, please fill out [**this form**](https://forms.gle/3qzv8z5vP2BT5ARN7) and check the [**Terms&Conditions**](https://docs.google.com/document/d/1kjFCTcDLtaycZiJVmSVhT9Yw8oCAHl-3XKdJapvRdW0/edit?usp=sharing). Please cite our paper if you use our dataset.

## GAPartNet Network and Inference

We release our network and checkpoint, check gapartnet folder for more details. You can segment part 
and estimate the pose of it. We also provide visualization code. This is an visualization example:
![example](gapartnet/output/example.png)

## How to use our code and model: 

### 1. Install dependencies
  - Python 3.8
  - Pytorch >= 1.11.0
  - CUDA >= 11.3
  - Open3D with extension (See install guide below)
  - epic_ops (See install guide below)
  - pointnet2_ops (See install guide below)
  - other pip packages

### 2. Install Open3D & epic_ops & pointnet2_ops
  See this repo for more details:
  
  [GAPartNet_env](https://github.com/geng-haoran/GAPartNet_env): This repo includes Open3D, [epic_ops](https://github.com/geng-haoran/epic_ops) and pointnet2_ops. You can install them by following the instructions in this repo.

### 3. Download our model and data
  See gapartnet folder for more details.

### 4. Run the code
  ```
  cd gapartnet
  python tools/visu.py  
  ```

## Citation
If you find our work useful in your research, please consider citing:

```
@article{geng2022gapartnet,
  title={GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts},
  author={Geng, Haoran and Xu, Helin and Zhao, Chengyang and Xu, Chao and Yi, Li and Huang, Siyuan and Wang, He},
  journal={arXiv preprint arXiv:2211.05272},
  year={2022}
}
```

## Contact
If you have any questions, please open a github issue or contact us:

Haoran Geng: ghr@stu.pku.edu.cn

Helin Xu: xuhelin1911@gmail.com

Chengyang Zhao: zhaochengyang@pku.edu.cn

He Wang: hewang@pku.edu.cn
