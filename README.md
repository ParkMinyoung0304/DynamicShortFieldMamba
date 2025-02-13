<div align="center">
<h1> Soma: Soil Mamba Network for Soil Profile Segmentation </h1>

[Shaohua Zeng]()<sup>1</sup> <sup>2</sup>, [Zhihao Chen]()<sup>1</sup> <sup>2</sup></sup>

<sup>1</sup>  School of Computer and Information Science, Chongqing Normal University, Chongqing, China  
<sup>2</sup>  Chongqing Center of Engineering Technology Research on Digital Agricultural Service, Chongqing, China

[![arXiv](https://img.shields.io/badge/arXiv-2404.04256-b31b1b.svg)](https://arxiv.org/abs/2404.04256) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![X](https://img.shields.io/twitter/url/https/twitter.com/bukotsunikki.svg)](https://x.com/_akhaliq/status/1777272323504025769)

</div>


## 👀Introduction

This repository contains the code for our paper `Soma: Soil Mamba Network for Soil Profile Segmentation`. [[Paper](https://arxiv.org/abs/2404.04256)]

![](figs/soma.png)

![](figs/overall_flops.png)

## 💡Environment

We test our codebase with `PyTorch 1.13.1 + CUDA 11.7` as well as `PyTorch 2.2.1 + CUDA 12.1`. Please install corresponding PyTorch and CUDA versions according to your computational resources. We showcase the environment creating process with PyTorch 1.13.1 as follows.

1. Create environment.
    ```shell
    conda create -n soma python=3.9
    conda activate soma
    ```

2. Install all dependencies.
Install pytorch, cuda and cudnn, then install other dependencies via:
    ```shell
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
    ```shell
    pip install -r requirements.txt
    ```

3. Install Mamba
    ```shell
    cd models/encoders/selective_scan && pip install . && cd ../../..
    ```

## ⏳Setup

### Datasets

1. We use Two datasets, including both Surface Soil and Soil Profile datasets:
    - [Surface Soil](Data Privacy)
    - [Soil Profile](Data Privacy)

2. If you are using your own datasets, please organize the dataset folder in the following structure:  

    ```shell
    <data>
    |-- cutting
    |   |-- images
    |   |   |-- <name1>.<ImageFormat>
    |   |   |-- <name2>.<ImageFormat>
    |   |   ...
    |   |-- masks  # Grayscale segmentation masks
    |       |-- <name1>.<LabelFormat>
    |       |-- <name2>.<LabelFormat>
    |       ...
    |-- entire_img
    |   |-- images
    |   |   |-- <name1>.<ImageFormat>
    |   |   |-- <name2>.<ImageFormat>
    |   |   ...
    |   |-- masks  # Grayscale segmentation masks
    |       |-- <name1>.<LabelFormat>
    |       |-- <name2>.<LabelFormat>
    |       ...
    |-- mix
    |   |-- images
    |   |   |-- <name1>.<ImageFormat>
    |   |   |-- <name2>.<ImageFormat>
    |   |   ...
    |   |-- masks  # Grayscale segmentation masks
    |       |-- <name1>.<LabelFormat>
    |       |-- <name2>.<LabelFormat>
    |       ...
    ```

📌 **Note:**  
- **All masks must be in grayscale format (single-channel).**  
- **Recommended formats:** `.png`, `.jpg`, `.tif` for images, and `.png` or `.tif` for masks.  
- **Ensure masks are properly labeled with distinct grayscale values representing different classes.**

## 📦Usage

### Training  
1. This model requires a pre-trained ResNet checkpoint. Please download it from [ResNet](https://pytorch.org/vision/stable/models.html#id2).  

    🔗 [Torchvision Official Pretrained Model](https://pytorch.org/vision/stable/models.html#id2)  
   
    📥 **Directly Download ResNet-50 Weight**：  
    - [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth)  

    <u> Please put them under `model/short_range_vmamba_unet/`. </u>  

2. **Start Training**  
    Run the following command to start training:  
    ```bash
    python train_vmamba_cutting_pre_entire.py
    ```
    This script will launch the training process with default configurations.  

3. **Other Training Modes**  
    If you want to experiment with different training approaches, you can run other training scripts in the repository.  
    Example:  
    ```bash
    python train_vmamba_entire_pre_entire.py
    ```
    Modify the script parameters as needed to customize the training process.

4. Results will be saved in `logs` folder and `model_pth` folder.

5. Testing and Visualization  

    After training the model, you can perform inference on test images and visualize the results using the provided script.  
    
    ### **Run Inference on Test Data**  
    To test the trained model on the dataset, execute the following command:  
    ```bash
    python model/short_range_vmamba_unet/test.py
    ```
    
    ### **IoU Evaluation for Each Class**  
    The model also provides per-class IoU (Intersection over Union) evaluation to assess segmentation performance.
    To compute IoU for each class, run:
   ```bash
   python model/short_range_vmamba_unet/test_iou.py
    ```

    The visualization results will be automatically saved in the following folder:local_visualizations/


## 📈Results

We provide our result on the Soil Profile datasets:

### MFNet (5 categories)
| Architecture | OA | mAcc | mIOU |
|:---:|:---:|:---:|:---:|
| Soma | 78.99 | 69.04% |60.22% |

## 🙏Acknowledgements

Our dataloader codes are based on [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation). Our Mamba codes are adapted from [Mamba](https://github.com/state-spaces/mamba) and [VMamba](https://github.com/MzeroMiko/VMamba). We thank the authors for releasing their code!
We also appreciate [DFormer](https://github.com/VCIP-RGBD/DFormer?tab=readme-ov-file) for providing their processed RGB-Depth datasets.

## 📧Contact

If you have any questions, please  contact at [Zhihao Chen](2023210516060@stu.cqnu.edu.cn).

## 📌 BibTeX & Citation

If you find this code useful, please consider citing our work:

```bibtex
@article{wan2024sigma,
  title={Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation},
  author={Wan, Zifu and Wang, Yuhao and Yong, Silong and Zhang, Pingping and Stepputtis, Simon and Sycara, Katia and Xie, Yaqi},
  journal={arXiv preprint arXiv:2404.04256},
  year={2024}
}
```

