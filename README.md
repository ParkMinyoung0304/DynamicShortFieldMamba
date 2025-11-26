<div align="center">
<h1> Stratified layering for soil profile: Dynamic short field Mamba network </h1>

[Shaohua Zeng](https://orcid.org/0009-0001-4152-3807)<sup>1,2,†</sup>, [Zhihao Chen]()<sup>1,2,†</sup>, [Ruolan Zeng]()<sup>3</sup>, [Shuai Wang]()<sup>4</sup>, [Yang Wang]()<sup>5</sup>

<sup>1</sup> College of Computer & Information Science, Chongqing Normal University, Chongqing 401331, China  
<sup>2</sup> Chongqing Centre of Engineering Technology Research on Digital Agricultural & Service, Chongqing 401331, China  
<sup>3</sup> Chongqing Electric Power College, Chongqing 400053, China  
<sup>4</sup> Chongqing Master Station of Agricultural Technology Promotion, Chongqing 401121, China  
<sup>5</sup> Jiangjin District Agricultural Technology Promotion Center, Chongqing 404799, China  
<sup>†</sup> Equal contribution  

</div>


## 👀Introduction

This repository contains the code for our paper  
**Stratified layering for soil profile: Dynamic short field Mamba network**. [[Paper](https://doi.org/10.1016/j.compag.2025.111212)]

---

### 📌 Technical Overview

- **Technical implementation route**  
  ![](figs/Fig.1.Technical_implementation_route.png)

- **Soil profile preprocessing(vertical sub-image segmentation process)**  
  ![](figs/Fig.2.Soil_profile_preprocessing_and_vertical_sub-image_segmentation_process.png)

- **Baseline VM-UNet network**  
  ![](figs/Fig.3.VM-UNet_network_proposed.png)

- **Network structure: DSFM**  
  ![](figs/Fig.4.Overall_structure_of_DSFM.png)

- **Modules in DSFM:**  
  - Similarity Encoder of Short Field Module(SESF)  
    ![](figs/Fig.5.Similarity_Encoder_of_Short_Field_Module.png)  
  - Dynamic Position Encoder Module(DPE) 
    ![](figs/Fig.6.Dynamic_Position_Encoder_Module.png)

---

### 📊 Performance and Comparison

- **Accuracy Comparison on Full-size Soil Profile Datasets vs Sub-image Soil Profile Datasets(VM-UNet)**  
  ![](figs/Fig.7.Accuracy_comparison_of_VM-UNet_across_dataset_of_full-size_soil_profile_and_dataset_of_soil_profile_sub-images.png)

- **Accuracy Across Sub-image Sizes(DSFM)**  
  ![](figs/Fig.8.Accuracy_comparison_results_of_DSFM_network_under_different_soil_profile_sub-image_sizes.png)

- **Accuracy Comparison under Different K Parameters (DSFM)**  
  ![](figs/Fig.9.Accuracy_variation_curves_of_DSFM_network_for_different_values_of_short_field_parameter_K.png)

- **Segmentation results comparing DSFM network with mainstream networks**  
  ![](figs/Fig.10.Segmentation_results_comparing_DSFM_network_with_mainstream_networks.png)

  *(a) Soil profile image. (b) Ground truth. (c) DSFM network. (d) Swin-Transformer. (e) Twins. (f) ViT. (g) ConvNeXt. (h) DeepLabV3. (i) SegFormer. (j) ResNeSt. (k) Fast-SCNN.)*


- **Test Accuracy Curves**  
  ![](figs/Fig.11.Test_accuracy_curves_of_DSFM_network_and_mainstream_segmentation_networks.png)  


- **Ablation Results**
  ![](figs/Fig.13.Image_segmentation_results_of_the_ablation_experiments.png)

  *(a) Soil profile. (b) Ground truth. (c) DSFM network. (d) VM-UNet + SESF. (e) VM-UNet + DPE. (f) VM-UNet.*

- **Test Accuracy Curves**  
  ![](figs/Fig.12.Test_accuracy_curves_of_DSFM_network_ablation_experiments.png)




## 💡Environment

We test our codebase with `PyTorch 1.13.1 + CUDA 11.7` as well as `PyTorch 2.2.1 + CUDA 12.1`. Please install corresponding PyTorch and CUDA versions according to your computational resources. We showcase the environment creating process with PyTorch 1.13.1 as follows.

1. Create environment.
    ```shell
    conda create -n DSFM python=3.9
    conda activate DSFM
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

---

### Table 1. Comparison table of mainstream algorithms

| Method          | Backbone      | OA(%)  | mAcc(%) | mIoU(%)  | FLOPs | Param. | FPS    |
|-----------------|---------------|--------|--------|-------|-------|--------|--------|
| Swin-Transformer | Swin-T        | 55.36  | 45.72  | 32.52 | 133G  | 59.0M  | 37.6   |
| Twins           | Twins-SVT-S   | 72.26  | 65.63  | 51.73 | 128G  | 53.2M  | 37.0   |
| ViT             | ViT-B         | 46.64  | 40.64  | 26.99 | 245G  | 142.0M | 28.9   |
| ConvNeXt        | ConvNeXt-T    | 59.78  | 50.85  | 36.82 | 132G  | 59.3M  | 37.2   |
| DeepLabV3       | ResNet50      | 43.21  | 34.16  | 20.61 | 152G  | 65.8M  | 41.9   |
| SegFormer       | MiT-B0        | 70.34  | 64.54  | 50.33 | 9G    | 3.8M   | 51.3   |
| ResNeSt         | ResNeSt-101   | 58.04  | 48.96  | 34.15 | 149G  | 69.4M  | 37.0   |
| Fast-SCNN       | Fast-SCNN     | 55.02  | 37.88  | 32.09 | 2G    | 1.4M   | 102.7  |
| UTANet          | TA-MoSC       | 64.03  | 53.50  | 39.66 | 64.61G| 24.17M | 54.35  |
| EGE-UNet        | EGE-UNet      | 69.53  | 62.08  | 48.75 | ** 0.29G** |**  0.05M **  | 96.65  |
| Rolling-UNet    | Rolling-UNet  | 63.94  | 65.34  | 44.81 | 3.73G | 1.78M  | 51.50  |
|-----------------|---------------|--------|--------|-------|-------|--------|--------|
| Ours            | VMamba-T      | **78.99<sup>a</sup>** |** 69.04 ** | **60.22 ** | 22G   | 31.1M  |**  103.62**  |
*a. The best experimental results are in bold.*

---

### Table 2. IoU comparison of DSFM and mainstream networks across soil layers

| Method           | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 |
|------------------|---------|---------|---------|---------|---------|
| Swin-Transformer | 38.9    | 44.1    | 40.8    | 36.6    | 2.1     |
| Twins            | 72.7    | 56.9    | 53.6    | 57.8    | 21.1    |
| ViT              | 43.3    | 23.8    | 30.9    | 31.4    | 5.5     |
| ConvNeXt         | 53.2    | 42.8    | 41.0    | 42.5    | 4.2     |
| DeepLabV3        | 42.9    | 19.3    | 34.2    | 5.7     | 0.9     |
| SegFormer        | 67.1    | 55.3    | 50.3    | 53.3    | **25.6**    |
| ResNeSt-101      | 61.7    | 33.2    | 43.4    | 13.5    | 14.1    |
| Fast-SCNN        | 42.8    | 42.5    | 27.4    | 12.2    | 0.0     |
| UTANet           | 67.5    | 53.4    | 40.9    | 36.5    | 0.0     |
| EGE-UNet         | 67.6    | 57.1    | 50.5    | 48.2    | 20.39   |
| Rolling-UNet     | 64.9    | 54.0    | 36.2    | 41.8    | 27.29   |
| Ours             | **78.2<sup>a</sup>**   | **71.7**    | **68.2**    | **66.1**    | 16.9    |

*a. The best and second-best results for each layer are highlighted in 1 and 2.*

---

### Table 3. Comparison table of the ablation experiments

| VM-UNet | SESF | DPE | OA      | mIoU  | FLOPs  | Param. | FPS     |
|---------|------|-----|---------|-------|--------|--------|---------|
| √       | ×    | ×   | 72.01   | 49.76 | 21.50G | 31.14M | **115.21**  |
| √       | √    | ×   | 76.53   | 55.57 | 21.50G | 31.14M | 108.17  |
| √       | ×    | √   | 72.92   | 50.61 | 21.50G | 31.14M | 108.25  |
| √       | √    | √   | **78.99<sup>a</sup>**  | **60.22** | 21.50G | 31.14M | 103.62  |
*a. The best experimental results are in bold.*

## 🙏Acknowledgements

Our Mamba codes are adapted from [Mamba](https://github.com/state-spaces/mamba) and [VMamba](https://github.com/MzeroMiko/VMamba). We thank the authors for releasing their code!

## 📧Contact

If you have any questions, please  contact at [Zhihao Chen](2023210516060@stu.cqnu.edu.cn).



