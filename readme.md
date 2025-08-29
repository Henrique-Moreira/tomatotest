# Comparative Analysis of Deep Learning Architectures for Semantic Segmentation of Tomatoes

## 📋 Project Description

This repository contains the implementation and comparative analysis of three convolutional neural network (CNN) architectures for semantic segmentation of tomato fruits in field images.

## 🎯 Objective

The main objective is to determine which deep learning architecture is most effective for the semantic segmentation task of tomatoes under real field conditions, contributing to the advancement of precision agriculture through automated phenotyping.

## 📊 Dataset Used

### 1. Dataset Selection
- **Dataset**: `tomatotest` - Public dataset provided by He et al. (2024)
- **Source**: Collected at the Mountain Horticultural Crops Research and Extension Center, NC, USA
- **Available at**: [https://huggingface.co/datasets/XingjianLi/tomatotest](https://huggingface.co/datasets/XingjianLi/tomatotest)
- **Characteristics**:
  - 21,367 images captured in real field environment
  - Original resolution: 2448 × 2048 pixels
  - Captured with autonomous ground robot (HuskyBot) equipped with stereo camera system
  - Challenging conditions: variable lighting, leaf occlusion, complex backgrounds

### 2. Selection Rationale
The dataset was chosen for:
- **Adequate complexity**: Real field images with challenging conditions
- **Research relevance**: Direct application in precision agriculture
- **Data quality**: Precise annotations and well-structured dataset
- **Appropriate scale**: Significant amount of data for robust training

## 🔍 Exploratory Data Analysis

### Preprocessing Performed
1. **Format Conversion**: Extraction of HDF5 (.h5) files to PNG
2. **Mask Generation**: Creation of binary segmentation masks
3. **Resizing**: Images resized to 256×256 pixels
4. **Data Augmentation**: Application of horizontal flipping

### Statistical Analysis
- **Tomato Pixel Proportion**: Average of 0.57% per image
- **Class Imbalance**: Strong imbalance identified (many background pixels vs. few tomato pixels)
- **Data Distribution**: 80% for training, 20% for validation

### Implemented Visualizations
- Histogram of tomato pixel proportion per mask
- Evolution graphs of validation metrics per epoch
- Loss curves during training
- Qualitative examples of model predictions

## 🤖 Machine Learning Techniques

### Applied Technique: Semantic Segmentation (Deep Learning)

#### Selection Rationale
**Semantic segmentation** was chosen because:
- **Location precision**: Necessary to exactly delineate tomato contours
- **Practical application**: Essential for accurate fruit counting and yield estimation
- **Advantage over other techniques**: 
  - Image classification: Too simple (one label per image)
  - Object detection: Only bounding boxes, without precise contours
  - Semantic segmentation: Pixel-by-pixel classification, enabling precise measurements

### Implemented Algorithms

#### 1. U-Net
- **Architecture**: Encoder-decoder with skip connections
- **Advantages**: Preserves high-resolution spatial details
- **Application**: Ideal for precise localization of small objects
- **Result**: **Best performance** - IoU: 0.8265, Dice/F1: 0.9033

#### 2. DeepLabV3
- **Architecture**: Atrous convolutions with ASPP module
- **Advantages**: Multi-scale analysis without resolution loss
- **Application**: Robust for objects of varying sizes
- **Result**: IoU: 0.5793, Dice/F1: 0.7309

#### 3. PSPNet (Pyramid Scene Parsing Network)
- **Architecture**: Pyramid pooling module for global context
- **Advantages**: Understanding of overall scene structure
- **Application**: Ideal for complex scene parsing
- **Result**: IoU: 0.5528, Dice/F1: 0.7120

### Hyperparameters Used

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 100 | Adequate convergence observed |
| Optimizer | Adam | Efficient and widely used |
| Loss Function | Dice Loss + BCE | Combats class imbalance |
| Resolution | 256×256 | Compromise between details and computational feasibility |
| Augmentation | Horizontal flipping | Doubles dataset and improves generalization |

### Evaluation Metrics
- **IoU (Intersection over Union)**: Most rigorous metric for segmentation
- **Dice Coefficient (F1-Score)**: Overlap measure, more flexible than IoU
- **Precision**: Accuracy of positive predictions
- **Recall**: Completeness of positive predictions

## 📈 Main Results

### Quantitative Performance
| Model | IoU | Dice/F1 | Precision | Recall |
|-------|-----|---------|-----------|--------|
| **U-Net** | **0.8265** | **0.9033** | **0.9487** | **0.8630** |
| DeepLabV3 | 0.5793 | 0.7309 | 0.7516 | 0.7140 |
| PSPNet | 0.5528 | 0.7120 | 0.7351 | 0.6903 |

### Key Findings
1. **U-Net Superiority**: 42-50% better than other architectures
2. **Importance of Skip Connections**: Spatial detail preservation is crucial
3. **Effectiveness of Supervised Approach**: Direct learning outperforms more complex methods for this specific task

## 🗂️ Repository Structure

```
├── README.md                              # This file
├── code/                                  # Source code
│   ├── experimento1/                      # U-Net experiments
│   ├── experimento2/                      # U-Net experiments (variations)
│   ├── experimento3/                      # U-Net experiments (variations)
│   ├── experimento4/                      # U-Net experiments (variations)
│   ├── experimento5/                      # DeepLabV3 experiments
│   ├── experimento6/                      # PSPNet experiments
│   ├── data_augmentation_tomates.ipynb    # Data analysis and augmentation
│   └── redimensionar imagens para 256.py # Preprocessing
├── tomatotest/                            # Dataset and scripts
│   ├── data/                              # Raw data
│   ├── processed_data/                    # Processed data
│   └── processed_data_256/                # Resized data
└── ARTIGO BASE - High‐Throughput Robotic... # Reference article
```

## 🚀 How to Run

### Prerequisites
```bash
pip install torch torchvision opencv-python matplotlib numpy pandas
```

### Running the Experiments
1. **Preprocessing**: Run `redimensionar imagens para 256.py`
2. **Exploratory Analysis**: Open `data_augmentation_tomates.ipynb`
3. **U-Net Training**: Run notebooks in `experimento1/` to `experimento4/`
4. **DeepLabV3 Training**: Run notebook in `experimento5/`
5. **PSPNet Training**: Run notebook in `experimento6/`

## 📊 Results Analysis

### Why was U-Net Superior?
1. **Architectural Alignment**: Skip connections ideal for precise localization
2. **Detail Preservation**: Maintains high-resolution spatial information
3. **Task Suitability**: Tomatoes are small objects requiring precise delineation
4. **Local Features**: More important than global context in this application

### Limitations and Future Work
- Explore more sophisticated augmentation techniques
- Test more modern backbones (ResNet, EfficientNet)
- Systematic hyperparameter optimization
- Validation on other agricultural datasets

## 🏆 Credits and References

### Dataset
This project uses the `tomatotest` dataset created and provided by:
- **Authors**: Weilong He, Xingjian Li, Zhenghua Zhang, Yuxi Chen, Jianbo Zhang, Dilip R. Panthee, Inga Meadows, Lirong Xiang
- **Available at**: [https://huggingface.co/datasets/XingjianLi/tomatotest](https://huggingface.co/datasets/XingjianLi/tomatotest)

### Base Article
The work is based on the scientific article:
**"High-Throughput Robotic Phenotyping for Quantifying Tomato Disease Severity Enabled by Synthetic Data and Domain-Adaptive Semantic Segmentation"**

**Authors**: 
- Weilong He¹'²
- Xingjian Li²'³ 
- Zhenghua Zhang¹'²
- Yuxi Chen³
- Jianbo Zhang⁴
- Dilip R. Panthee
- Inga Meadows
- Lirong Xiang¹'²

## 👥 Contributions

This project was developed as part of a Master's research in Computer Science at the Federal University of Uberlândia, under specific academic guidance for the Data Mining course.

## 📄 License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

---

**Contact**: henriquemoreiraa@gmail.com  
**Institution**: Federal University of Uberlândia  
**Program**: Master's in Computer Science