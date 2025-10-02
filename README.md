# 🩺 Diabetic Retinopathy Detection using Deep Learning

This project focuses on **Diabetic Retinopathy (DR) detection** from retinal fundus images using **deep learning**.  
We trained multiple **CNN architectures (AlexNet, VGG16, ResNet50, EfficientNet)** and finally built an **ensemble** of the best models to achieve high classification accuracy.  

---

##  Project Overview  

- **Task**: Classify fundus images into **5 stages of Diabetic Retinopathy**:  
  - No_DR  
  - Mild  
  - Moderate  
  - Severe  
  - Proliferative_DR  

- **Dataset**: [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)  
  - Total Images: **3662 retinal fundus images**  
  - Labeled by expert ophthalmologists  
  - Training/validation split used  

- **Approach**:  
  1. Data preprocessing & augmentation  
  2. Training different CNN models  
  3. Evaluation using accuracy, precision, recall, and F1-score  
  4. Ensemble learning of the top-performing models  

---

##  Pipeline  

1. **Data Preprocessing**
   - Images resized to fixed input size (depending on model, e.g., 224×224 for ResNet/VGG).  
   - Normalization applied.  
   - Data augmentation: horizontal/vertical flip, rotation, zoom, brightness adjustment.  

2. **Model Training**  
   - CNN Architectures:  
     - AlexNet  
     - VGG16  
     - ResNet50  
     - EfficientNet  
   - Optimizer: **Adam**  
   - Loss: **CrossEntropyLoss**  
   - Regularization: **EarlyStopping** to prevent overfitting  

3. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
   - Confusion Matrix  

4. **Ensemble**  
   - Combined predictions of **ResNet50 + EfficientNet**  
   - Gave best balance across all classes  

---

##  Results  

### 🔹 AlexNet
- Accuracy: **72.95%**  
- Weak generalization on minority classes.  

### 🔹 VGG16
- Accuracy: **65.57%**  
- Good on **No_DR**, struggled with Mild/Severe classes.  

### 🔹 ResNet50
- Accuracy: **78.14%**  
- High recall for Mild and No_DR classes.  

### 🔹 EfficientNet
- Accuracy: **80.33%**  
- Balanced results across all classes.  

### 🔹 **Ensemble (ResNet50 + EfficientNet)**
- **Final Accuracy: 82.65%**  
- Strongest performance overall.  

---

##  Classification Report (Ensemble)  

| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Mild             | 0.56      | 0.82   | 0.67     |
| Moderate         | 0.79      | 0.69   | 0.73     |
| No_DR            | 0.98      | 0.98   | 0.98     |
| Proliferative_DR | 0.66      | 0.54   | 0.60     |
| Severe           | 0.47      | 0.49   | 0.48     |

- **Macro Avg** → Precision: 0.69 | Recall: 0.70 | F1: 0.69  
- **Weighted Avg** → Precision: 0.84 | Recall: 0.83 | F1: 0.83  

---

##  Visualizations  

### Training Accuracy & Loss  
 
### AlexNet
![Training Curves](assets\AlexNet_accuracy.png)  
![Loss](assets\AlexNet_loss.png)

### VGGNet 
![Training Curves](assets\VGG16_accuracy.png)  
![Loss](assets\VGG16_loss.png)

### ResNet50

![Training Curves](assets\ResNet50_accuracy.png)  
![Loss](assets\ResNet50_loss.png)

### EfficientNet

![Training Curves](assets\EfficientNet_accuracy.png)  
![Loss](assets\EfficientNet_loss.png)

---

##  Tech Stack  

- **Language**: Python  
- **Frameworks**: PyTorch  
- **Libraries**:  
  - NumPy, Pandas  
  - scikit-learn (metrics)  
  - Matplotlib, Seaborn (visualization)  

---

##  Future Work  

- Handle **class imbalance** using weighted loss, focal loss, or oversampling  
- Explore **Vision Transformers (ViT)** for medical imaging  
- Deploy as a **FastAPI/Flask web app** for real-time inference  

---

##  References  

- [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection)  
- PyTorch Documentation  
- "Deep Learning for Medical Image Analysis", Elsevier  

---


