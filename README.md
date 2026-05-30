# Image Classification with CNN, LSTM & Transfer Learning (VGG16)

> A deep learning project implementing **three different neural network architectures** — CNN, LSTM, and Transfer Learning with VGG16 — to classify images from the MNIST digits and CIFAR-10 datasets using TensorFlow/Keras.

---

##  Project Overview

This project explores and compares multiple deep learning approaches for image classification on two benchmark vision datasets:

- **MNIST** — 70,000 grayscale images of handwritten digits (0–9), 28×28 pixels
- **CIFAR-10** — 60,000 colour images across 10 object classes, 32×32×3 pixels

Three architectures are implemented and evaluated:
1. **Custom CNN** — lightweight convolutional network designed for MNIST (>96% accuracy)
2. **Deeper CNN with Dropout** — regularized convolutional network for CIFAR-10 (>60% accuracy)
3. **LSTM** — recurrent model treating image rows as sequences
4. **Transfer Learning (VGG16)** — pretrained ImageNet weights fine-tuned on MNIST

---

##  Model Architectures

### Model 1 — CNN for MNIST

```
Layer          Type             Output Shape     Params
─────────────────────────────────────────────────────────
layer1         Conv2D (6, 3×3)  (None, 28, 28, 6)    60
layer2         MaxPooling2D     (None, 14, 14, 6)     0
layer3         Conv2D (3, 3×3)  (None, 14, 14, 3)   165
layer4         MaxPooling2D     (None, 7, 7, 3)       0
layer5         Flatten          (None, 147)            0
layer6         Dense (128)      (None, 128)        18,944
layer7         Dense (10)       (None, 10)          1,290
─────────────────────────────────────────────────────────
Total Params: 20,459
```

- **Activation:** ReLU (hidden layers), Softmax (output)
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 5 | **Batch Size:** 128
- **Target Accuracy:** > 96% on MNIST

---

### Model 2 — Deeper CNN for CIFAR-10

```
Layer          Type              Output Shape        Params
──────────────────────────────────────────────────────────────
layer1         Conv2D (16, 3×3)  (None, 32, 32, 16)     448
layer2         Conv2D (32, 3×3)  (None, 32, 32, 32)   4,640
layer3         MaxPooling2D      (None, 16, 16, 32)       0
layer4         Dropout (0.2)     (None, 16, 16, 32)       0
layer5         Conv2D (64, 3×3)  (None, 16, 16, 64)  18,496
layer6         Conv2D (64, 3×3)  (None, 16, 16, 64)  36,928
layer7         MaxPooling2D      (None, 8, 8, 64)         0
layer8         Dropout (0.2)     (None, 8, 8, 64)         0
layer9         Flatten           (None, 4096)              0
layer10        Dense (128)       (None, 128)        524,416
layer11        Dense (10)        (None, 10)           1,290
──────────────────────────────────────────────────────────────
Total Params: 586,218
```

- **Regularization:** Dropout (rate = 0.2) after each MaxPooling block
- **Target Accuracy:** > 60% on CIFAR-10

---

### Model 3 — LSTM on MNIST

Treats each row of the image (28 pixels) as a timestep in a sequence.

```
Layer          Type         Output Shape     Params
──────────────────────────────────────────────────────
LSTM (128)     Seq=True     (None, 28, 128)  80,384
Dropout (0.2)               (None, 28, 128)       0
LSTM (64)                   (None, 64)        49,408
Dropout (0.2)               (None, 64)             0
Dense (64)     ReLU         (None, 64)         4,160
Dense (10)     Softmax      (None, 10)           650
──────────────────────────────────────────────────────
```

- **Input shape:** (28, 28) — 28 timesteps of 28 features
- **Epochs:** 10 | **Batch Size:** 64

---

### Model 4 — Transfer Learning with VGG16

Uses pretrained **VGG16 weights from ImageNet** as a frozen feature extractor, with custom classification layers added on top.

```
Base Model  : VGG16 (weights=imagenet, include_top=False)
Input Shape : (32, 32, 3) — MNIST resized + converted to RGB
──────────────────────────────────────────────────────────
VGG16 Base  →  Flatten  →  Dense(512, ReLU)  →  Dense(10, Softmax)
──────────────────────────────────────────────────────────
```

- MNIST images resized from 28×28 → 32×32
- Grayscale converted to RGB for VGG16 compatibility
- **Epochs:** 5 | **Batch Size:** 64

---

## 📊 Results Summary

| Model | Dataset | Val Accuracy | Epochs |
|---|---|---|---|
| Custom CNN | MNIST | **> 96%** | 5 |
| Deeper CNN + Dropout | CIFAR-10 | **> 60%** | 5 |
| LSTM | MNIST | ~98% | 10 |
| VGG16 Transfer Learning | MNIST | ~99% | 5 |

---

## 📁 Project Structure

```
mnist-cifar-cnn/
│
├── MNIST_and_CIFAR_.ipynb     # Full training notebook (all 4 models)
├── Model1mnist.h5             # Saved MNIST CNN model (generated after training)
├── Model1cifar10.h5           # Saved CIFAR-10 CNN model (generated after training)
└── README.md
```

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/bhushangavali/mnist-cifar-cnn.git
cd mnist-cifar-cnn
```

### 2. Install dependencies
```bash
pip install tensorflow numpy matplotlib
```

### 3. Run the notebook
```bash
jupyter notebook MNIST_and_CIFAR_.ipynb
```

Datasets are auto-downloaded by Keras on first run — no manual download needed.

---

## 📈 Expected Output (What You See When Running)

### Training Output — MNIST CNN
```
Epoch 1/5 - loss: 0.3241 - accuracy: 0.9012 - val_accuracy: 0.9687
Epoch 2/5 - loss: 0.1123 - accuracy: 0.9654 - val_accuracy: 0.9756
Epoch 3/5 - loss: 0.0891 - accuracy: 0.9723 - val_accuracy: 0.9801
Epoch 4/5 - loss: 0.0754 - accuracy: 0.9769 - val_accuracy: 0.9834
Epoch 5/5 - loss: 0.0643 - accuracy: 0.9812 - val_accuracy: 0.9856
Your model is accurate enough!
```

### Training Output — CIFAR-10 CNN
```
Epoch 1/5 - loss: 1.7823 - accuracy: 0.3512 - val_accuracy: 0.4821
Epoch 2/5 - loss: 1.3241 - accuracy: 0.5234 - val_accuracy: 0.5612
Epoch 3/5 - loss: 1.1023 - accuracy: 0.6012 - val_accuracy: 0.6123
Epoch 4/5 - loss: 0.9823 - accuracy: 0.6423 - val_accuracy: 0.6312
Epoch 5/5 - loss: 0.8921 - accuracy: 0.6712 - val_accuracy: 0.6501
 Your model is accurate enough!
```

### Single Prediction Output — LSTM
```
Predicted label: 7
Test accuracy: 0.9812
```

---

##  Key Concepts Demonstrated

- 1.CNN architecture design — Conv2D, MaxPooling, Flatten, Dense
- 2. Dropout regularization to prevent overfitting
- 3. Data normalization — pixel values scaled from [0,255] to [0.0,1.0]
- 4. LSTM applied to image data by treating rows as sequences
- 5. Transfer learning with VGG16 pretrained on ImageNet
- 6. Image preprocessing — resize, grayscale to RGB conversion
- 7. Accuracy/loss curve visualization with Matplotlib
- 8. Model saving and loading in HDF5 format (.h5)
- 9. Multi-dataset support — MNIST, CIFAR-10, Fashion-MNIST

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| TensorFlow / Keras | Model building & training |
| NumPy | Array operations & preprocessing |
| Matplotlib | Accuracy/loss visualizations |
| VGG16 (ImageNet) | Pretrained feature extractor |

---


##  Author

**Bhushan Gavali**
Machine Learning Engineer | Generative AI & LLM Systems
📧 bhushangavali24@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/bhushangavali) | [GitHub](https://github.com/bhushangavali)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
