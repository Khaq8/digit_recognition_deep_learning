# Handwritten Digit Recognition

A browser-based handwritten digit recognition app built with **TensorFlow.js**. Draw a digit on a canvas, and the model predicts which number (0–9) you drew.

## Features

- ✏️ Draw digits on a 28×28 canvas (matching MNIST dimensions)
- 🧠 Train a neural network directly in the browser on MNIST data
- 💾 Model auto-saved to IndexedDB — persists across page reloads
- 📊 Real-time loss & accuracy graphs during training
- 📋 Epoch-by-epoch training log
- 🎯 Confidence bar showing all 10 digit probabilities

## Model Architecture

| Layer | Type | Units | Activation |
|-------|------|-------|------------|
| 1 | Flatten | 784 | — |
| 2 | Dense | 256 | ReLU |
| 3 | Dropout | 0.3 | — |
| 4 | Dense | 128 | ReLU |
| 5 | Dense | 10 | Softmax |

## 🚀 Getting Started

1. ```text
   git clone https://github.com/Khaq8/digit_recognition_deep_learning.git
    ```
2. Open `index.html` in **Chrome** or **Firefox** (WebGL required)
3. Click **Train Model** on first use (~30–60s)
4. Draw a digit and click **Predict**

> The model is saved to IndexedDB automatically after training. On subsequent visits it loads instantly — no retraining needed.

## 🎯 Training

- **Dataset:** MNIST (60,000 train / 10,000 test), fetched from Google's CDN
- **Epochs:** 10
- **Batch size:** 256
- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Categorical Cross-Entropy
- **Typical accuracy:** ~97–99%

## 📂 Project Structure
```text
   ├──index.html
   └── style.css
   └── script.js
   └── README.md
```

No build step, no dependencies to install — just open the file. 
