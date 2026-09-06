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

## Getting Started

1. Open `index.html` in **Chrome** or **Firefox** (WebGL required)
2. Click **Train Model** on first use (~30–60s)
3. Draw a digit and click **Predict**

> The model is saved to IndexedDB automatically after training. On subsequent visits it loads instantly — no retraining needed.

## Training

- **Dataset:** MNIST (60,000 train / 10,000 test), fetched from Google's CDN
- **Epochs:** 10
- **Batch size:** 256
- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Categorical Cross-Entropy
- **Typical accuracy:** ~97–99%

## Files
   ├──index.html
   └── style.css
   └── script.js
   └── README.md

No build step, no dependencies to install — just open the file.

## External Resources (CDN)

| Library | Version | Purpose |
|---------|---------|---------|
| `@tensorflow/tfjs` | 4.17.0 | Model training & inference |

MNIST data is fetched from:
https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png
https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8

## Browser Support

| Browser | Status |
|---------|--------|
| Chrome | ✅ Recommended |
| Firefox | ✅ |
| Edge | ✅ |
| Brave | ✅ |
| Safari | ⚠️ May have WebGL issues |

## Notes

- Requires internet connection on first load (CDN + MNIST data)
- IndexedDB storage is per-origin — clearing browser data removes the model
- The drawing canvas is 280×280 px, internally scaled to 28×28 for prediction   
