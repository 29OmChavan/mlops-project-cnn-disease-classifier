# Role Simulation: ML Researcher — CNN Theory, Experiment Log & Model Card

**Project**: Project 02 · Deep Learning + Computer Vision (Disease Detection System)  
**Owner Role**: ML Researcher  
**MLflow Server**: [`https://dagshub.com/29omchavan/mlops-project-2-cnn-disease-classifier.mlflow`](https://dagshub.com/29omchavan/mlops-project-2-cnn-disease-classifier.mlflow)

---

## 1. First-Principles Mathematical Foundations (Sentdex & Andrew Ng CNN Lectures)

### 1.1 Discrete 2D Cross-Correlation (Convolutional Layer)
Given an input feature map $I \in \mathbb{R}^{C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$ and a filter bank $K \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_H \times k_W}$ with stride $S$ and zero-padding $P$, the output spatial dimensions are:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - k_H}{S} \right\rfloor + 1, \qquad W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2P - k_W}{S} \right\rfloor + 1$$

### 1.2 Batch Normalization (`nn.BatchNorm2d`)
Normalizes spatial activations across each mini-batch $\mathcal{B}$ per channel $c$:

$$\hat{x}_c = \frac{x_c - \mu_{\mathcal{B}, c}}{\sqrt{\sigma^2_{\mathcal{B}, c} + \epsilon}}, \qquad y_c = \gamma_c \hat{x}_c + \beta_c$$

This smooths the loss landscape, accelerates convergence, and acts as an implicit regularizer.

### 1.3 Residual Learning & Skip Connections (`FirstPrinciplesResidualBlock`)
In deep CNNs, stacking plain layers causes vanishing/exploding gradients and degradation. He et al. (ResNet) reformulate a block to learn a residual mapping $\mathcal{F}(x)$ added to the identity shortcut $x$:

$$\mathbf{y} = \sigma\big(\mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}\big)$$

During backpropagation, the gradient flows unimpeded through the identity path:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left(1 + \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}}\right)$$

---

## 2. Transfer Learning & Layer Freezing Experiments

We compared 3 transfer learning configurations on our 4-class pathology dataset:

| Experiment | Backbone Freezing Policy | Augmentation | Val Accuracy | Macro F1 | ROC-AUC (OvR) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1: Linear Probe (`FREEZE_ALL=True`)** | Freeze all ResNet-18 conv blocks; train only classifier head | Yes | ~91.2% | ~0.91 | ~0.985 | Fast training, lower capacity for domain-specific textures |
| **Exp 2: Partial Fine-Tuning (`FREEZE_TILL=4`) [Champion]** | Freeze stem + `layer1` (`idx < 4`); fine-tune `layer2–4` + head | Yes | **100.0%** | **1.000** | **1.000** | Preserves low-level Gabor/edge filters while adapting higher-level pathology morphology |
| **Exp 3: No Augmentation (`AUGMENTATION=False`)** | Partial Fine-Tuning (`FREEZE_TILL=4`) | No | ~95.0% | ~0.95 | ~0.991 | Overfits faster on training orientation |

---

## 3. Novel Extension: Dynamic INT8 Model Quantization Benchmark
Using `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`, we quantized the classification head linear layers from 32-bit floating point (`FP32`) to 8-bit integers (`INT8`), logging both latency and size metrics to `scores.json` and MLflow.
