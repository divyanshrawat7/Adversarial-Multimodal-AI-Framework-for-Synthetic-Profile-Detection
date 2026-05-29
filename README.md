# Adversarial Multimodal AI Framework for Synthetic Profile Detection

An advanced multimodal AI framework designed to detect **AI-generated, manipulated, and synthetic online profiles** using a combination of:

* Computer Vision
* Transformer-based NLP
* Multimodal Learning
* Deep Learning
* Robustness Fine-Tuning

The framework analyzes **profile images** and **profile bios** and combines outputs through a multimodal decision engine to estimate synthetic profile risk.

---

# Live Demo

### Try the deployed application directly

**Live Website:**
https://synthetic-profile-detection.streamlit.app/

No setup required.

Simply:

1. Open the website.
2. Upload a profile image.
3. Enter a profile bio.
4. Click **Analyze Profile**.

The system will generate:

* Image Prediction
* Text Prediction
* Final Verdict
* Risk Score

---

# Project Overview

Online platforms increasingly face challenges from:

* AI-generated profile photos
* Fake identities
* Synthetic personas
* Bot-assisted accounts
* Deepfake-driven impersonation

This project addresses the problem using a **multimodal synthetic profile detection pipeline**.

---

## Image Analysis Module

Detects suspicious or AI-generated profile pictures using:

* EfficientNet-B0
* Transfer Learning
* Computer Vision
* Binary Classification

Capabilities:

* Fake profile image detection
* Confidence scoring
* Robustness-oriented fine-tuning

---

## Text Analysis Module

Analyzes profile bios using:

* DistilBERT Transformer
* NLP classification
* Semantic text understanding
* Suspicious text pattern detection

Capabilities:

* Bio authenticity analysis
* Text confidence prediction
* Suspicious vs Real classification

---

## Multimodal Fusion Engine

Combines:

* Image Prediction
* Image Confidence
* Text Prediction
* Text Confidence

to generate:

* Final Synthetic Profile Verdict
* Risk Score Estimation
* Combined Profile Assessment

---

# Features

* Fake profile image detection
* Transformer-based bio analysis
* Multimodal fusion engine
* Risk score generation
* Cyber-styled Streamlit dashboard
* Transfer learning pipeline
* Robustness fine-tuning experiments
* Live deployment support
* Real-time prediction workflow

---

# Tech Stack

### Machine Learning

* PyTorch
* TorchVision
* Transformers
* OpenCV
* Scikit-learn
* NumPy

### Application Layer

* Streamlit
* CSS Custom UI Styling
* Pillow

### Development & Deployment

* Google Colab
* GitHub
* Streamlit Community Cloud
* Google Drive Model Hosting

---

# Model Architecture

## Image Model

**Architecture:** EfficientNet-B0

Approach:

* Transfer Learning
* Binary Classification

Classes:

* Fake
* Real

---

## Text Model

**Architecture:** DistilBERT

Approach:

* Transformer Sequence Classification

Classes:

* Suspicious
* Real

---

## Fusion Logic

The framework combines multimodal outputs to estimate synthetic profile probability.

Inputs:

* Image Prediction
* Image Confidence
* Text Prediction
* Text Confidence

Outputs:

* Final Verdict
* Risk Score

---

# Dataset

## Image Dataset

Base Dataset:

**Real vs AI Face Dataset**

Contains approximately:

* 5000 Real Human Faces
* 4630 AI-Generated Human Faces

Dataset Access:

https://drive.google.com/file/d/1lLf4KHG8UbQGaCQJrQklathL3hZBgJa6/view?usp=sharing

---

## Additional Robustness Dataset

To improve robustness against modern photorealistic AI portraits:

* 30 additional photorealistic AI-generated portraits were added to the fake class.

Used for:

* Hard-negative fine-tuning
* Robustness experimentation

---

# Robustness Evaluation Experiment

## Baseline Observation

Initial testing showed difficulty handling highly realistic synthetic portraits.

Example:

Input:

**Gemini-generated photorealistic portrait**

Initial Prediction:

**REAL**

---

## Robustness Fine-Tuning

Improvements introduced:

* Added 30 targeted synthetic samples
* Stronger data augmentation strategy

Augmentations:

* Random Rotation
* Horizontal Flip
* Color Jitter
* Random Affine Transformation
* Gaussian Blur

---

## Post-Training Findings

Observed Improvements:

* Better synthetic portrait detection performance.

Observed Trade-off:

Some edited or filtered real images were occasionally predicted as:

**FAKE (high confidence)**

This highlighted a practical research challenge:

Improving robustness against sophisticated synthetic media may increase false positives on visually enhanced real photographs.

---

# Project Structure

```text
Adversarial-Multimodal-AI-Framework-for-Synthetic-Profile-Detection/

app/
│
├── app.py
├── image_model.py
├── text_model.py
├── fusion_engine.py
├── styles.css

models/
│
├── fake_profile_detector.pth

notebooks/
│
├── dataset_setup.ipynb
├── train_text_model.ipynb

requirements.txt
README.md
.gitignore
```

---

# Running The Project Locally

If you would like to experiment with the project locally instead of using the deployed website, follow these steps.

---

## Step 1 — Clone Repository

```bash
git clone https://github.com/divyanshrawat7/Adversarial-Multimodal-AI-Framework-for-Synthetic-Profile-Detection.git

cd Adversarial-Multimodal-AI-Framework-for-Synthetic-Profile-Detection
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Launch Application

```bash
cd app

streamlit run app.py
```

---

## Step 4 — Open Browser

Streamlit will generate a local URL similar to:

```text
http://localhost:8501
```

Open it in your browser.

---

## Optional: Reproducing Training Pipeline

If you would like to retrain the models from scratch instead of using the deployed application or pretrained inference pipeline, dataset setup is required.

### Kaggle API Setup (if using Kaggle datasets)

1. Create a Kaggle account.

2. Go to:

```text
Kaggle → Account → API → Create New API Token
```

3. Download:

```text
kaggle.json
```

4. Place the file inside:

```text
~/.kaggle/
```

or configure it inside **Google Colab**.

5. Download the required datasets.

Dataset used in this project:

**Real vs AI Face Dataset**

Dataset Access:

https://drive.google.com/file/d/1lLf4KHG8UbQGaCQJrQklathL3hZBgJa6/view?usp=sharing

Additional robustness experiment:

- 30 photorealistic AI-generated portrait samples were added for targeted robustness fine-tuning.

---

## Model Handling

To keep the repository lightweight and deployment-friendly:

* Smaller model weights are included within the repository.
* Large DistilBERT model weights are hosted externally and downloaded automatically during runtime.

No manual model download is required for normal usage.

---

# Deployment

**Live Application:**
https://synthetic-profile-detection.streamlit.app/

Deployment Stack:

* Streamlit Community Cloud
* GitHub Repository Integration
* Google Drive Model Hosting

---

# Future Improvements

Planned enhancements:

* Explainable AI visualizations
* Behavioral feature integration
* Improved robustness against edited real photos
* Adversarial defense strategies
* Larger synthetic identity benchmark datasets
* API deployment layer
* Attention heatmaps
* Hugging Face deployment

---

# Author

**Divyansh Rawat**

GitHub:
https://github.com/divyanshrawat7

---

# Repository

GitHub Project:

https://github.com/divyanshrawat7/Adversarial-Multimodal-AI-Framework-for-Synthetic-Profile-Detection
