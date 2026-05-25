# Adversarial Multimodal AI Framework for Synthetic Profile Detection

An advanced multimodal AI framework designed to detect **AI-generated, manipulated, and synthetic online profiles** using a combination of:

- Computer Vision
- Transformer-based NLP
- Multimodal Learning
- Deep Learning
- Robustness Fine-Tuning

The system analyzes both **profile images** and **profile bios** and combines outputs through a multimodal decision engine to estimate synthetic profile risk.

---

## Project Overview

Online platforms increasingly face challenges from:

- AI-generated profile photos
- Fake identities
- Synthetic personas
- Bot-like profile behavior
- Deepfake-assisted impersonation

This project addresses the problem using a **multimodal detection pipeline** consisting of:

### Image Analysis Module

Detects suspicious or AI-generated profile pictures using:

- EfficientNet
- Transfer Learning
- Computer Vision

### Text Analysis Module

Analyzes profile bios using:

- DistilBERT Transformer
- NLP classification
- Suspicious text pattern detection

### Multimodal Fusion Engine

Combines image and text predictions to produce:

- Final synthetic profile verdict
- Risk score estimation

---

## Features

- Fake profile image detection
- Transformer-based bio analysis
- Multimodal fusion engine
- Risk score generation
- Streamlit deployment
- Transfer learning pipeline
- Robustness fine-tuning experiments

---

## Tech Stack

- PyTorch
- Transformers
- Streamlit
- OpenCV
- scikit-learn
- Google Colab
- Hugging Face Transformers

---

## Model Architecture

### Image Model

- EfficientNet-B0
- Transfer Learning
- Binary Classification

Classes:

- Fake
- Real

### Text Model

- DistilBERT
- Sequence Classification

Classes:

- Suspicious
- Real

### Fusion Logic

The framework combines:

- Image prediction
- Image confidence
- Text prediction
- Text confidence

to generate:

- Final Verdict
- Risk Score

---

## Dataset

### Image Dataset

Base Dataset:

Real vs AI Face Dataset

Contains approximately:

- 5000 Real Human Faces
- 4630 AI-Generated Human Faces

Dataset Access:

**Google Drive Link:**

[Dataset Link]

### Additional Robustness Dataset

To improve robustness against modern photorealistic AI portraits:

- 30 additional photorealistic AI-generated portraits were added to the fake class.

These samples were used for targeted hard-negative robustness fine-tuning.

---

## Robustness Evaluation Experiment

### Baseline Observation

During testing, the baseline model struggled with highly realistic modern AI portraits.

Example:

Input:

Gemini-generated photorealistic human image

Prediction:

**REAL**

---

### Robustness Fine-Tuning

A targeted experiment was performed.

Changes introduced:

- Added 30 photorealistic AI portrait samples.
- Stronger data augmentation.

Augmentations used:

- Random Rotation
- Horizontal Flip
- Color Jitter
- Random Affine Transformation
- Gaussian Blur

---

### Post-Training Findings

Improvements:

- Better detection of photorealistic synthetic portraits.

Observed Tradeoff:

Some edited or filtered real images were occasionally predicted as:

**FAKE (high confidence)**

This revealed an important robustness challenge:

Improving synthetic detection may increase false positives on visually enhanced real images.

---

## Project Structure

```text
Adversarial-Multimodal-AI-Framework-for-Synthetic-Profile-Detection/

app/
│
├── app.py
├── image_model.py
├── text_model.py
├── fusion_engine.py

models/
│
├── fake_profile_detector.pth
├── distilbert_text_model.pth

notebooks/
│
├── dataset_setup.ipynb
├── train_text_model.ipynb

datasets/

requirements.txt
README.md
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/divyanshrawat7/Adversarial-Multimodal-AI-Framework-for-Synthetic-Profile-Detection.git
```

Go into project:

```bash
cd Adversarial-Multimodal-AI-Framework-for-Synthetic-Profile-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Download Models

Download trained model files and place them inside:

```text
models/
```

Required files:

- fake_profile_detector.pth
- distilbert_text_model.pth

Model Download Link:

[INSERT MODEL LINK HERE]

---

## Running The Application

Move into app folder:

```bash
cd app
```

Run Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## How To Use

1. Upload a profile image.
2. Enter profile bio text.
3. Click **Analyze Profile**.
4. View:

- Image Prediction
- Text Prediction
- Final Verdict
- Risk Score

---

## Future Improvements

Planned enhancements:

- Explainable AI visualizations
- Behavioral feature integration
- Improved robustness against edited real photos
- Adversarial defense strategies
- Larger synthetic identity benchmark dataset
- Hugging Face deployment

---

## Author

**Divyansh Rawat**

AI / ML • Deep Learning • Multimodal Systems • Cyber Security
