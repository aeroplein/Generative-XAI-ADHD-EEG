# Modeling Differences Between Adult and Pediatric ADHD Using Generative XAI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

This repository contains the official implementation of the study **"Differences In Adult ADHD and Children ADHD Using Generative XAI"**.

We introduce a novel **Transfer Learning-driven Denoising Diffusion Probabilistic Model (D4PM)** framework to address data scarcity in pediatric ADHD research. By bridging the quantitative "Stationarity Gap" between adult and pediatric EEG signals, we generate biologically plausible synthetic data for augmentation and analysis.

##  Key Features

* **Stationarity Gap Analysis:** Quantifies the **11.0% statistical gap** between adult (dynamic) and pediatric (rigid) EEG domains using Augmented Dickey-Fuller tests.
* **EEG Diffusion Transformer:** A specialized Transformer architecture utilizing Sinusoidal Time Embeddings and Patch Embeddings for continuous time-series generation.
* **Transfer Learning Strategy:** Pre-trains on a large-scale healthy adult dataset (PhysioNet) and fine-tunes on a pediatric ADHD dataset to overcome data scarcity.
* **Generative XAI:** Visualizes the "Evolution of Thought" (reverse diffusion) and validates biological plausibility via Power Spectral Density (PSD) analysis.

##  Project Structure

```bash
├── models/
│   ├── embeddings.py
│   ├── transformer.py
│   └── d4pm.py
├── utils/
├── preprocessing/
├── train.py
├── finetune.py
├── generative_xai.py
├── test.py
└── analyze_gap.ipynb

```

##  Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



##  Usage Pipeline

### 1. Pre-training (Source Domain)

Trains the model on the Adult dataset to learn the fundamental "grammar" of EEG signals.

```bash
python train.py

```

### 2. Fine-tuning (Target Domain)

Transfers the learned weights to the Pediatric ADHD dataset, adapting to the specific spectral rigidity of the disorder.

```bash
python finetune.py

```

### 3. Generative XAI & Validation

Generates synthetic signals, visualizes the diffusion process, and performs PSD comparison for biological validation.

```bash
python generative_xai.py

```

##  Results

* **Stationarity Gap:** Confirmed a significant domain shift: Adults (51.70% stationary) vs. Pediatric ADHD (62.70% stationary).
* **Reconstruction Fidelity:** Achieved a validation MSE of **1.32** on the target pediatric domain.
* **Biological Validity:** The generated synthetic signals successfully replicate the **Theta-band (4-8 Hz) power anomalies** characteristic of pediatric ADHD.

##  Contributors

* **Pelin Zeynep Kaya** - Lead Developer & Model Architecture
* **Arjin İlhan** - Problem Definition & Statistical Analysis
* **Mustafa Baver Çalış** - Experimental Validation & XAI

##  Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kaya2025adhd,
  title={Differences In Adult ADHD and Children ADHD Using Generative XAI},
  author={Kaya, Pelin Zeynep and İlhan, Arjin and Çalış, Mustafa Baver},
  booktitle={IEEE Conference},
  year={2025}
}

```

---

*This project is built using PyTorch and MNE-Python.*

```

```
