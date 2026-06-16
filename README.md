# Deep Learning-Based Secure Blind Image Watermarking

## Overview

This project implements a secure blind image watermarking framework that embeds and extracts watermarks without requiring the original image during recovery.

The system is designed to address three key requirements of practical watermarking systems:

* Imperceptibility: minimal visual distortion after embedding
* Robustness: reliable extraction under image manipulations and attacks
* Security: prevention of unauthorized watermark recovery

Unlike conventional watermarking approaches that rely on handcrafted features or transform-domain techniques, this work employs an end-to-end deep learning architecture with key-conditioned watermark recovery and geometric attack handling.

---

## Architecture

### Encoder

An attention-based encoder embeds watermark information into the host image while preserving visual quality.

Key components:

* Convolutional feature extraction
* Channel attention
* Spatial attention
* Residual watermark embedding

### Security Module

A SHA-256-derived key tensor is integrated into the embedding and extraction process.

Additional protection is provided through Error Correction Coding (ECC), improving watermark recovery under distortion.

### Decoder

The decoder performs blind watermark extraction without access to the original image.

To improve robustness against geometric transformations, a Spatial Transformer Network (STN) is incorporated before watermark reconstruction.

---

## Methodology

The model is trained end-to-end using attack-aware training.

During training, watermarked images are subjected to various distortions including:

* Gaussian noise
* JPEG compression
* Blur
* Resize
* Rotation
* Translation
* Cropping

This enables the decoder to learn robust watermark recovery under realistic conditions.

---

## Evaluation

### Image Quality

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)

### Watermark Recovery

* NCC (Normalized Cross Correlation)
* BER (Bit Error Rate)

### Security Validation

Watermark extraction was evaluated using both valid and invalid keys.

| Scenario      | Outcome                            |
| ------------- | ---------------------------------- |
| Correct Key   | Successful recovery                |
| Incorrect Key | Extraction failure / random output |

---

## Key Contributions

* Attention-based blind watermark embedding
* Key-conditioned watermark extraction
* SHA-256-based security mechanism
* Error Correction Coding integration
* Spatial Transformer Network for geometric robustness
* Attack-aware training strategy
* Interactive Streamlit deployment

---

## Results

The framework achieves:

* High visual fidelity of watermarked images
* Reliable watermark recovery under common signal-processing attacks
* Improved robustness against geometric distortions
* Secure extraction through key-based authentication

---

## Running the Project

```bash
pip install -r requirements.txt

python train.py

python test.py

streamlit run app.py
```

---

## Research Context

This work builds upon recent advances in deep learning-based watermarking and addresses limitations in existing methods related to geometric robustness, security, and practical deployment.
