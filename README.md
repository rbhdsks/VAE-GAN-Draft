# VAE-GAN-Draft

## A Novel Adaptation of the UNIT Framework for Cross-Domain Climate Data Translation with Enhanced Feature Preservation

Published in the **12th International Conference on Signal Processing and Integrated Networks (SPIN 2025)** — Springer, 2025.

This repository contains an adaptation of the UNIT (Unsupervised Image-to-Image Translation) framework, designed to translate climate data across domains while preserving salient geospatial and multi-variable features. The work was presented at SPIN 2025 and is published via Springer.

**Citation:**
> N. K. Shah et al. (2025). "A Novel Adaptation of the UNIT Framework for Cross-Domain Climate Data Translation with Enhanced Feature Preservation." *12th International Conference on Signal Processing and Integrated Networks (SPIN 2025)*. Springer.

**Paper:** [Link to publication](https://link.springer.com/chapter/10.1007/978-981-96-9967-4_16)

---

## Abstract

This paper introduces a novel adaptation of the Unsupervised Image-to-Image Translation (UNIT) framework, specifically designed to address the challenges of translating complex climate data across domains. Traditional UNIT models often fail to handle climate datasets due to their intricate geospatial constraints and multi-variable nature.

To overcome this limitation, the proposed approach:
- Integrates **land masks** to maintain geospatial specificity, accurately representing geographical boundaries such as coastlines.
- Introduces a **flexible loss calculation strategy** combining multiple reconstruction loss functions — Frechet Inception Distance (FID), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS).
- Provides a mechanism for handling **multi-variable datasets** by selectively applying different loss functions to individual data channels, balancing the importance of each variable.

Experimental results show that the adapted UNIT framework significantly outperforms traditional methods, achieving superior geospatial feature preservation and more precise climate data translation — providing a robust tool for environmental analysis and more reliable climate modelling and prediction.

---

## Repository Structure

```
.
├── configs/        # Configuration files for training/evaluation
├── scripts/        # Helper scripts
├── data.py         # Dataset loading and preprocessing
├── networks.py     # Model architectures (encoders, decoders, discriminators)
├── ssim.py         # SSIM loss implementation
├── train.py        # Training entry point
├── trainer.py      # Training loop logic
├── translate.py    # Inference / domain translation script
├── utils.py        # Utility functions
└── README.md
```

## Authors

Nitesh Kumar Shah, Abhishek Bidhan, Bhavini Mathur, Ujair Alam, Vyom Kumar Gupta, Surya Prakash
