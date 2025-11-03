# MD-Net: Underwater Image Enhancement via Multiscale Disentanglement Strategy

This repository contains a PyTorch implementation of the paper ["Underwater image enhancement via multiscale disentanglement strategy"](https://www.nature.com/articles/s41598-025-89109-7), published in Nature Scientific Reports.

## Overview

MD-Net is a novel deep learning framework for underwater image enhancement that employs a multiscale disentanglement strategy to effectively separate and process content and style features at different scales. This approach enables superior enhancement results by addressing the complex degradation patterns found in underwater environments.

## Installation

### Clone the Repository
```bash
git clone https://github.com/BassamXYZ/md-net-underwater-image-enhancement.git
cd md-net-underwater-image-enhancement
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
md-net-underwater-image-enhancement/
├── model/
│   ├── custom_blocks/
│   │   ├── __init__.py
│   │   ├── mfm_branch.py
│   │   ├── mfm.py
│   │   ├── cw_layer.py
│   │   ├── pw_layer.py
│   │   ├── dweu.py
│   │   ├── adversarial_net_block.py
│   │   └── gbl.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── lsui.py               # LSUI Dataset Module
│   │   ├── uieb.py               # UIEB Dataset Module
│   │   └── uieb_challenging.py   # UIEB Challenging Dataset Module
│   ├── __init__.py
│   ├── neural_net.py             # Main MD-Net Module
│   ├── total_loss.py             # Custom Loss Module
│   ├── trainer.py                # Module For Network Training
│   └── util.py                   # Utility functions
├── mdnet-uieb.ipynb              # Jupiter file to train and test model on UIEB dataset
├── mdnet-lsui.ipynb              # Jupiter file to train and test model on LSUI dataset
├── datasets-installation.ipynb   # Jupiter file for database installation
├── evaluate.py                   # PSNR, SSIM, UCIQE, UIQM Tests
├── nevaluate.py                  # UCIQE, UIQM Tests
├── README.md
├── requirements.txt
└── .gitignore
```

## Citation

If you use this implementation in your research, please cite the original paper and this repository:

```bibtex
@article{original_paper,
  title = {Underwater image enhancement via multiscale disentanglement strategy},
  journal = {Scientific Reports},
  volume = {14},
  pages = {12345},
  year = {2025},
  publisher = {Nature},
  doi = {10.1038/s41598-025-89109-7},
  url = {https://www.nature.com/articles/s41598-025-89109-7}
}

@software{mdnet_implementation,
  title = {MD-Net: PyTorch Implementation for Underwater Image Enhancement},
  author = {Bassam Ahmad},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/BassamXYZ/md-net-underwater-image-enhancement}},
}
```

## Acknowledgments

- The original authors of the MD-Net paper
- [xueleichen](https://github.com/xueleichen) for the PSNR, SSIM, UCIQE, and UIQM implementation
- The open-source community for various utility functions and components

## License

This project is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use.

For commercial use or modifications, please contact the repository maintainer.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve this implementation.

## Contact

For questions or suggestions regarding this implementation, please open an issue or contact me at [contact@bassamahmad.com].
