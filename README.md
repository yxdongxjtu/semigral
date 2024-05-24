# SemiGraL: Semi-Supervised Graph Contrastive Learning with Virtual Adversarial Augmentation

This repository contains the source code for the paper *["Semi-Supervised Graph Contrastive Learning with Virtual Adversarial Augmentation"](https://www.computer.org/csdl/journal/tk/5555/01/10438042/1UyVbVTbrm8)*.

## Introduction
SemiGraL is a novel approach to semi-supervised graph deep learning, enhanced with virtual adversarial augmentation techniques. This repository provides the code to reproduce the experiments described in the paper.

## Requirements
To set up the environment and dependencies for running the code, please follow the instructions below. It is recommended to use `conda` for managing the Python environment.

### Environment
- **Python** (tested on 3.8): We recommend using the conda package manager.
  ```sh
  conda create -n semigral python=3.8
  conda activate semigral
  ```

### Dependencies
- **PyTorch** (tested on 1.8.2+cu111): Install PyTorch with CUDA support (CPU version is also supported). Please see the [official PyTorch website](https://pytorch.org/) for details.
  ```sh
  pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1
  ```

- **PyTorch Geometric** (tested on 2.0.1): Please refer to the [official PyTorch Geometric website](https://pytorch-geometric.readthedocs.io/) for installation details.
  > If you encounter errors while importing `torch_sparse`, please re-install it with `torch-sparse==0.6.12`.
  ```sh
  conda install pyg -c pyg
  ```

Other dependencies are listed in `requirements.txt` and can be installed using:
```sh
pip install -r requirements.txt
```

## Datasets
The datasets used in this project will be automatically downloaded and processed on the first run. These datasets are publicly available and do not have licensing restrictions.

## Running the Code
To reproduce the experimental results on the Cora dataset (tested on a single NVIDIA GeForce RTX 3090 GPU (24GB)), execute the following command:
```sh
python main_cora.py
```
For more details on the setup and reproducibility on other datasets, please refer to our paper.

## Citing SemiGraL
If you find this work helpful, please consider citing our paper:
```bibtex
@article{dong2024semi,
  title={Semi-Supervised Graph Contrastive Learning with Virtual Adversarial Augmentation},
  author={Dong, Yixiang and Luo, Minnan and Li, Jundong and Liu, Ziqi and Zheng, Qinghua},
  journal={IEEE Transactions on Knowledge \& Data Engineering},
  number={01},
  pages={1--12},
  year={2024},
  publisher={IEEE Computer Society}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
