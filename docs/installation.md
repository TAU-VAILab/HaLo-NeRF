# Requirements and Installation

## Hardware

Tested on:
* OS: Ubuntu 20.04
* NVIDIA GPU with CUDA=11.4 (tested with 1 RTXA5000)

## Installation Instructions

Clone this repo with `git clone https://github.com/TAU-VAILab/HaLo-NeRF`.

Requirements are slightly different for the different stages of training, so create seperate environments for each:

### Concept Distillation and Semantic Adaptation
* Run in a Python>=3.8 environment. Recommended: create and use conda environment via `conda create -n HaLo-NeRF-2D python=3.8` and `conda activate HaLo-NeRF-2D`
* Install core requirements with `pip install -r requirements-2d.txt`

### 3D Localization
* Run in a Python>=3.8 environment. Recommended: create and use conda environment via `conda create -n HaLo-NeRF python=3.8` and `conda activate HaLo-NeRF`
* Install core requirements with `pip install -r requirements.txt`
* Install the following torch packages using the command:
  
      `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`