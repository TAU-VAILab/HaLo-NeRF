# 😇  HaLo-NeRF: Learning Geometry-Guided Semantics for Exploring Unconstrained Photo Collections

(TBA) **[project page](https://tau-vailab.github.io/HaLo-NeRF/) | paper**

[Chen Dudai](https://www.linkedin.com/in/chen-dudai-108a72136/?originalSubdomain=il)¹\*, 
[Morris Alper](https://morrisalp.github.io/)¹\*, 
[Hana Bezalel](https://www.linkedin.com/in/hanabezalel/?originalSubdomain=il)¹, 
[Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)², 
[Itai Lang](https://itailang.github.io/)²,
[Hadar Averbuch-Elor](https://www.elor.sites.tau.ac.il/)¹*

*¹[Tel Aviv University](https://english.tau.ac.il/),
²[University of Chicago](https://www.uchicago.edu/)&nbsp;&nbsp;&nbsp;&nbsp;\* Denotes equal contribution*

This repository is the official implementation of [HaLo-NeRF: Learning Geometry-Guided Semantics for Exploring Unconstrained Photo Collections](https://github.com/TAU-VAILab/HaLo-NeRF/).

# Abstract
Internet image collections containing photos captured by crowds of photographers show promise for enabling digital exploration of large-scale tourist landmarks. However, prior works focus primarily on geometric reconstruction and visualization, neglecting the key role of language in providing a semantic interface for navigation and fine-grained understanding. In more constrained 3D domains, recent methods have leveraged modern vision-and-language models as a strong prior of 2D visual semantics. While these models display an excellent understanding of broad visual semantics, they struggle with unconstrained photo collections depicting such tourist landmarks, as they lack expert knowledge of the architectural domain and fail to exploit the geometric consistency of images capturing multiple views of such scenes. In this work, we present a localization system that connects neural representations of scenes depicting large-scale landmarks with text describing a semantic region within the scene, by harnessing the power of SOTA vision-and-language models with adaptations for understanding landmark scene semantics. To bolster such models with fine-grained knowledge, we leverage large-scale Internet data containing images of similar landmarks along with weakly-related textual information. Our approach is built upon the premise that images physically grounded in space can provide a powerful supervision signal for localizing new concepts, whose semantics may be unlocked from Internet textual metadata with large language models. We use correspondences between views of scenes to bootstrap spatial understanding of these semantics, providing guidance for 3D-compatible segmentation that ultimately lifts to a volumetric scene representation. To evaluate our method, we present a new benchmark dataset containing large-scale scenes with ground-truth segmentations for multiple semantic concepts. Our results show that HaLo-NeRF can accurately localize a variety of semantic concepts related to architectural landmarks, surpassing the results of other 3D models as well as strong 2D segmentation baselines. Our code and data are publicly available at https://tau-vailab.github.io/HaLo-NeRF/

# Requirements and Installation

See the linked [installation instructions](docs/installation.md).

# Data and Downloads

Domain-specific concepts and semantic segmentation are learned from a large set of cathedral and mosque scenes (excluding the ones used for testing, described below).

We use six different scenes to evaluate HaLo-NeRF for text-guided 3D localization:

* Three cathedrals - [Milan Cathedral](https://en.wikipedia.org/wiki/Milan_Cathedral), [St Paul's Cathedral](https://en.wikipedia.org/wiki/St_Paul%27s_Cathedral), [Notre-Dame](https://en.wikipedia.org/wiki/Notre-Dame_de_Paris)

* Two mosques - [Badshahi Mosque](https://en.wikipedia.org/wiki/Badshahi_Mosque), [Blue Mosque](https://en.wikipedia.org/wiki/Blue_Mosque,_Istanbul)

* One synagogue - [Hurva Synagogue](https://en.wikipedia.org/wiki/Hurva_Synagogue) 

You may download the relevant scenes and pretrained models at links provided in the [data documentation](docs/data.md).

# Training

HaLo-NeRF is trained in multiple stages; instructions on training these from scratch can be found at their respective documentation pages:

* [Concept Distillation and Semantic Adaptation](docs/distillation_adaptation.md)
* [3D Localization](docs/3d_localization.md)

Alternatively, you may use the pretrained results, downloadable from the links provided in the [data documentation](docs/data.md).


Note: For each command, you may pass `--help` to see additional flags and configuration options.


# Evaluation

See [documentation](docs/evaluation.md) for instructions on running evaluation, given a scene and the respective pretrained models.


# Citation
If you find this project useful, you may cite us as follows:
```
@InProceedings{dudai2024halonerf,
 author = {Dudai, Chen and Alper, Morris and Bezalel, Hana and Hanocka, Rana and Lang, Itai and Averbuch-Elor, Hadar},
 title = {HaLo-NeRF: Learning Geometry-Guided Semantics for Exploring Unconstrained Photo Collections},
 booktitle = {Proceedings of the Eurographics Conference (EG)},
 year = {2024}
}
```

# Acknowledgements
This implementation is based on the official repository of [Ha-NeRF](https://rover-xingyu.github.io/Ha-NeRF/).
