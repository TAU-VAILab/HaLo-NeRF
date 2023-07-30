# 😇  HaLo-NeRF: Text-Driven Neural 3D Localization in the Wild

(TBA) **project page | paper**

*Chen Dudai¹\*, 
[Morris Alper](https://morrisalp.github.io/)¹\*, 
Hana Bezalel¹, 
[Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)², 
[Itai Lang](https://scholar.google.com/citations?user=q0bBhtsAAAAJ)²,
[Hadar Averbuch-Elor](https://www.elor.sites.tau.ac.il/)¹*

*¹[Tel Aviv University](https://english.tau.ac.il/),
²[The University of Chicago](https://www.uchicago.edu/)&nbsp;&nbsp;&nbsp;&nbsp;\*Equal contribution*


This repository is the official implementation of [HaLo-NeRF](https://github.com/TAU-VAILab/HaLo-NeRF/) (Text-Driven Neural 3D Localization in the Wild).

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
(TBA)
```

# Acknowledgements
This implementation is based on the official repository of [Ha-NeRF](https://rover-xingyu.github.io/Ha-NeRF/).
