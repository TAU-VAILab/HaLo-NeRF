# HaLo-NeRF: Text-Driven Neural 3D Localization in the Wild

(TBA) **project page | paper**

*Chen Dudai¹, 
[Morris Alper](https://morrisalp.github.io/)¹, 
Hana Bezalel¹, 
[Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)², 
[Itai Lang](https://scholar.google.com/citations?user=q0bBhtsAAAAJ)²,
[Hadar Averbuch-Elor](https://www.elor.sites.tau.ac.il/)¹*

*¹[Tel Aviv University](https://english.tau.ac.il/),
²[The University of Chicago](https://www.uchicago.edu/)*


This repository is the official implementation of [HaLo-NeRF](https://github.com/TAU-VAILab/HaLo-NeRF/) (Text-Driven Neural 3D Localization in the Wild).

# Requirements and Installation

See the linked [installation instructions](docs/installation.md).

# Data Downloads

You may download the relevant scenes and pretrained models at the links below.

Download links:

* Scene images and COLMAP reconstructions: (Link TBA)
* GT semantic masks: (Link TBA)
* Segmentation model: (Link TBA)
* Scene semantic segmentation data: (Link TBA)
* Retrieval data: (Link TBA)
* RGB NeRF models: (Link TBA)
* Semantic NeRF models: (Link TBA)


Note that we use six different scenes:

* Three cathedrals - [Milan Cathedral](https://en.wikipedia.org/wiki/Milan_Cathedral), [St Paul's Cathedral](https://en.wikipedia.org/wiki/St_Paul%27s_Cathedral), [Notre-Dame](https://en.wikipedia.org/wiki/Notre-Dame_de_Paris)

* Two mosques - [Badshahi Mosque](https://en.wikipedia.org/wiki/Badshahi_Mosque), [Blue Mosque](https://en.wikipedia.org/wiki/Blue_Mosque,_Istanbul)

* One synagogue - [Hurva Synagogue](https://en.wikipedia.org/wiki/Hurva_Synagogue) 

# Training

HaLo-NeRF is trained in three stages; instructions on training these from scratch can be found at their respective documentation pages:

* [LLM-Based Concept Distillation](docs/concept_distillation.md)
* [Semantic Adaptation](docs/semantic_adaptation.md)
* [3D Localization](docs/3d_localization.md)

Altenratively, you may use the pretrained results, downloadable from the links given above.


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
