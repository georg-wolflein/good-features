<div align="center">
<h1>A Good Feature Extractor Is All You Need</h1>
</div>

This repository contains the official code for the paper:

> [**A Good Feature Extractor Is All You Need for Weakly Supervised Pathology Slide Classification**](https://arxiv.org/abs/2311.11772)  
> Georg Wölflein, Dyke Ferber, Asier Rabasco Meneghetti, Omar S. M. El Nahhas, Daniel Truhn, Zunamys I. Carrero, David J. Harrison, Ognjen Arandjelović and Jakob N. Kather  
> _arXiv_, Nov 2023.

<details>
<summary>Read full abstract.</summary>
Stain normalisation is thought to be a crucial preprocessing step in computational pathology pipelines. We question this belief in the context of weakly supervised whole slide image classification, motivated by the emergence of powerful feature extractors trained using self-supervised learning on diverse pathology datasets. To this end, we performed the most comprehensive evaluation of publicly available pathology feature extractors to date, involving more than 8,000 training runs across nine tasks, five datasets, three downstream architectures, and various preprocessing setups. Notably, we find that omitting stain normalisation and image augmentations does not compromise downstream slide-level classification performance, while incurring substantial savings in memory and compute. Using a new evaluation metric that facilitates relative downstream performance comparison, we identify the best publicly available extractors, and show that their latent spaces are remarkably robust to variations in stain and augmentations like rotation. Contrary to previous patch-level benchmarking studies, our approach emphasises clinical relevance by focusing on slide-level biomarker prediction tasks in a weakly supervised setting with external validation cohorts. Our findings stand to streamline digital pathology workflows by minimising preprocessing needs and informing the selection of feature extractors.
</details>

## Main results

![](assets/teaser_AttentionMIL.png)

- We compare 12 feature extractors, and find [CTransPath](https://github.com/Xiyue-Wang/TransPath) and [Lunit's DINO](https://github.com/lunit-io/benchmark-ssl-pathology) model produce the best representations for downstream weakly supervised slide classification tasks.
- We show that stain normalisation and image augmentations may be omitted without compromising downstream performance.

> [!NOTE]
> *March 2024:* We updated our [preprint](https://arxiv.org/abs/2311.11772) to include two additional feature extractors: Phikon-Teacher and Lunit-MoCo.

## Overview
![](assets/overview.png)

## Citing

If you find this useful, please cite:

```bibtex
@misc{wolflein2023good,
    title   = {A Good Feature Extractor Is All You Need for Weakly Supervised Pathology Slide Classification}, 
    author  = {W\"{o}lflein, Georg and Ferber, Dyke and Meneghetti, Asier Rabasco and El Nahhas, Omar S. M. and Truhn, Daniel and Carrero, Zunamys I. and Harrison, David J. and Arandjelovi\'{c}, Ognjen and Kather, Jakob N.},
    journal = {arXiv:2311.11772},
    year    = {2023},
}
```