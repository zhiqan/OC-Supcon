# OC-Supcon
## [A supervised contrastive learning method based on online complement strategy for long-tailed fine-grained fault diagnosis](https://www.sciencedirect.com/science/article/pii/S1474034624007304)
As industrial automation and intelligence advance, equipment complexity rises, leading to diverse fault patterns. In fine-grained fault diagnosis, sample scarcity causes a significant long-tail effect, where main fault categories dominate. High intra-class variance and inter-class similarity in fine-grained categories impede the performance of traditional supervised contrastive learning, particularly for underrepresented tail categories in feature space. To address the above problems, a novel supervised contrast learning method for long-tailed fine-grained fault diagnosis, OC-SupCon, is proposed to improve the feature representations through the online complement strategy. Supervised contrastive learning is used as the model framework to ensure that each batch contains the inherent features of all fine-grained categories by introducing a class-centered prototype. Then, data augmentation is dynamically complemented by assessing the neighborhood sparsity of the samples to reduce the unfavorable influence on the features of the tail categories. Finally, the dominance of the head category is mitigated by balancing the gradient contributions of different fine-grained categories. In addition, Logit compensation technique is used in the classifier branch to adjust the category boundaries, and the class center prototypes are dynamically updated during the training process. The experimental results show that the proposed method exhibits significant performance in long-tailed fine-grained fault diagnosis tasks compared to existing state-of-the-art methods.


# If it is helpful for your research, please kindly cite this work:
﻿
```html
article{bai2023effectiveness,
  title={On the effectiveness of out-of-distribution data in self-supervised long-tail learning},
  author={Bai, Jianhong and Liu, Zuozhu and Wang, Hualiang and Hao, Jin and Feng, Yang and Chu, Huanpeng and Hu, Haoji},
  journal={arXiv preprint arXiv:2306.04934},
  year={2023}
}

@article{ZHAO2025103079,
title = {A supervised contrastive learning method based on online complement strategy for long-tailed fine-grained fault diagnosis},
journal = {Advanced Engineering Informatics},
volume = {64},
pages = {103079},
year = {2025},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2024.103079},
url = {https://www.sciencedirect.com/science/article/pii/S1474034624007304},
author = {Zhiqian Zhao and Yinghou Jiao and Yeyin Xu and Runchao Zhao}
}
```
