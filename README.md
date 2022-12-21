# Uncertainty-guided mutual consistency learning for semi-supervised medical image segmentation


This is the repository of our paper '[Uncertainty-guided mutual consistency learning for semi-supervised medical image segmentation](https://www.sciencedirect.com/science/article/pii/S0933365722002287)' (AIIM 2022), which is developed for our previous works [DTML](https://link.springer.com/chapter/10.1007/978-3-030-88010-1_46) (PRCV 2021).



### Introduction

* We Incorporate both intra-task consistency (learning from up-to-date predictions  for self-ensembling) and cross-task consistency (learning from task-level regularization to exploit geometric shape information) with the guidance of estimated segmentation uncertainty to utilize unlabeled data for semi-supervised learning. 

* This repository is our implementation on BraTS dataset.

* Our pre-trained models can be found at [here](https://github.com/YichiZhang98/UG-MCL/tree/main/model).

* More details can be found in [our paper](https://www.sciencedirect.com/science/article/pii/S0933365722002287).




### Usage

1. Clone the repo
```
git clone https://github.com/YichiZhang98/UG-MCL
cd UG-MCL
```
2. Put the data in data/BraTS2019.

3. Train the model
```
cd code
python train_UGMCL_3D.py
```

4. Test the model
```
python test_3D_dt.py
```




## Acknowledgement

* This code and experimental setting is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and other implementations including  [UA-MT](https://github.com/yulequan/UA-MT),  [DTC](https://github.com/HiLab-git/DTC) and [DTML](https://github.com/YichiZhang98/DTML). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

* More semi-supervised approaches for medical image segmentation have been summarized in our [survey](https://arxiv.org/abs/2207.14191).

* If our project is useful for your research, please consider citing the following works:



```
@article{zhang2022uncertainty,
  title={Uncertainty-guided mutual consistency learning for semi-supervised medical image segmentation},
  author={Zhang, Yichi and Jiao, Rushi and Liao, Qingcheng and Li, Dongyang and Zhang, Jicong},
  journal={Artificial Intelligence in Medicine},
  pages={102476},
  year={2022},
  publisher={Elsevier}
}

@inproceedings{zhang2021dual,
  title={Dual-task mutual learning for semi-supervised medical image segmentation},
  author={Zhang, Yichi and Zhang, Jicong},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={548--559},
  year={2021},
  organization={Springer}
}

@article{jiao2022learning,
  title={Learning with limited annotations: a survey on deep semi-supervised learning for medical image segmentation},
  author={Jiao, Rushi and Zhang, Yichi and Ding, Le and Cai, Rong and Zhang, Jicong},
  journal={arXiv preprint arXiv:2207.14191},
  year={2022}
}
```
