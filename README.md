# CSED399B
2024 Summer Research Participation @ POSETCH MLLab

## Based Repositories
[![1](https://img.shields.io/static/v1?label=lxz1217&message=weather4cast-2023-lxz&color=181717)](https://github.com/lxz1217/weather4cast-2023-lxz)
[![2](https://img.shields.io/static/v1?label=jhhuang96&message=ConvLSTM-PyTorch&color=181717)](https://github.com/jhhuang96/ConvLSTM-PyTorch)

## Roadmap
### Papers
* [Precipitation Prediction Using an Ensemble of Lightweight Learners (2023)](https://arxiv.org/abs/2401.09424)
* [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting (2015)](https://arxiv.org/abs/1506.04214)
* [Skilful precipitation nowcasting using deep generative models of radar (2021)](https://www.nature.com/articles/s41586-021-03854-z)
* [Skilful nowcasting of extreme precipitation with NowcastNet (2023)](https://www.nature.com/articles/s41586-023-06184-4)

### 


## Comments
* weather4cast-2023-lxz
  * dataset can be downloaded [here](https://weather4cast.net/get-the-data/) (or [here](https://cds.climate.copernicus.eu/#!/home))
  * train_stage1.py: submodule UNet of model MoE is causing errors
    sat2rad() should return a Tensor but returns tuple
* w4c-ConvLSTM
  * error on train.py line 120: dimension issues
  * sample visualized data point from w4c dataset:
    ![](w4c-ConvLSTM/d.png)
    Time 1, 8, 9 (and even 10) contains potential erronous images
