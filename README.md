# CSED399B
2024 Summer Research Participation @ POSETCH MLLab

## Based Repositories
[![1](https://img.shields.io/static/v1?label=lxz1217&message=weather4cast-2023-lxz&color=181717)](https://github.com/lxz1217/weather4cast-2023-lxz)
<!-- [![2](https://img.shields.io/badge/TomaszGolan-hdf5_manipulator-181717)](https://github.com/TomaszGolan/hdf5_manipulator) -->

## Roadmap
### Papers
* [Precipitation Prediction Using an Ensemble of Lightweight Learners](https://arxiv.org/abs/2401.09424)
* [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)

### 


## Comments
* weather4cast-2023-lxz
  * dataset can be downloaded [here](https://weather4cast.net/get-the-data/) (or [here](https://cds.climate.copernicus.eu/#!/home))
  * train_stage1.py: submodule UNet of model MoE is causing errors
    sat2rad() should return a Tensor but returns tuple
