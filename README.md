# HNN-HM

Hypergraph Neural Networks for Hypergraph Matching

```
@article{Liao2021HNNHM,
  title={Hypergraph Neural Networks for Hypergraph Matching},
  author={Xiaowei Liao and Yong Xu and Haibin Ling},
  journal={International Conference on Computer Vision (ICCV)},
  year={2021},
  pages={1246-1255}
}
```

## Download data

```sh
make data
```

## Environment

### Docker

My environment
```
OS: Ubuntu 16.04
Docker version: 19.03.4
GPU: TITAN V (12 GB)
Nvidia Driver version: 450.80.02
```

Build docker image
```sh
make -C envs/docker env
```

Run in docker container
```sh
make -C envs/docker run
```

### Others

Please check `envs`.

## Compile

Run in docker container
```sh
make compile
```

## Test (using trained model)

Test willow (run in docker container)
```sh
make  -C scripts evaluate-willow-out-0
```

Test all (run in docker container)
```sh
make  -C scripts all
```

Please check `scripts/Makefile` for more detail.


## Train

Run in docker container
```sh
make  -C methods/HNN_HM/scripts/synthetic train
make  -C methods/HNN_HM/scripts/house train
make  -C methods/HNN_HM/scripts/willow train
make  -C methods/HNN_HM/scripts/pascal_voc train
```
