# Click to Move
Click to Move: Controlling Video Generation with Sparse Motion

Pytorch implementation of our paper [Click to Move: Controlling Video Generation with Sparse Motion](https://arxiv.org/abs/2108.08815)
In [ICCV 2021](https://iccv2021.thecvf.com/home).
Please cite with the following Bibtex code:
```
@inproceedings{ardino2021click,
  title={Click to Move: Controlling Video Generation with Sparse Motion},
  author={Ardino, Pierfrancesco and De Nadai, Marco and Lepri, Bruno and Ricci, Elisa and Lathuili{\`e}re, St{\'e}phane},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14749--14758},
  year={2021}
}
```

Please follow the instructions to run the code.

# Scripts

## 1. Installation

 - See the [`c2m.yml`](./c2m.yml) configuration file. We provide an user-friendly configuring method via Conda system, and you can create a new Conda environment using the command:

```
conda env create -f c2m.yml
conda activate c2m
```

 - Install `cityscapesscripts` with `pip`
```
cd cityscapesScripts
pip install -e .

```

WIP

## 2. Data Preprocessing
### 2.1 Generate instance segmentation
  We apply a modified version of [Panoptic-deeplab](https://github.com/bowenc0221/panoptic-deeplab/) to get the corresponding semantic and instance maps. You can find it into ```panoptic_deeplab``` folder. For this work we have used the ```HRNet``` backbone. You can download it from [here](https://drive.google.com/drive/folders/1bJLyZkKsharpGykxjR7hmb6yzp8nmxMj?usp=sharing). 
#### Cityscapes
* Please download the Cityscapes dataset from the [official website](https://www.cityscapes-dataset.com/) (registration required). After downloading, please put these files under the ```~/dataset_cityscape_video/``` folder and run the following command in order to generate the correct segmentation maps
  ```
  cd panoptic_deeplab
  python tools/generate_segmentation.py --cfg configs/cityscapes_{trainset/valset}.yaml TEST.MODEL_FILE YOUR_DOWNLOAD_MODEL_FILE
  ```
  Remember to set up the config file with the correct input folder, output folder and dataset split

  You should end up with the following structure:
  ```
  dataset_cityscape_video
  ├── leftImg8bit_sequence
  │   ├── train
  │   │   ├── aachen
  │   │   │   ├── aachen_000003_000019_leftImg8bit.png
  │   │   │   ├── ...
  │   ├── val
  │   │   ├── frankfurt
  │   │   │   ├── frankfurt_000000_000294_leftImg8bit.png
  │   │   │   ├── ...
  │   ├── train_semantic_segmask
  │   │   ├── aachen
  │   │   │   ├── aachen_000003_000019_ssmask.png
  │   │   │   ├── ...
  │   ├── val_semantic_segmask
  │   │   ├── frankfurt
  │   │   │   ├── frankfurt_000000_000294_ssmask.png
  │   │   │   ├── ...
  │   ├── train_instance
  │   │   ├── aachen
  │   │   │   ├── aachen_000003_000019_gtFine_instanceIds.png
  │   │   │   ├── ...
  │   ├── val_instance
  │   │   ├── frankfurt
  │   │   │   ├── frankfurt_000000_000294_gtFine_instanceIds.png
  │   │   │   ├── ...
  ```

### 2.2 Generate object trajectories


### 3 Train the model
  We store the configuration of the model as a ```YAML``` configuration file. You can have a look at a base configuration in ```src/config/c2m_journal_cityscapes.yaml```.
  The training file takes as input the following parameters:
  - ```config```: path to configuration file
  - ```device_ids```: names of the devices comma separated
  - ```seed```: seed of the training
  - ```profile```: debug using PyTorch profiler

  Our code support multi-gpu training using ```DistributedDataParallel```. Here's an example of how you can run the code with one or more gpus.
#### Single gpu
  ```python train.py --device_ids 0 --config config/c2m_journal_cityscapes.yaml```
#### Multi gpu
 ```python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 train.py --device_ids 0,1 --config config/c2m_journal_cityscapes.yaml```
  The example considers a scenario with a single node and two gpus per node. Please change according to your needs. For more information check the [DDP example](https://github.com/pytorch/examples/tree/master/distributed/ddp)
### 4 Test the model
  ```python test.py --device_ids 0 --config config/c2m_journal_cityscapes.yaml```