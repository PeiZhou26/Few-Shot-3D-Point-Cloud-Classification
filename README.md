# Few-Shot 3D Point Cloud Classification

This repo contains the source code for the ECE 228 course project: Few-Shot 3D Point Cloud Classification. In this project, we extend Few-Shot method from 2D domain to 3D domain.

## Enviroment
 - Python3
 - Pytorch
 - json
 - h5py
 - tensorboard

## Getting started
### Dataset download and split
* Change directory to `../dataset/`. ('dataset' folder shoule be the same level as the project folder. If there is no this folder, make the directory first)
* Download dataset: download the resampled dataset [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and unzip the ModelNet40 in `../dataset/modelnet40_normal_resampled`.
* Split dataset: You can split the dataset by yourself, or you can use our split by copying './data/base.json' and './data/novel.json' to `../dataset/modelnet40`.

(WARNING: If you split the dataset by yourself, please keep the same format as ours)

### Data preprocessing
* Before you run experiments, you should preprocess the data at the first time to increase the speed of training.
* Run `python ./data/dataset.py`, it may take more than two hour to finish the preprocessing for the whole dataset, since farthest point sample method is time-consuming.

(WARNING: In different systems, the path may be different. If it display file is not found, please modify the 'pwd' variable of the main function in './data/dataset.py')


## Train your model now!
Run
```python ./train.py --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```
For example, run `python ./train.py --model pointnet --method protonet --weightdecay 0.01`  
If you want to implement different methods with different hyperparameter, and please refer to 'io_utils.py'.


## Save features
After training, save feature first! it can speed the repeated experiments in testing. 
For example, run ```python ./save_features.py --model pointnet --method protonet```. Please refer to 'io_utils.py' for more details.
You can find your stored feature file in './features'.

## Test your model
For example, run ```python ./test.py --model pointnet --method protonet```. Please refer to 'io_utils.py' for more details.

(Reminder: You can run the test command directly without running 'train.py' and 'save_features.py', since we have uploaded a trained model for PointNet+ProtoNet)

You can check your experment results in `./record/results.txt`


## References
* We thank [A Closer Look at Few-shot Classification](https://github.com/wyharveychen/CloserLookFewShot). The framework of meta-learning is modified based on this paper.


