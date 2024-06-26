# My_PoseNet
This is the implementation of [PoseNet](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf) using InceptionV3 as the original implementation. We used KingsCollege dataset to train and test our implementeation. In addition this code is in PyTorch using the last version and ROS Noetic to estimate the pose frame-by-frame.

## TODO
- [x] Implementation of PoseNet in PyTorch
- [x] Features Extractor using our PoseNet implementation
- [x] ROS Evaluation using our PoseNet implementation

## Prerequisites
- Ubuntu 20.04
- Python 3.8.10
- Pytorch 2.3.0+cu121
- CUDA 12.2
- CUDNN 8.9.7

## Dataset
The dataset loader is based on dataset of original PoseNet. Please download on of the dataset in [PoseNet Dataset](http://mi.eng.cam.ac.uk/projects/relocalisation/). You can use other datasets your own. Please dowload the KingsCollege Dataset from PoseNet using:

```
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip?sequence=4&isAllowed=y
```

## Training
- Train a model
  ```
  python train.py
  ```

- Trained models are saved in ./models_{Folder name of image path}_. If you want use other parameter for the training, please change them into the script.

## Test
- Test a model
  ```
  python test.py
  ```
  
- Tested the models saved in ./models_{Folder name of image path}_. Please change path model and image path into the script.

## PoseNet as Feature Extractor (Transfer Learning)
- Extractor of features with PoseNet architecture
  ```
  python extractor.py
  ```

- If you want use the features to train another network or to evaluate the images with other methods use this script. Features are saved in ./features_{Folder name of image path}_.


## Evaluation using ROS Noetic (Frame-by-Frame)
- Test a model using a flow of images by ROS node
  ```
  python ros-test.py
  ```
  
- Tested the models saved in ./models_{Folder name of image path}_ using ROS. Please define a ros topic where there is a flow of images. In this script you should pass through a ROS node the images to evaluate the model frame-by-frame.

