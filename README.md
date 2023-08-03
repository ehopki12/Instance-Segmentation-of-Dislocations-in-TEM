# 2023_IEEE_Instance-Segmentation-of-Dislocations-in-TEM

This repo is the implementation of the physical based metric to evaluate Machine learning models to segment dislocations in TEM images of dislocation microstructure. For more details please refer our paper [Instance Segmentation of Dislocations in TEM Images](URL here). 

## A Quick Overview 
We use Yolov8 model from [Ultralytics](https://github.com/ultralytics/ultralytics) to perform instance segmentation on Real TEM data which is handlabelled using [RoboloFlow](https://roboflow.com/). We add our [physical based evaluation metric](./ultralytics/ultralytics/yolo/v8/segment/loss.py) to the yolov8 model.  

<div align="center">
  <img width="330" height="500" src="imgs/Result.png">
  <br>
  <b>Instance segmentation of the TEM images of dislocation microstructure using Yolov8</b>
</div>


## Requirement
Since the metric is integerated in the yolo framework, you just need to clone this repository using 

``git clone https://gitlab.com/computational-materials-science/public/publication-data-and-code/2023_IEEE_Instance-Segmentation-of-Dislocations-in-TEM.git``

Create a python environment using 
``pip install -r requirements.txt``

Download the [checkpoint](https://drive.google.com/file/d/1ABDDwBTycn-z8JIRTqfRIQlycoHlMQNc) of the model and create a directory checkpoint and place it there . 

The dataset is provided as a zip file "datasets.zip". you may unzip this file and get the complete dataset used in our work. Some of the sample TEM images of dislocation microstructure to test the code can be found at ./sample_images. 

## Usage 
A jupyter notebook [Predict](./Jupyternotebook/Predict.ipynb) can be  used to start making predictions using checkpoint provided by our training.  


Please create an issue if you any problems, we will get back to you and fix it. 
## Cite
Please cite 
