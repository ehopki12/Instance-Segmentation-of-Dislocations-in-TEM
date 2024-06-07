# 2023_IEEE_Instance-Segmentation-of-Dislocations-in-TEM

This repo implements a physical-based metric to evaluate Machine learning models for segmenting dislocations in TEM images of dislocation microstructure. For more details, please refer to our paper [Instance Segmentation of Dislocations in TEM Images](https://ieeexplore.ieee.org/document/10231169). 

## A Quick Overview 
We use the Yolov8 model from [Ultralytics](https://github.com/ultralytics/ultralytics) to perform instance segmentation on Real TEM data, which is hand labelled using [RoboloFlow](https://roboflow.com/). We add our [physical-based evaluation metric](./ultralytics/ultralytics/yolo/v8/segment/loss.py) to the Yolov8 model.  

<div align="center">
  <img width="330" height="500" src="imgs/Result.png">
  <br>
  <b>Instance segmentation of the TEM images of dislocation microstructure using Yolov8</b>
</div>


## Requirement
Since the metric is integrated in the Yolo framework, you just need to clone this repository using 

``git clone https://gitlab.com/computational-materials-science/public/publication-data-and-code/2023_IEEE_Instance-Segmentation-of-Dislocations-in-TEM.git``

Create a Python environment and install Ultralytics package locally using 

```bash
pip install -r requirements.txt

cd ultralytics 

pip install -e . 

```

## Usage 
### Prediction on new TEM images  
A Jupyter notebook [Predict](./Jupyternotebook/Predict.ipynb) can be  used to start making predictions using checkpoint provided by our training. To make the predictions, one can directly install the YOLO using 
```bash
pip install ultralytics
```  

## Cite
K. Ruzaeva, K. Govind, M. Legros and S. Sandfeld, "Instance Segmentation of Dislocations in TEM Images," 2023 IEEE 23rd International Conference on Nanotechnology (NANO), Jeju City, Korea, Republic of, 2023, pp. 1-6, doi: 10.1109/NANO58406.2023.10231169.
