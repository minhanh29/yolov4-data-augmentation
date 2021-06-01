# Yolov4 Data Augmentation Implementation

*Author*: **Minh Anh Nguyen**  

## Introduction

This repo contains the source code that implements some Yolov4 Data Augmentation techniques using Python3 and OpenCV.  

#### Implemented techniques:
- [x] Mosaic
- [x] Mix Up
- [x] Cut Mix
- [x] Grid Mask

#### Requirements
- [x] Python3
- [x] OpenCV
- [x] Numpy


## Quick Start

1. Open the config.py file to specify the following.
	* DATA_DIR = directory storing the images
	* ANNO_DIR = directory storing the annotation file
	* TARGET_DIR = directory storing the augmented images
	* TARGET_ANNO = directory storing the  annotation file for the augmented images
	* TOTAL_SAMPLES = total images to generate (each technique will run on 1/4 of the total samples)
	* IMAGE_SIZE = lenght of the square output image

#### Data annotation file format:
	* One row for one image in annotation file;
	* Row format: `image_file_path box1 box2 ... boxN`;
	* Box format: `x_min,y_min,x_max,y_max,class_id` (no space).
	* Here is an example:
	```
	path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
	path/to/img2.jpg 120,300,250,600,2
	...
	```
2. Run the main.py file in Python3 to generate all the augmented images:

```
python3 main.py
```

**Note**: you can import the functions from the file augmentation_techniques.py to perform each task separately. Please read the comments in each function to know the expected inputs and outputs.  

### TODO

Open the yolov4-data-augmentation-demo.ipynb file to see the demo usage.  
