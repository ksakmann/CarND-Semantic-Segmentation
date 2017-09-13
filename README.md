# Semantic Segmentation
### Introduction
In this project the pixels of a road in images are labelled using a Fully Convolutional Network (FCN).
The network uses the architecture described in [Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
and is trained on the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).

[//]: # (Image References)
[image1]: ./images/um_000032.png
[image2]: ./images/um_000093.png
[image3]: ./images/umm_000032.png
[image4]: ./images/umm_000063.png
[image5]: ./images/uu_000082.png
[video1]: ./images/project_video.mp4

Some of the results are shown below:

![sample][image1]
![sample][image2]
![sample][image3]
![sample][image4]

The code performs a hyperparameter search using 200 epochs for training each network. 
A test of the trained network on road conditions very different to the training data can be found here [project_video](./images/project_video.mp4)

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
