# SurgicalCubePose
This is the main repository for tracking surgical attachment for accurate monocular pose estimation with aruco boards to provide navigation support in an intraoperative environment.

## Dependencies

* Python 3
* NumPy
* Pandas
* Scipy
* OpenCV
* Sklearn
* Pickle
* circle_fit

## Structure of the repository
(`main.py`) is the principal program where from the images the pose of the tip of the instrument is estimated for each frame, (`calibration_tip.py`) is where the instrument tip is calibrated using two different methods, one of these least squares and some utility functions (`auxiliar_functions.py`). Finally in (`visualization.py`) the models of the experiments carried out are fitted.

## Usage example
With this example, from tracing the tip of the surgical instrument on top of a porcelain sphere with a radius of 7.5 mm hidden in a fluffy body, the following sphere model is obtained, shown in the gif.

<p align="center">
    <img src="figures/tracing.gif" alt="tracing the sphere" width="300px"/>
</p>

To run this example:

* Clone this repository: `git clone https://github.com/camicontr/SurgicalCubePose.git`
* You need to download the `sphere/` folder which contains the data of the position of the instrument tip in the (x, y, z) axes in mm for three sphere fit experiments in addition to the intrinsic parameters of the camera [here](https://drive.google.com/drive/folders/1uhU6IE9X8NizRNrs7ECeyxzukqAC4v5q?usp=sharing).
* Run `visualization.py` to read the data and fit the sphere, you can choose the example between 1, 2 or 3.

A visualization of the point cloud in addition to the sphere model obtained 

<p align="center">
    <img src="figures/Sphere.jpeg" alt="Result of fit sphere" width="500px"/>
</p>
With a radius rms error of 0.1701 mm 
