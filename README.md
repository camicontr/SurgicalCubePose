# SurgicalCubePose
This is the main repository for tracking surgical attachment for accurate monocular pose estimation with aruco boards to provide navigation support.


## Usage example
With this example, for tracking the tip of the surgical instrument on top of a porcelain sphere with a radius of 7.5 mm, then a sphere fit is obtained, shown in the following gif.

<p align="center">
    <img src="figures/tracing.gif" alt="tracing the sphere" width="300px"/>
</p>

To run this example:

* Clone this repository: `git clone https://github.com/camicontr/SurgicalCubePose.git`
* You need to download the `sphere/` folder which contains the data of the position of the instrument tip in the (x, y, z) axes in mm for three sphere fit experiments in addition to the intrinsic parameters of the camera and the relative vector of the calibration of the tip  [here](https://drive.google.com/drive/folders/1E_RvevjmIQaYquqwyUEOO_2DNnOohgSj?usp=sharing).
* Run `Recontruction.py` to read the data and get the results of the sphere fit. You can change the example between 0, 1 or 2.

A visualization of the point cloud in addition to the sphere model obtained 

<p align="center">
    <img src="figures/Sphere.png" alt="Result of fit sphere" width="500px"/>
</p>
With a radius rms error of 0.1701 mm 

## Dependencies

* Python 3
* NumPy
* Pandas
* Scipy
* OpenCV
* Sklearn
* circle_fit
* Scikit-surgery-sphere-fitting
