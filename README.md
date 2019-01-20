# Advanced Lane Finding Project and Yolo Objection Detection.   
## Tabel of Content
1 - [Algorithm Details](#algorithm-details).  
2 - [YOLO Object Detection](#yolo-object-detection).  
3 - [Video Output](#video-output).  
4 - [Dependencies and Compiling](#dependencies-and-compiling).  
5 - [Tensorflow Static Build](#tensorflow-static-build).   
## Algorithm Details.     
### Distortion corrected calibration image.   
The code for this step is contained in the calibration.cpp [Here](../LaneandYolovehicle-Detection/calibration.cpp).  
Start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here we are assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Then use the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the calibrateCamera() function.  
Example:
<p align="center">
  <img src="https://github.com/apreddyy/LaneandYolovehicle-Detection/blob/master/images/image1.png">
</p>  


#### Pipeline.   
At First,  resize the image and then  convert frame as Bird view and then use a combination of color and gradient thresholds to generate a binary image.  
**Step 1:** Undisort Image.   
**Step 2:** Binary Image.   
**Step 3:** Take a histogram along all the columns in the lower half of the image and split histogram for two sides for each lane.   
**Step 4:** Use the two highest peaks from histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image to determine where the lane lines go.   
Example:
<p align="center">
  <img width="640" height="640" src="https://github.com/apreddyy/LaneandYolovehicle-Detection/blob/master/images/image2.png">
</p>  


**Step 5:**  Identify lane-line pixels.  
Find all non zero pixels.  
Example:   
<p align="center">
  <img width="640" height="360" src="https://github.com/apreddyy/LaneandYolovehicle-Detection/blob/master/images/image3.png">
</p>


**Step 6:** Fit their positions with a polynomial.   
After performing 2nd order polynomial fit for nonzero pixels, drawing polyline and unwrap image the final output.  
Example:   
<p align="center">
  <img width="640" height="360" src="https://github.com/apreddyy/LaneandYolovehicle-Detection/blob/master/images/image4.png">
</p>


### Radius of curvature of the lane and the position of the vehicle with respect to center.
Get the left and right cordinates and calculate the midpoint of lanes and use the image center as reference to calculate distance away from center.  
1-	LANEDETECTION::center_dist – Distance of Vechicle from center.  
2-	LANEDETECTION::left_curverad – Left Lane Curvature.  
3-	LANEDETECTION::right_curverad – Right Lane Curvature.  
## YOLO Object Detection.  
The Tensorflow Model is trained for five classes of objects: cars, pedestrians, truck, cyclists and traffic lights.  
Example:   
<p align="center">
  <img width="640" height="360" src="https://github.com/apreddyy/LaneandYolovehicle-Detection/blob/master/images/image5.png">
</p>


## Video Output.  
The Video output can be found [Here]( https://github.com/apreddyy/LaneandYolovehicle-Detection/blob/master/out.avi).  
## Dependencies and Compiling.
### Environment Linux (Ubuntu 16.04 LTS).  
1-	CUDA 9.0. For Linux installation Guide and Requirements [Here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).     
2-	Tensorflow 1.5.0. For Building a static Tensorflow C++ library on Linux [Section](#tensorflow-static-build).    
3-	OpenCV 3.4 or Greater. More information can be found [Here](http://leadtosilverlining.blogspot.com/2018/09/build-opencv-341-with-cuda-90-support.html).    
4-	Tensorflow trained model (graph-vehicledetection.pb) is included in repository.    
## Tensorflow Static Build.
### Prerequisite.
sudo apt-get install openjdk-8-jdk  
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python  
wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh  
./bazel-0.5.4-installer-linux-x86_64.sh.  
### Clone Tensorflow.  
wget https://github.com/tensorflow/tensorflow/archive/v1.5.0.tar.gz  
tar -zxf v1.5.0.tar.gz
### Build Tensorflow  
 cd tensorflow-1.5.0 && ./configure  
 bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 --config=monolithic //tensorflow:libtensorflow_cc.so.  
 ./tensorflow/contrib/makefile/download_dependencies.sh.  
### Compiling.
 Run ./buildlane.sh.

# For Windows Version [Here](https://github.com/apreddyy/LaneandYolovehicle-Detection)
# Thanks
