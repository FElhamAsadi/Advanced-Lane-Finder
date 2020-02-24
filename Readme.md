## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Step 0: Defining AdvancedLaneFinder Class
In order to have a consie code, first I created a class which includes all methods/techniques required for detecting the lane lines in an image.  


### Step 1: Camera Calibration and Distortion Correction
The first thing which should be done before image processing, is calibration of the camera. Using calibrated images (chessboard), I prepared "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. Here I assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.`imgpoints` are also appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Then `objpoints` and `imgpoints` are used in computeation of the camera matrix and distortion coefficients. These coefficients were employed to undistort all raw images. 
The code for this step is can be found in `AdvancedLaneFinder:: camera_calibration()` and `AdvancedLaneFinder::distortion_correction` in [Advanced_Lane_Finder.py].


---
### Step 2: Creating a Threshold Binary Image
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are in `AdvancedLaneFinder::CombinedThresholds` (./advanced_lane_finding.py) which subsequently calls `AdvancedLaneFinder::GradientThresh`, `AdvancedLaneFinder::MagThresh`, `AdvancedLaneFinder::gradDirectionThresh`, and `AdvancedLaneFinder::ColorThresh`. In `AdvancedLaneFinder::ColorThresh` I used HLS and HSV color spaces together in order to distinguish white and yellow pixels in the image. 
Here's an example of my output for this step.


### Step 3: Selecting the Region of Interest
Since the lane lines apper in the bottom half and the center of the image, a region of interest has been defined using a triangle or tripozide (depending on the user preference). This region mask is applied on the threshold binary image form the previous step (`AdvancedLaneFinder::region_select` in advanced_lane_finding.py). 


### Step 4: Performing the Perspective Transform
I order to transform the perspective of an image to birds-eye view, at first source and destination point in normal view and top view of an example image were specified as follow: 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 532, 496      | 284, 464      | 
| 756, 496      | 1016, 464     |
| 288, 664      | 288, 664      |
| 1016, 664     | 1016, 664     |

The perspective matrix and its inverse were calculated using cv2.getPerspectiveTransform() and these source and destination points. The `AdvancedLaneFinder::warp()` function takes as inputs an image (`img`) and returns the birds-eye perspective of that image. 

### Step 5: Identifying lane-line pixels and fitting a polynomial on each line
Using the prominent peaks in the histogram of where the binary activations occur across the image, I foun the starting point of lane lines. 
Then, I have used a sliding window, placed around the line centers in order to find and follow the lines up to the top of the frame `AdvancedLaneFinder::sliding_window_fit`. Having found the lane line pixels, I fitted a polynomial on eavh line. 
Since using the sliding window for each frame is inefficient, in `AdvancedLaneFinder::search_aroun_poly` I just searched in a margin around the previous lane line position. This function is appliyed on each frame first and if it wasnot sucessful in finding the line polynomials for couple of steps, the function `AdvancedLaneFinder::sliding_window_fit` is employed again. Functions `AdvancedLaneFinder::fit_polynomial` and `AdvancedLaneFinder::update_lane_line` perform the sanity check and updating the lines information. 


### Step 6: Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.
Curvature radius and bias fron center were calculated in `AdvancedLaneFinder::measure_curvature_real` and `AdvancedLaneFinder::bias`. 


### An example image of the result plotted back down onto the road 
I implemented this step in `AdvancedLaneFinder:: warp_display` and `AdvancedLaneFinder:: Lane_display`. The text displaying radius of curvature and the vehicle position was added to the image in the last step of pipeline. Here is an All examples of my result on athe test images can be found in "./ouput_images" 

---

### Pipeline (video)
My ouput video can be found in "output_images/output_video5.mp4"

---

### Discussion
The current pipline with tunned parameters performs reasonably good on the project video. However, when I implemeted this on the challenge video, I found that the pipeline will likely fail in the situations when the road has cracks coming alongside the lane lines or there are significant road marks on the lane.  Also, it did not produce a good result when there is a sharp bend in lane lines. I think region masking should be adaptable to prevent loosing useful information. Besides that, smoothing and failure recovery should be modified to make `AdvancedLaneFinder` work with videos capturing worse environment conditions. 
