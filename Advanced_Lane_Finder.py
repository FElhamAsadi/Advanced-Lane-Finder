#Importing the required packages
import numpy as np
import cv2
import os
import logging
import math
import matplotlib.pyplot as plt


class AdvancedLaneFinder: 
    '''This defines a class implementing lane line finding advanced techniques'''
    
    def __init__(self,
                image_size = (720, 1280),
                chessboard_image_dir = 'camera_cal',
                gradient  =(7, 30,100),
                magnitude =(7, 0,255),
                direction = (15, 0, np.pi/2),
                s_channel_thresh= (150,255),
                h_channel_thresh =(15, 100),
                region_interest=np.array([[(0, 720),(640, 405),(640, 405),(1280, 720)]], dtype=np.int32),
                source_points = np.float32([[200, 736], [1120, 736], [700, 458], [590, 458] ]), 
                dest_points = np.float32([[315, 736], [968, 736], [968, 0 ], [315, 0]]), 
                ym_per_pix = 30/720, 
                xm_per_pix = 3.7/700,
                sliding_window_param = (9, 50, 80),
                n_max_failure = 10):
        
                
        # ******* initialize directory with chessboard images ********
        if os.path.isdir(chessboard_image_dir):
            self.chessboard_image_dir = os.path.abspath(chessboard_image_dir)
            logging.info("Directory with calibration images: %s.", self.chessboard_image_dir)
        else:
            raise AdvancedLaneFinderError("%s directory does not exist." % chessboard_image_dir)

        # initialize list of calibration chessboard images
        self.chessboard_image_path_list = [os.path.join(self.chessboard_image_dir, fname)
                                            for fname in os.listdir(self.chessboard_image_dir)]
          
        if not self.chessboard_image_path_list:
            raise AdvancedLaneFinderError("No calibration images found in %s." % self.chessboard_image_dir)
        else:
            logging.info("There are %d calibration images.", len(self.chessboard_image_path_list))

        
        #*************** Initialization ********************************  
        self.image_size = image_size
        self.chessboard_image_dir = chessboard_image_dir  # path to directory containig chessboard calibration images
        
        self.grad_thresh = gradient[1:3]   # Gradient thresholds
        self.sobel_kernel = gradient[0]    # Kernel size in sobel calculation of gradient 
        
        self.mag_thresh = magnitude[1:3]   # Thresholds of magnitude of the gradient 
        self.mag_kernel = magnitude[0]     # Kernel size in sobel calculation - magnitude of the gradient calculation 
        
        self.dir_thresh = direction[1:3]   #Threshold of direction of gradient
        self.dir_kernel = direction[0]    # Kernel size in sobel calculation - direction of gradient calculation 
        
        self.s_thresh = s_channel_thresh   # S channel threshold - HLS color space
        self.h_thresh = h_channel_thresh   # H channel threshold - HLS color space
        
        self.region = region_interest      # Coordinates of vertices - Region of interset 
        self.src = source_points           # Source points for perspective transform
        self.dst = dest_points             # Destination points in region of transform
        
        self.ym_per_pix = ym_per_pix       # meters per pixel in y dimension
        self.xm_per_pix = xm_per_pix       # meters per pixel in x dimension
        
        self.calibration_matrix = None          # Camera matrix
        self.distortion_coefficients = None     # Distortion matrix
        self.rotation_vectors = None            # Rotation vector
        self.translation_vectors = None          # Translation vector
        self.calibrated = False
        
        
        self.mtx_perspect = cv2.getPerspectiveTransform(self.src, self.dst)        # The perspective transorm matrix 
        self.inv_mtx_perspect = cv2.getPerspectiveTransform(self.dst, self.src)    #Inverse of perspective transform matrix
        
        self.nwindows = sliding_window_param[0]  # The number of sliding windows
        self.margin = sliding_window_param[1]    # The width of the windows +/- margin
        self.minpix = sliding_window_param[2]    # The minimum number of pixels found to recenter window
       
    
        # lane lines tracking
        self.left_line = Line()
        self.right_line = Line()
        self.ploty = np.int32(np.linspace(0, self.image_size[0]-1, self.image_size[0]))  # linear space along Y axis

        # lane detection failure counter
        self.lane_detection_failure_count = 0
        self.max_lane_detection_failures_before_sliding_window = n_max_failure

        self.distance = 0    # Distance of the vehicle from the lane centerline
        self.direction = None
    
        self.radius_mean_curvature = 0
        
    
    def  camera_calibration(self):
        '''Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        Return tuple of calibration matrix and distortion coefficients.
        '''
        
        #Preparing the object points for 9*6 chessboard (0,0,0), (1,0,0), ..., (8,5,0) 
        objp = np.zeros((6*9, 3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        #Array to store object points and image points from all images 
        objpoints = []
        imgpoints = []

        # Step through the list and search for chessboard corners
        for fname in self.chessboard_image_path_list:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            #find the chessbord corners
            ret , corners = cv2.findChessboardCorners(gray, (9,6), None)
    
            #if corners are found, add them to imgpoints
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
        
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if not ret:
            raise AdvancedLaneFinderError("Camera calibration has failed.")
        else:
            # initialize corresponding class instance fields
            self.calibration_matrix = mtx
            self.distortion_coefficients = dist
            self.rotation_vectors = rvecs
            self.translation_vectors = tvecs
            self.calibrated = True

    def get_calibration_camera_output(self) :
        '''Getter for the tuple of calibration matrix, distortion coefficients, rotation and translation vectors.
        output format : -> Tuple[np.ndarray, np.ndarray, list, list]'''
        
        return (self.calibration_matrix,
                self.distortion_coefficients,
                self.rotation_vectors,
                self.translation_vectors)
    
    
    def distortion_correction(self, image) :
        '''Apply distortion correction to the input image.'''
        
        return cv2.undistort(image, self.calibration_matrix, self.distortion_coefficients, None, self.calibration_matrix)
   
    
    def GradientThresh(self, gray):
     
        #******  Gradient Threshold *******
        # Sobel x
        # Take the derivative in x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize= self.sobel_kernel) 
       
    # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx) 
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sx_binary = np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel >= self.grad_thresh[0]) & (scaled_sobel <= self.grad_thresh[1])] = 1
    
        return sx_binary
   

    def MagThresh(self, gray):    

        #******** Magnitude of the Gradient *******
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = self.mag_kernel) 
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = self.mag_kernel)
 
        # Calculate the magnitude 
        abs_sobel =  np.sqrt(sobelx**2 + sobely**2)

        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= self.mag_thresh[0]) & (scaled_sobel <= self.mag_thresh[1])] = 1
    
        return mag_binary

    def gradDirectionThresh(self, gray):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = self.dir_kernel) 
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = self.dir_kernel)

        # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        grad_dir = np.arctan2(np.abs(sobely), np.abs(sobelx))

        # Return a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(grad_dir)
        dir_binary [(grad_dir >= self.dir_thresh[0]) & (grad_dir <= self.dir_thresh[1])] = 1
 
        return  dir_binary


    def ColorThresh(self, img):
        # Convert to HLS color space and separate the S channel
        hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls_image[:,:,2]
        l_channel = hls_image[:,:,1]
           
        # Convert to the HSV color space    
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Selecting colors yellow and white
        white_color = cv2.inRange(hls_image, np.uint8([10,200,0]), np.uint8([255,255,255]))
        yellow_color = cv2.inRange(hsv_image, np.uint8([15,60,130]), np.uint8([150,255,255]))
    
        # Combine yellow and white masks
        combined_color_images = cv2.bitwise_or(white_color, yellow_color)
    
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        # Threshold color channel
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= 100) & (l_channel <= self.s_thresh[1])] = 1

        # Combined binaries
        c_binary = np.zeros_like(s_channel)
        c_binary[((s_binary > 0) & (l_binary > 0)) | (combined_color_images > 0)] = 1
        
          
        return c_binary
 

    def CombinedThresholds(self, img):
    
        # img: undistorted image
 
        # Grayscale image
        gray_0 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray_0, (15, 15), 0)

        #******  Gradient Threshold *******
        sx_binary = self.GradientThresh(gray)
    
    
        #******** Magnitude of the Gradient *******
        mag_binary = self.MagThresh(gray)
    

        #************ Gradient Direction Threshold *********
        dir_binary = self.gradDirectionThresh(gray)
 

        #************* Color Thresholds **************
        c_binary = self.ColorThresh(img)
    
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sx_binary)
        combined_binary[((c_binary == 1) | (sx_binary == 1)) | ((mag_binary==1) & (dir_binary==1))] = 1
    
        return combined_binary
       
    
    def region_select(self, img):
    
        ''' Applies an image mask. It only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        '''
     
        ysize = img.shape[0]
        xsize = img.shape[1]
        XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

        if  len(self.region[0]) == 3:
            left_bottom = self.region[0][0]
            right_bottom = self.region[0][1]
            apex = self.region[0][2]
            
            #Fit lines (y = Ax+B) to identify the three sided region of interest for lane finding
            fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
            fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
            fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
    
            #find the region inside the lines
            region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                     (YY > (XX*fit_right[0] + fit_right[1])) & (YY < (XX*fit_bottom[0] +fit_bottom[1])) 
            
        elif len(self.region[0]) == 4:
            left_bottom = self.region[0][0]
            left_up = self.region[0][1]
            right_up = self.region[0][2]    
            right_bottom = self.region[0][3]

            #Fit lines (y = Ax+B) to identify the three sided region of interest for lane finding
            fit_left = np.polyfit((left_bottom[0], left_up[0]), (left_bottom[1], left_up[1]), 1)
            fit_right = np.polyfit((right_bottom[0], right_up[0]), (right_bottom[1], right_up[1]), 1)
            fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
            fit_up = np.polyfit((left_up[0], right_up[0]), (left_up[1], right_up[1]), 1)

            #find the region inside the lines
            region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                     (YY > (XX*fit_right[0] + fit_right[1])) & \
                     (YY < (XX*fit_bottom[0] +fit_bottom[1])) & \
                     (YY > (XX*fit_up[0] +fit_up[1]))


        img[~region_thresholds] = 0

        return img


    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.mtx_perspect, img_size, flags=cv2.INTER_LINEAR)
    
        return warped
    
    
    
    def sliding_window_fit(self, binary_warped, draw):
        '''Detect lane pixels and fit to find the lane boundary using sliding windows technique.
        Args:
            binary_warped: warped gray scale image
        '''
    
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
    
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
        
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
                   
            # Draw the windows on the visualization image
            if draw =='Yes' : 
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2) 
    
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
        
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) >self. minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit
        # Fit new polynomials
        if (lefty.size != 0) & (leftx.size !=0 ): 
            left_fit = np.polyfit(lefty, leftx, 2)
            # Colors in the left and right lane regions
            out_img[lefty, leftx] = [255, 0, 0]
            
        if (rightx.size !=0 ) & (righty.size !=0 ):
            right_fit = np.polyfit(righty, rightx, 2)
            out_img[righty, rightx] = [0, 0, 255]


        # update tracked metrics of lane lines
        self.update_lane_lines(left_fit = left_fit, right_fit = right_fit)
         
        return out_img
                

           
    
    def search_around_poly(self, binary_warped, draw):
        ''' Detect lane pixels and fit to find the lane boundary based on knowledge where the current lane lines are.
        Args:
            binary_warped: warped gray scale image
        '''
   
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit
    
    
        #Set the area of search based on activated x-values within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - self.margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + self.margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - self.margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + self.margin)))
    
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit
        # Fit new polynomials
        if (lefty.size != 0) & (leftx.size !=0 ): 
            left_fit = np.polyfit(lefty, leftx, 2)
        if (rightx.size !=0 ) & (righty.size !=0 ):
            right_fit = np.polyfit(righty, rightx, 2)
     
        self.update_lane_lines(left_fit = left_fit, right_fit = right_fit)
        
        ## Visualization ##
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        result = out_img

        if draw == 'Yes': 
            # Create an image to draw on and an image to show the selection window
            window_img = np.zeros_like(out_img)

            # Generate a polygon to illustrate the search window area and recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([self.left_line.allx - self.margin, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_line.allx + self.margin, 
                                  self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([self.right_line.allx - self.margin, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_line.allx + self.margin, 
                              self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

           # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
                     
            result = cv2.addWeighted(out_img, 1, window_img, 0.4, 0)

        return result
    
  

    
    def fit_polynomial(self, img,  draw) :
        '''Detect lane pixels and fit to find the lane boundary.
           img : warped gray scale image    
        '''
        if self.left_line.detected and self.right_line.detected \
        and (self.lane_detection_failure_count < self.max_lane_detection_failures_before_sliding_window):
            return self.search_around_poly(img, draw)
        else:
            self.lane_detection_failure_count = 0
            self.left_line.detected = False
            self.right_line.detected = False
            return self.sliding_window_fit(img, draw)

    
    def update_lane_line(self, line, fit, allx):
        line.detected = True
        line.current_fit = fit
        line.ally = self.ploty
        line.allx = allx

    def update_lane_lines(self, left_fit, right_fit):
        new_left_line_allx = left_fit[0]*self.ploty**2 + left_fit[1]*self.ploty + left_fit[2]
        new_right_line_allx = right_fit[0]*self.ploty**2 + right_fit[1]*self.ploty + right_fit[2]

        # check that detection was correct
        if self.left_line.detected \
           and math.fabs(new_left_line_allx[self.image_size[0]-1] - self.left_line.allx[self.image_size[0]-1]) \
               > 5 \
           and self.right_line.detected \
           and math.fabs(new_right_line_allx[self.image_size[0]-1] - self.right_line.allx[self.image_size[0]-1]) \
               > 5:
            self.lane_detection_failure_count += 1
            return

        self.update_lane_line(line=self.left_line, fit=left_fit, allx=new_left_line_allx)
        self.update_lane_line(line=self.right_line, fit=right_fit, allx=new_right_line_allx)

                
    def measure_curvature_real(self):
        '''Calculate curvature of left and right lane lines'''
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)
    
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit

        # Convert the coeficient to meter
        left = [(self.xm_per_pix/(self.ym_per_pix**2))* left_fit[0], (self.xm_per_pix/self.ym_per_pix) * left_fit[1], left_fit[2] ]
        right= [(self.xm_per_pix/(self.ym_per_pix**2))* right_fit[0], (self.xm_per_pix/self.ym_per_pix) * right_fit[1], right_fit[2] ]

        # Calculation of R_curve (radius of curvature)
        self.left_line.radius_of_curvature  = ((1 + (2*left[0]*y_eval*self.ym_per_pix + left[1])**2)**1.5) / np.absolute(2*left[0])
        self.left_line.radius_of_curvature  = ((1 + (2*right[0]*y_eval*self.ym_per_pix + right[1])**2)**1.5) / np.absolute(2*right[0])
        self.radius_mean_curvature = (self.left_line.radius_of_curvature + self.left_line.radius_of_curvature)/2
  
        
    def bias(self): 
        '''Calculate bias from center of the road'''
        mid = self.image_size[1]/2
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit
        
        Xl = left_fit[0]*self.image_size[0]**2+ left_fit[1]*self.image_size[0] + left_fit[2]
        Xr = right_fit[0]*self.image_size[0]**2+ right_fit[1]*self.image_size[0] + right_fit[2]
    
        self.distance = (mid - (Xl+Xr)/2)* self.xm_per_pix
        if self.distance <0:
            self.direction = 'right of center'
        elif self.distance>0:
            self.direction = 'left of center'
        else:
            self.direction = 'at center'
        

    def warp_display(self, binary_warped) :
        ''' Output visual display of the lane boundaries '''
        
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Grab activated pixels
        nonzero = binary_warped
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        left_fit = self.left_line.current_fit
        right_fit = self.right_line.current_fit
        
        #Set the area of search based on activated x-values within the +/- margin of our polynomial function ###
        between_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] ))) & \
                        ((nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2])))
    
        
        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area and recast the x and y points into usable format for cv2.fillPoly()
        left_line_window = np.array([np.transpose(np.vstack([self.left_line.allx, self.ploty]))])
        right_line_window = np.array([np.flipud(np.transpose(np.vstack([self.right_line.allx, self.ploty])))])
        line_pts = np.hstack((left_line_window, right_line_window))
   
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        ## End visualization steps ##
    
        return result

    def Lane_display(self, img_orig, warped_lane):
        img_size = (warped_lane.shape[1], warped_lane.shape[0])
        img_lane = cv2.warpPerspective(warped_lane, self.inv_mtx_perspect, img_size, flags=cv2.INTER_LINEAR)
        result = cv2.addWeighted(img_orig,1,img_lane ,1,0)
   
        return result

    def addText(self, img):
        cv2.putText(img, "Radius of Curvature= %8.2f m" % self.radius_mean_curvature , \
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,1.5,color=(255, 255, 255), thickness=3)
        cv2.putText(img, "Position:%8.2f m from the centerline" % self.distance, \
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX,1.5,color=(255, 255, 255), thickness=3)
        return img

    
    
class Line:
    '''Class to receive the characteristics of each line detection.'''
    def __init__(self ) :
        
        # was the line detected in the last iteration?
        self.detected = False

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

        