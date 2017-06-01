# NCAP Video Analyser
This project contains a program that analyses an NCAP crash test video and calculates the distance traveled, speed and acceleration of the car.

![Video](http://i.imgur.com/tfDPQFw.gif)

## Inhoud
* [NCAP crash test](#ncap-crash-test)
  * [Video](#video)
  * [NCAP logo](#ncap-logo)
* [Detecting NCAP logo](#detecting-ncap-logo)
  * [Region of interest](#region-of-interest)
  * [Color filtering](#color-filtering)
  * [Thresholding](#thresholding)
  * [Closing and Opening](#closing-and-opening)
  * [SimpleBlobDetector](#simpleblobdetector)
* [Data analysis](#detecting-ncap-logo)
  * [Calculating mm per pixel](#calculating-mm-per-pixel)
  * [Delta-distance](#delta-distance)
  * [Distance](#distance)
  * [Speed](#speed)
  * [Frame time](#frame-time)
  * [Acceleration](#acceleration)
  * [Result](#result)

## NCAP crash test
Data from the Euro NCAP is used to determine the distance, speed and acceleration. The datapoints are extracted from a video. The scale is measured using the size of the logo.

### Video
The video that is used in this program is a crash test of a Toyota Hilux 2016 model. The test that is performed is the _FRONTAL OFFSET DEFORMABLE BARRIER IMPACT_ test. This test is performed with a speed of 64km/u. A frame from this video is displayed below.

![frame from ncap video](http://i.imgur.com/vtTua2e.jpg)  
_Video frame_

The NCAP logo in this video is used to determine distance traveled between frames.

The video file can be found in [video/hilux_ncap_short.mp4](https://github.com/JohanOsinga/NCAP-Video-Analyser/blob/master/video/hilux_ncap_short.mp4)

### NCAP logo
The NCAP logo was chosen because it is a yellow circle on the side of the car. This means it has a good contrast with the car and can easily be used to track with some thresholding. The part of the logo that will be tracked is shown below.

![NCAP logo](http://i.imgur.com/460gf7k.png)  
_NCAP logo_

The size of this circle in the logo is 176mm. This will be used later on when analysing the data.

## Detecting NCAP logo
There are a few steps performed before the NCAP logo can be detected accurately.
### Region of interest
First a region of interest (ROI) is specified. The original video footage contains multiple NCAP logo's. One on the car and one as a watermark on the video, this can be seen in the frame displayed above. A ROI can be created because the logo on the car is always around the y-center of the screen. The ROI used in this program is shown below.

![ROI displayed in UI](http://i.imgur.com/fQqmAoC.jpg)  
_ROI displayed in UI_

Creating a ROI in OpenCV using Python is done by creating a binary mask with the same height and width of the original frame. This binary mask has to contain one's at the pixels that need to be in the ROI, the rest is zero's. The ROI is created by by performing a ```cv2.bitwise_and()``` operation on the original image and the mask. Only the pixels matching a 1 in the mask will be kept. The python code is shown below.


```python
#apply roi
height, width = frame.shape[:2]
#np -> y,x
tmp_height = height / 100 * self.roi_margin_perc

#create binary mask
roi_mask = np.zeros((height, width), dtype=np.uint8)
tmp_roi_mask_ones = np.ones(((height-(2*tmp_height)), width), dtype=np.uint8)
roi_mask[tmp_height:(height-tmp_height), 0:width] = tmp_roi_mask_ones

frame_roi = cv2.bitwise_and(frame, frame, mask=roi_mask)
```

The resulting ROI is shown below. This ROI is used later on in the program to detect the NCAP logo.

![ROI as used in program](http://i.imgur.com/P6jaFT2.png)  
_ROI used in program_

### Color filtering
The frame is converted from RGB / BGR to 
[HSV](https://en.wikipedia.org/wiki/HSL_and_HSV).
This makes filtering the colors a lot easier. The user can pick a color by clicking on the image shown in the interface. The x,y-coordinates of the mouse-click are used to get the HSV-value of the pixel in the frame.

The below mask is created by clicking on the yellow part of the NCAP logo in the interface. The thresholding used will be explained later.

![Color threshold yellow](http://i.imgur.com/tSkhLeo.png)  
_Mask created from color_

This originial image is muliplied by the mask using the ```cv2.bitwise_and()``` function. The resulting image is shown below.

![Color masked image](http://i.imgur.com/geV5SuZ.png)  
_Frame color masked_

The above image still shows multiple NCAP logos, but because of the ROI this will not impact the detection.

### Thresholding
The frame is thresholded with the ```cv2.inRange()``` function. This function uses the frame and upper and lower limits for H, S and V to return a binary image of only the pixels which values are within the lower and upper range. The code used by the program to perform the thresholding is shown below.

```python
def _apply_threshold(self, value_h, value_s, value_v):
    """Uses cv2.inRange() to apply threshold"""

    #Set upper and lower threshold values
    upper = (value_h + self.h_margin, value_s[1], value_v + self.v_margin)
    lower = (value_h - self.h_margin, value_s[0], value_v - self.v_margin)

    #threshold image
    thres_img = cv2.inRange(self.cv_img_hsv, lower, upper)
```

The output of this is a binary image. For example, filtering the NCAP logo (yellow) color results in the following mask.

![Color threshold yellow](http://i.imgur.com/tSkhLeo.png)  
_Mask created from color_

This image had to be transformed before it can be used to track the NCAP logo.

### Closing and Opening
The next step is performing two morphological transformations: 
[Closing](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#closing)
and
[Opening](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#opening). 
These transformations transform the binary logo into a binary circle, making it easier to track with the SimpleBlobDetector.

The first transformation that is performed on the masked image (binary) is Closing. This operation closes the gaps in the NCAP logo by first performing a dilate operation and then an erode operation. 

The second transformation is Opening. This is the opposite of Closing. Opening first performs an erode operation and then a dilate operation. This transformation is executed to make sure any noise is removed from the binary image, as this may interfere with the working of the SimpleBlobDetector.

The images below displays the process of opening and closing. First the binary image of the color thresholded ROI is shown.

![Binary color thresholded ROI](http://i.imgur.com/ymE13Na.png)  
_Binary color thresholded ROI_

First the Closing transformation is performed on the binary image. The result is shown below.

![ROI Closed](http://i.imgur.com/OlduNbx.png)  
_Binary ROI after Closing_

Then to filter out the noise in the ROI the Opening transformation is performed.  

![ROI Closed and Openend](http://i.imgur.com/yiHZddu.png)  
_Binary ROI after Closing and Opening_

The following code is used for the Closing and Opening transformations.

```python
#opening and closing
kernel_size = 20
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
```

The final result a circle the size of the NCAP logo from the frame. Perfect for using with the SimpleBlobDetector.

### SimpleBlobDetector
The 
[SimpleBlobDetector](http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html#simpleblobdetector) 
is used to detect the position and size of the NCAP logo from the binary image created after the morphological transformations. 

The SimpleBlobDetector is defined and executed using the following code.

```python
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 10

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(cv2.bitwise_not(img_bin))
```

Executing this SimpleBlobDetector returns one keypoints for the NCAP logo. The keypoint drawn on top of the ROI is shown in the image below.

![Keypoint in ROI](http://i.imgur.com/GWX1pce.png)  
_Keypoint drawn on ROI_

These vision operations are performed for every frame of the video. The result of this is an array with datapoints. These datapoints are then analysed. This process is described in the next part.

## Data Analysis
The datapoints from the SimpleBlobDetector are analysed. The result of this are four graphs with distance traveled, speed and acceleration.

### Predefined values
There are some values that are predefined and not extracted from the datapoints. These are the following:

#### NCAP logo size
The size of the NCAP logo is __176mm__.  
This size is used to calculate the relation between pixels and millimeters.

#### Car speed
The starting speed of the car is __64km/u__.  
This is used to extrapolate the speed at every frame using the starting speed and the distance traveled per frame.

### Calculating mm per pixel
The relation between millimeters an pixels is calculated using the size of the NCAP logo circle and the size of the keypoint (size in pixels). The code below is used when calculating the conversion ratio for millimeters per pixel.

```python
def _calc_mm_px(self, datapoints):
    """Calculated the relation between mm and pixels"""
    mm_px_total = 0
    for frame in datapoints:
        #get the size of the blob
        logo_size = frame[2]
        mm_px = self.ncap_logo_mm / logo_size
        mm_px_total += mm_px
    return mm_px_total / len(datapoints)
```

### Delta-distance
The delta-distance is used for calculating the Δx in the formula for calculating speed.  
The python function below is used when calculating the delta-distance. The input is the datapoints from the vision analysis. The output is an array with distance traveled per frame.

```python
def _calc_delta_distance(self, datapoints):
    """Calculates the distance from datapoints

    input -> datapoints from file

    output -> distance in m per frame
    """
    #mm per px is known, so convert delta pixel to mm
    d_distance_points = []
    prev_point = None
    for point in datapoints:
        if prev_point is not None:
            delta = [point[1][0] - prev_point[0], point[1][1] - prev_point[1]]
            delta_vector = self.__pythagoras(delta[0], delta[1])
            #Apply mm per pixel conversion
            distance_vector = delta_vector * (self.mm_px / 1000)
            d_distance_points.append(distance_vector)
        else:
            prev_point = [0, 0]
        prev_point[0] = point[1][0]
        prev_point[1] = point[1][1]

    return self.__moving_avg(d_distance_points, 5)
```

The data is filtered using a moving average filter before returned. This smooths out the output and removes unwanted inconsistencies.

### Distance
The total distance is calculated from the delta-distance by simply adding the delta-distance values each frame. The function for calculating the distance points is shown below.

```python
def _calc_distance(self, d_distance_points):
    """Calculates the total distance from delta-distance points

    input   -> delta-distance points

    output  -> accumilated distance per frame
    """
    total_distance = 0
    distance_points = [0]
    for d_dist_point in d_distance_points:
        total_distance += d_dist_point
        distance_points.append(total_distance)

    return distance_points
```

This function returns an array with the cumulative distance of every frame.

### Speed
The speed is calculated using the delta-distance points.  
The assumption is made that the first delta-distance point is at the speed of __64km/u__. Every following point is calculated using the relation between distance traveled and speed.  
The function used to calculate the speed is shown below.

```python
def _calc_speed(self, d_distance_points):
    """Calculates speed from delta-distance points
    using known starting speed

    input   -> delta-distance points

    output  -> speed points
    """
    #ASSUMPTION: first d_x is at max speed

    #speed per d_distance
    d_vel_px = self.start_speed / d_distance_points[0]
    vel_points = []
    for d_x in d_distance_points:
        vel = d_x * d_vel_px
        vel_points.append(vel)

    return vel_points
```


### Frame time
The frame time is the time that ellapses between individual frames. The frame time is used to calculate the acceleration from the speed-datapoints and is used to provide the x-axis values when plotting the results.

The frame time is calculated using the delta-distance and the speed values of a frame. It calculates the time it would take to cover that distance with that speed. This is the time that ellapsed. The function that is used to calculate the frame time is shown below.

```python
def _calc_frame_time(self, d_distance_points, vel_points):
    """Calculates frametime from delta_distance and velocity

    input   -> delta-distance, velocity

    output  -> frame time
    """
    #Take velocity between frames
    #Take d_distance between frames
    #Calc time between frames
    frame_times = []
    for i in range(0, len(d_distance_points)):
        if not (d_distance_points[i] == 0) and not (vel_points[i] == 0):
            frame_time = d_distance_points[i] / vel_points[i]
            frame_times.append(frame_time)

    frame_time_avg = 0
    for frame_time in frame_times:
        frame_time_avg += frame_time
    frame_time_avg = frame_time_avg / len(frame_times)

    return frame_time_avg
```

The result is the average of all frame times.  
(In the used video, all the frame times are the same so averaging is not required)

### Acceleration
The final calculation is the acceleration. Acceleration is calculated using the following formula:

a = Δv / Δt

Δv = current speed point - previous speed point  
Δt = frame time

Every frame is iterated over and the Δv is calculated, the Δt is a constant factor. The acceleration is then calculated and added to an array. The function is shown below.

```python
def _calc_acceleration(self, vel_points):
    """Calculates acceleration from velocity points

    input   -> velocity per frame

    output  -> acceleration per frame
    """

    acceleration_points = []
    prev_vel = self.start_speed
    for vel_point in vel_points:
        d_v = prev_vel - vel_point
        d_t = self.frame_time
        a = d_v / d_t
        acceleration_points.append(a)
        prev_vel = vel_point

    return self.__moving_avg(acceleration_points, 5)
```

This function returns the acceleration points. This is the last component needed before the plot can be made.
The acceleration is also calculated in G's, this is done by dividing the acceleration in m/s² by 9,81.

### Result
The results are plotted using Matplotlib. The figure consists of four plots: distance traveled, speed, acceleration in m/s² and acceleration in G's. The resulting plots are displayed below.

![Four plots](http://i.imgur.com/1SnWOu1.png)  
_Plotted results_