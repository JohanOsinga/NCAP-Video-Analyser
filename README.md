# NCAP-Video-Analyser
This project contains a program that analyses an NCAP crash test video and calculates the traveled distance, speed and acceleration of the car.

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
* [Program](#program)
  * [Tkinter](#tkinter)
  * [Matplotlib](#matplotlib)
  * [OpenCV](#opencv)

## NCAP crash test
Data from the Euro NCAP is used to determine the distance, speed and acceleration. The datapoints are extracted from a video. The scale is measured using the size of the logo.

### Video
The video that is used in this program is a crash test of a Toyota Hilux 2016 model. The test that is performed is the _FRONTAL OFFSET DEFORMABLE BARRIER IMPACT_ test. This test is performed with a speed of 64km/u. A frame from this video is displayed below.

![frame from ncap video](http://i.imgur.com/vtTua2e.jpg)
__Video frame__

The NCAP logo in this video is used to determine travelled distance between frames.

The video file can be found in [video/hilux_ncap_short.mp4](https://github.com/JohanOsinga/NCAP-Video-Analyser/blob/master/video/hilux_ncap_short.mp4)

### NCAP logo
The NCAP logo was chosen because it is a yellow circle on the side of the car. This means it has a good contrast with the car and can easily be used to track with some thresholding. The part of the logo that will be tracked is shown below.

![NCAP logo](http://i.imgur.com/460gf7k.png)

The size of this circle in the logo is 176mm. This will be used later on when analysing the data.

## Detecting NCAP logo
There are a few steps performed before the NCAP logo can be detected accurately.
### Region of interest
First a region of interest (ROI) is specified. The original video footage contains multiple NCAP logo's. One on the car and one as a watermark on the video, this can be seen in the frame displayed above. A ROI can be created because the logo on the car is always around the y-center of the screen. The ROI used in this program is shown below.

![ROI displayed in UI](http://i.imgur.com/fQqmAoC.jpg)
__ROI displayed in UI__

Creating a ROI in OpenCV using Python is done by creating a binary mask with the same height and width of the original frame. This binary mask has to contain one's at the pixels that need to be in the ROI, the rest is zero's. The ROI is created by by performing a ```cv2.bitwise_and()``` operation on the original image and the mask. Only the pixels matching a 1 in the mask will be kept. The python code is shown below.

__Applying ROI to frame__
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
__ROI used in program__

### Color filtering
The frame is converted from RGB / BGR to 
[HSV](https://en.wikipedia.org/wiki/HSL_and_HSV).
This makes filtering the colors a lot easier. The user can pick a color by clicking on the image shown in the interface. The x,y-coordinates of the mouse-click are used to get the HSV-value of the pixel in the frame.

The below mask is created by clicking on the yellow part of the NCAP logo in the interface. The thresholding used will be explained later.

![Color threshold yellow](http://i.imgur.com/tSkhLeo.png)
__Mask created from color__

This originial image is muliplied by the mask using the ```cv2.bitwise_and()``` function. The resulting image is shown below.

![Color masked image](http://i.imgur.com/geV5SuZ.png)
__Frame color masked__

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
__Mask created from color__

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
__Binary color thresholded ROI__

First the Closing transformation is performed on the binary image. The result is shown below.

![ROI Closed](http://i.imgur.com/OlduNbx.png)
__Binary ROI after Closing__

Then to filter out the noise in the ROI the Opening transformation is performed.

![ROI Closed and Openend](http://i.imgur.com/yiHZddu.png)
__Binary ROI after Closing and Opening__

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
__Keypoint drawn on ROI__

These vision operations are performed for every frame of the video. The result of this is an array with datapoints. These datapoints are then analysed. This process is described in the next part.

## Data Analysis
Analysing keypoint data 

### Predefined values
Predefined values eg, begin speed, size of ncap marker

### Calculating mm per pixel
MM_PX

### Delta-distance
ddistance

### Distance
Distance

### Speed
Speed

### Frame time
Calculating frame timing

### Acceleration
Acceleration

## Program
Program

### Tkinter
tkinter for UI

### Matplotlib
matplotlib for graphs

### OpenCV
opencv for computer vision
