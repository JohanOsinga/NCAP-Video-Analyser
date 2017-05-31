"""ncap_video_analyser.py
"""
from __future__ import division
import math
import time
from tkFileDialog import askopenfilename
from Tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt


class NCAPVideoAnalyser(object):
    """NCAPVideoAnalyser
    """

    def __init__(self):
        """Constructor"""

        #########################################
        # General settings
        #########################################
        #Image width setting (in px)
        self.img_max_size = 800
        self.h_margin = 20
        self.v_margin = 40
        self.roi_margin_perc = 40

        #Sidebar settings
        self.btn_padx = 45
        self.btn_pady = 5

        #Size of NCAP logo
        self.ncap_logo_mm = 176

        #########################################
        # Variables
        #########################################
        #Image to display
        self.cv_img = None
        self.cv_img_hsv = None
        self.tk_img = None
        self.cv_img_filtered = None
        self.path = None

        #UI settings
        self.lock_ui = False

        #Vars for image
        self.value_h = 0
        self.value_s_min = 0
        self.value_s_max = 0
        self.value_v = 0

        #Vars for data analysis
        self.datapoints = None
        self.start_speed = (64 / 3.6) #in m/s
        self.mm_px = 0 #mm per pixel
        self.frame_time = 0

        #########################################
        # Tkinter setup
        #########################################

        #setup Tkinter
        self.root = Tk()
        self.root.title("NCAP Video Analyser")
        self.root.resizable(width=False, height=False)
        #set default window size
        self.__resize_window(715, 600)
        self.root.update()

        #########################################
        # Program start
        #########################################

        #create UI
        self._draw_ui()

        #loop
        self.__main_loop()


    def _draw_ui(self):
        #add menu items
        menubar = Menu(self.root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open video", command=lambda: self.__btn_pressed(1))

        filemenu.add_separator()

        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)


        #setting up a tkinter canvas
        self.frame_top = Frame(self.root, bd=2, width=800, height=400)
        self.frame_top.grid_rowconfigure(0, weight=1)
        self.frame_top.grid_columnconfigure(0, weight=1)
        self.canvas = Canvas(self.frame_top, bd=0)
        self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
        self.frame_top.pack(fill=BOTH, expand=1)
        self.canvas.bind("<Button 1>", self.__canvas_coords)

        self.frame_bottom = Frame(self.root)

        #show status label
        self.status = StringVar()
        Label(self.frame_bottom, textvariable=self.status).pack(in_=self.frame_bottom, side=LEFT)
        self.status.set('Open file')

        #show original img button
        self.btn_show_orig_img = Button(self.frame_bottom,
                                        text="Show original",
                                        command=lambda: self.__btn_pressed(3))
        self.btn_show_orig_img['state'] = 'disabled'
        self.btn_show_orig_img.pack(padx=self.btn_padx, pady=self.btn_pady)
        self.btn_show_orig_img.pack(in_=self.frame_bottom, side=LEFT)


        #load execute button
        self.btn_execute_video = Button(self.frame_bottom,
                                        text="Execute video",
                                        command=lambda: self.__btn_pressed(4))
        self.btn_execute_video['state'] = 'disabled'
        self.btn_execute_video.pack(padx=self.btn_padx, pady=self.btn_pady+10)
        self.btn_execute_video.pack(in_=self.frame_bottom, side=LEFT)

        #load analyse data button
        self.btn_analyse_results = Button(self.frame_bottom,
                                          text="Analyse",
                                          command=lambda: self.__btn_pressed(2))
        self.btn_analyse_results['state'] = 'disabled'
        self.btn_analyse_results.pack(padx=self.btn_padx, pady=self.btn_pady+10)
        self.btn_analyse_results.pack(in_=self.frame_bottom, side=LEFT)


        self.frame_bottom.pack()

    def new_image(self):
        """Opens new image using file dialog"""
        #Get path from dialog
        self.path = askopenfilename(parent=self.root, title='Choose a video.')
        if self.path == '':
            print("No video selected")
            return

        #open video, get first frame
        cap = cv2.VideoCapture(self.path)
        if cap.isOpened():
            ret, frame = cap.read()
        else:
            print("error?")
            return

        #close video
        cap.release()

        #get dimensions
        height, width = frame.shape[:2]
        #calc x-y factor
        xy_factor = width / height

        if width >= height:
            new_img_width = self.img_max_size
            new_img_height = int(self.img_max_size/xy_factor)
        else:
            new_img_width = int(self.img_max_size*xy_factor)
            new_img_height = self.img_max_size

        #resize
        frame = cv2.resize(frame, (new_img_width, new_img_height))
        self.cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cv_img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #empty filtered image
        self.cv_img_filtered = None

        #display on screen
        self.__display_cv_img_on_screen(self.cv_img)

        #reset everything
        self.status.set('Pick color from image')
        self.btn_show_orig_img['state'] = 'disabled'
        self.btn_execute_video['state'] = 'disabled'
        self.btn_analyse_results['state'] = 'disabled'
        self.datapoints = None

    def _analyse_datapoints(self, datapoints):
        """Analyses datapoints and return plots

        input   -> datapoints (keypoints)

        output  -> plots of delta-distance,
                            distance,
                            speed,
                            acceleration
        """

        #calculate mm per pixel
        self.mm_px = self._calc_mm_px(datapoints)

        #Calc delta distance over time
        d_distance_points = self._calc_delta_distance(datapoints)

        #Calc distance over time
        distance_points = self._calc_distance(d_distance_points)

        #Calc speed over time
        vel_points = self._calc_speed(d_distance_points)

        #Calc frame times
        self.frame_time = self._calc_frame_time(d_distance_points, vel_points)

        #Calc and plot acceleration over time
        acceleration_points = self._calc_acceleration(vel_points)
        acceleratoin_points_g = self._calc_acceleration_g(vel_points)

        #plot graphs
        fig = plt.figure()

        plot_1 = fig.add_subplot(221)
        plot_1_points = self.__get_plot_time_points(distance_points)
        plot_1.plot(plot_1_points[0], plot_1_points[1])
        plot_1.set_title("total distance [m]")

        plot_2 = fig.add_subplot(222)
        plot_2_points = self.__get_plot_time_points(vel_points)
        plot_2.plot(plot_2_points[0], plot_2_points[1])
        plot_2.set_title("velocity [m/s]")

        plot_3 = fig.add_subplot(223)
        plot_3_points = self.__get_plot_time_points(acceleration_points)
        plot_3.plot(plot_3_points[0], plot_3_points[1])
        plot_3.set_title("acceleration [m/s2]")

        plot_4 = fig.add_subplot(224)
        plot_4_points = self.__get_plot_time_points(acceleratoin_points_g)
        plot_4.plot(plot_4_points[0], plot_4_points[1])
        plot_4.set_title("acceleration [G]")

        plt.show()


    def __get_plot_time_points(self, datapoints):
        """Returns plot time based datapoints"""
        x_values = []
        y_values = []
        x_value = 0
        for point in datapoints:
            x_values.append(x_value * self.frame_time)
            y_values.append(point)
            x_value += 1
        return (x_values, y_values)

    def _calc_mm_px(self, datapoints):
        """Calculated the relation between mm and pixels"""
        mm_px_total = 0
        for frame in datapoints:
            #get the size of the blob
            logo_size = frame[2]
            mm_px = self.ncap_logo_mm / logo_size
            mm_px_total += mm_px
        return mm_px_total / len(datapoints)


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

    def _calc_acceleration(self, vel_points):
        """Calculates acceleration from velocity points

        input   -> velocity per frame

        output  -> acceleration per frame
        """
        # a = d_v / d_t
        # d_v = vel_point
        # d_t = frame_time

        acceleration_points = []
        prev_vel = self.start_speed
        for vel_point in vel_points:
            d_v = prev_vel - vel_point
            d_t = self.frame_time
            a = d_v / d_t
            acceleration_points.append(a)
            prev_vel = vel_point

        return self.__moving_avg(acceleration_points, 5)

    def _calc_acceleration_g(self, vel_points):
        """Calculates acceleration from velocity points in G's

        input   -> velocity per frame

        output  -> acceleration in G per frame
        """
        # a = d_v / d_t
        # d_v = vel_point
        # d_t = frame_time

        g = 9.81

        acceleration_points = []
        prev_vel = self.start_speed
        for vel_point in vel_points:
            d_v = prev_vel - vel_point
            d_t = self.frame_time
            a = d_v / d_t
            a_g = a / g
            acceleration_points.append(a_g)
            prev_vel = vel_point

        return self.__moving_avg(acceleration_points, 5)


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

    def __pythagoras(self, x, y):
        """Return pythagoras of x, y"""
        return math.sqrt(pow(x, 2) + pow(y, 2))

    def __moving_avg(self, x, N):
        """Moving average filter

        input   -> x : samples

                   N : kernel size

        output  -> filtered x
        """
        return np.convolve(x, np.ones((N,))/N)[(N-1):]

    def _apply_to_video(self, value_h, value_s, value_v):
        """Applies the settings to the video, to find the NCAP marker"""
        #time the operation
        start_time = time.time()

        #open video
        cap = cv2.VideoCapture(self.path)
        frame_id = 0
        self.datapoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            upper = (value_h + self.h_margin, value_s[1], value_v + self.v_margin)
            lower = (value_h - self.h_margin, value_s[0], value_v - self.v_margin)

            #apply roi
            height, width = frame.shape[:2]
            #np -> y,x
            tmp_height = height / 100 * self.roi_margin_perc

            #create binary mask
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            tmp_roi_mask_ones = np.ones(((height-(2*tmp_height)), width), dtype=np.uint8)
            roi_mask[tmp_height:(height-tmp_height), 0:width] = tmp_roi_mask_ones

            frame_roi = cv2.bitwise_and(frame, frame, mask=roi_mask)
            thres_img = cv2.inRange(frame_roi, lower, upper)

            masked = cv2.bitwise_and(frame, frame, mask=thres_img)

            frame_id += 1
            self.status.set('Frame ' + str(frame_id))
            #find keypoints
            keypoints, img_bin = self._find_ncap_marker(thres_img)
            im_with_keypoints = cv2.drawKeypoints(masked,
                                                  keypoints,
                                                  np.array([]),
                                                  (255, 255, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            for keypoint in keypoints:
                keypoint.size = keypoint.size * 2

            #save datapoints
            self.datapoints.append([frame_id, keypoints[0].pt, keypoints[0].size])

            self.root.update()

        d_time = round(time.time() - start_time,2)

        #update ui
        self.status.set(str(frame_id) + " frames in " + str(d_time) + "s")
        self.btn_analyse_results['state'] = 'normal'

        #close video when done
        cap.release()

    def _find_ncap_marker(self, frame):
        """Finds NCAP marker keypoints in frame"""
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


        #create binary image
        _, img_bin = cv2.threshold(frame,
                                   params.minThreshold,
                                   params.maxThreshold,
                                   cv2.THRESH_BINARY)
        #opening and closing
        kernel_size = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

        # Detect blobs.
        keypoints = detector.detect(cv2.bitwise_not(img_bin))

        #return the keypoints, and the binary image used
        return (keypoints, img_bin)

    def _apply_threshold(self, value_h, value_s, value_v):
        """Uses cv2.inRange() to apply threshold"""

        #Set upper and lower threshold values
        upper = (value_h + self.h_margin, value_s[1], value_v + self.v_margin)
        lower = (value_h - self.h_margin, value_s[0], value_v - self.v_margin)

        #threshold image
        thres_img = cv2.inRange(self.cv_img_hsv, lower, upper)

        #set global img
        self.cv_img_filtered = thres_img

        #invert image for mask
        masked = cv2.bitwise_and(self.cv_img, self.cv_img, mask=thres_img)

        #display on window canvas
        self.__display_cv_img_on_screen(masked)

    def __set_ui_lock(self, locked):
        """Set the UI lock, useful for processing loops"""
        self.lock_ui = locked

    def __display_cv_img_on_screen(self, cv_disp_img):
        """Displays an image on the windows canvas"""
        #apply roi
        height, width = cv_disp_img.shape[:2]
        tmp_height = int(height / 100 * self.roi_margin_perc)

        #create roi mask
        cv2.rectangle(cv_disp_img,
                      (0, tmp_height),
                      (width, int(height-tmp_height)),
                      (0, 255, 0),
                      3)

        #convert to tk_img
        #convert color from BGR to RGB
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(cv_disp_img))
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox(ALL))

        #resize window
        self.__resize_window_to_image(self.tk_img.width(), self.tk_img.height())

    def __canvas_coords(self, event):
        """Gets the coordinates from mouse click and applies threshold"""
        #check if interface is not locked
        if not self.lock_ui:
            rgb_pixel = self.cv_img_hsv[event.y, event.x]

            self.value_h = rgb_pixel[0]
            self.value_s_min = 0
            self.value_s_max = 255
            self.value_v = rgb_pixel[2]

            #apply threshold
            self._apply_threshold(rgb_pixel[0], (0, 255), rgb_pixel[2])

            #update ui
            self.status.set('Color selected, execute to continue')
            self.btn_show_orig_img['state'] = 'normal'
            self.btn_execute_video['state'] = 'normal'

    def __btn_pressed(self, event):
        """Handles internal button presses"""
        if event == 1:
            #Load new image
            self.new_image()
        elif event == 2:
            #analyse datapoints
            if self.datapoints is not None:
                self._analyse_datapoints(self.datapoints)
        elif event == 3:
            #Only applies when filtered img is available
            if self.cv_img_filtered is not None:
                #Check checkbox status
                self.__display_cv_img_on_screen(self.cv_img)
        elif event == 4:
            #Only applies when filtered img is available
            if self.cv_img_filtered is not None:
                #lock ui
                self.__set_ui_lock(True)
                #apply thresholding to videi
                self._apply_to_video(
                    self.value_h,
                    [self.value_s_min, self.value_s_max],
                    self.value_v
                )
                #unlock ui
                self.__set_ui_lock(False)
                #return image to original
                self.__display_cv_img_on_screen(self.cv_img)

    def __main_loop(self):
        """Execute tkinter main loop"""
        self.root.mainloop()

    def __resize_window(self, window_width, window_height):
        """Change window dimensions"""
        self.root.geometry('{}x{}'.format(window_width, window_height))

    def __resize_window_to_image(self, img_width, img_height):
        """Resize window to fit image"""
        self.__resize_window(img_width, img_height+50)



if __name__ == "__main__":
    NCAP = NCAPVideoAnalyser()
