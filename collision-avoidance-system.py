
import torch
import numpy as np
import cv2
from time import time
import depthai as dai
import math

from calc_nofactor import HostSpatialsCalc
from texthandler import TextHandler

from multiprocessing import Process, Manager
from pathprocess import CustomProcess

import logging
logging.basicConfig(filename='example_test.log', encoding='utf-8', filemode="a", level=logging.DEBUG,
                            format='%(asctime)s|%(levelname)s|%(message)s')

def calculate_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distance

def startprocess():
    with Manager() as manager:
        general_exchange = manager.dict()
        path_exchange = manager.list()

        p = Process(target=f, args=(
            general_exchange, path_exchange))
        p.start()
        #p.join()

        print(general_exchange)
        print(path_exchange)


class CAS:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def create_pipeline(self):

        # Create pipeline
        pipeline = dai.Pipeline()

        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.video.link(xoutRgb.input)

        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
        camRgb.setInterleaved(False)
        camRgb.setIspScale(1, 4)  # 4056x3040 -> 812x608
        camRgb.setPreviewSize(812, 608)
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setFps(25)

        xoutIsp = pipeline.create(dai.node.XLinkOut)
        xoutIsp.setStreamName("isp")
        camRgb.isp.link(xoutIsp.input)

        #dai.Device(pipeline).getFov()

        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(1080000)  # 300x300x3
        manip.initialConfig.setResizeThumbnail(600, 600)
        camRgb.video.link(manip.inputImage)

        xoutmanip = pipeline.create(dai.node.XLinkOut)
        xoutmanip.setStreamName("manip")
        manip.out.link(xoutmanip.input)

        # depth pipeline
        # Define sources and outputs
        monoLeft = pipeline.create(dai.node.MonoCamera)
        xoutLeft = pipeline.create(dai.node.XLinkOut)
        xoutLeft.setStreamName("left")
        #monoLeft.out.link(xoutLeft.input)

        manipleft = pipeline.create(dai.node.ImageManip)
        manipleft.setMaxOutputFrameSize(1080000)  # 300x300x3
        manipleft.initialConfig.setResizeThumbnail(600, 600)
        monoLeft.out.link(manipleft.inputImage)
        manipleft.out.link(xoutLeft.input)

        monoRight = pipeline.create(dai.node.MonoCamera)


        stereo = pipeline.create(dai.node.StereoDepth)
        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.initialConfig.setConfidenceThreshold(255)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        stereo.depth.link(xoutDepth.input)
        #stereo.disparity.link(xoutDepth.input)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("disp")
        stereo.depth.link(xoutDepth.input)

        manipdepth = pipeline.create(dai.node.ImageManip)
        manipdepth.setMaxOutputFrameSize(1080000)  # 300x300x3
        manipdepth.initialConfig.setResizeThumbnail(600, 600)
        stereo.depth.link(manipdepth.inputImage)

        xoutmanipdepth = pipeline.create(dai.node.XLinkOut)
        xoutmanipdepth.setStreamName("manipdepth")
        manipdepth.out.link(xoutmanipdepth.input)

        return pipeline, stereo

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device_type)

        self.pipeline, self.stereo = self.create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.calibData = self.device.readCalibration()
        self.FOV = self.calibData.getFov(dai.CameraBoardSocket.RGB)

        self.delta = 5
        self.LowerThreshold = 0     # in millimeters
        self.UpperThreshold = 30000 # in millimeters
        self.mapthreshold = 10 # in meters

        self.hostSpatials = HostSpatialsCalc(self.device)
        self.hostSpatials.setDeltaRoi(self.delta)
        self.hostSpatials.setLowerThreshold(self.LowerThreshold)
        self.hostSpatials.setUpperThreshold(self.UpperThreshold)

        self.max_z = 4
        self.min_z = 1
        self.max_x = 0.9
        self.min_x = -0.

        self.text = TextHandler()

        self.mapheight = 1000
        self.mapwidth = 1000
        self.numberofaxis = 10
        self.birdframe = self.make_bird_frame()
        self.minobjectdistance = 0.4 #m

        self.minimumdistance = 0.5

        self.pathlist = []
        self.obstaclelist = []
        self.logginglist = []


    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device_type)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def make_bird_frame(self):
        text = self.text
        img = np.zeros((self.mapheight, self.mapwidth, 3), np.uint8)
        fov = self.FOV
        print(f"fov:{fov}")
        #fov = 68.7938
        min_distance = 0.827
        # frame = np.zeros((320, 100, 3), np.uint8)
        min_y = int((1 - (min_distance - self.min_z) / (self.max_z - self.min_z)) * img.shape[0])
        cv2.rectangle(img, (0, min_y), (img.shape[1], img.shape[0]), (70, 70, 70), -1)

        alpha = (180 - self.FOV) / 2
        center = int(img.shape[1] / 2)
        max_p = img.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array([
            (0, img.shape[0]),
            (img.shape[1], img.shape[0]),
            (img.shape[1], max_p),
            (center, img.shape[0]),
            (0, max_p),
            (0, img.shape[0]),
        ])

        axisdistance = self.mapheight/self.numberofaxis
        lst = list(np.arange(1,self.numberofaxis + 1))

        counter = 0
        counter2 = 0
        for i in lst:
            cv2.line(img, (0, counter * int(axisdistance)), (1000,  counter * int(axisdistance)), (70, 70, 70), 2)

            counter += 1
            counter2 += self.mapthreshold/self.numberofaxis

        cv2.fillPoly(img, [fov_cnt], color=(70, 70, 70))

        counter = 0
        counter2 = 0
        for i in lst:
            #cv2.line(img, (0, counter * int(axisdistance)), (1000,  counter * int(axisdistance)), (70, 70, 70), 2)
            text.putText(img, f"{self.mapthreshold - counter2}", (960, counter * int(axisdistance) -5))
            counter += 1
            counter2 += self.mapthreshold/self.numberofaxis

        text.putText(img, f"distance threshold:{self.mapthreshold}", (10, 900))
        text.putText(img, f"number of axis:{self.numberofaxis}", (10, 880))
        text.putText(img, f"FOV:{self.FOV}", (10, 860))

        print(center)
        print(alpha)
        print(max_p)
        print(fov_cnt)
        return img

    def calc_map_z(self, zval):
        return int(self.mapheight - (zval / self.mapthreshold * self.mapheight))

    def calc_map_x(self, xval):
        return int((xval / 30 * self.mapwidth) + 500)

    def convert_x_to_depth(self, x):
        return int(x/1014*640)

    def convert_y_to_depth(self, y):
        return int(y/760*400)

    def convert_x_to_rgb(self, x):
        return int(x/4)

    def calc_center(self, roi):
        return (int((roi[0]+roi[2])/2), int((roi[1]+roi[3])/2))

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        # Connect to device and start pipeline
        with self.device as device:

            # Output queue will be used to get the rgb frames from the output defined above
            video = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
            dispQ = device.getOutputQueue(name="disp", maxSize=1, blocking=False)
            leftMono = device.getOutputQueue(name="left", maxSize=1, blocking=False)
            ispQueue = device.getOutputQueue(name="isp", maxSize=1, blocking=False)
            manipQueue = device.getOutputQueue(name="manip", maxSize=1, blocking=False)
            manipdepthQueue = device.getOutputQueue(name="manipdepth", maxSize=1, blocking=False)

            text = self.text
            process = CustomProcess(self.obstaclelist, 50)


            while True:
                videoFrame = video.tryGet()

                if videoFrame is not None:
                    frame = videoFrame.getCvFrame()
                    depthFrame = depthQueue.get().getFrame()
                    monoFrame = leftMono.get().getCvFrame()
                    isp = ispQueue.get().getCvFrame()
                    manip = manipQueue.get().getCvFrame()
                    manipdepth = manipdepthQueue.get().getCvFrame()
                    manipdepth = (manipdepth * (63.75 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
                    manipdepth = cv2.applyColorMap(manipdepth, cv2.COLORMAP_JET)
                    print(f"frame shape:{frame.shape}")

                    disp = dispQ.get().getFrame()
                    #change to 1020
                    disp = (disp * (63.75 / self.stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
                    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

                    start_time = time()
                    results = self.score_frame(frame)


                    labels, cord = results
                    n = len(labels)
                    x_shape, y_shape = frame.shape[1], frame.shape[0]
                    self.birdframe = self.make_bird_frame()

                    print(f"birdframe array: {self.birdframe} birdframe shape:{self.birdframe.shape} ")


                    for i in range(n):
                        row = cord[i]
                        if row[4] >= 0.7:
                            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)

                            text.rectangle(frame, (x1, y1), (x2, y2))
                            text.putText(frame, self.class_to_label(labels[i]), (x1, y1))
                            #--> give roi
                            #xmin, ymin, xmax, ymax
                            depthroi = (self.convert_x_to_depth(x1), self.convert_y_to_depth(y1), self.convert_x_to_depth(x2), self.convert_y_to_depth(y2))
                            spatials, centroid = self.hostSpatials.calc_spatials(depthFrame, depthroi)
                                                                             # centroid == x/y in our case
                            print(f"spatials: {spatials}, centroid: {centroid}")

                            text.circle(disp, (centroid['x'], centroid['y']), 2, (0, 0, 255), 2)

                            depthroi_checked_input = self.hostSpatials._check_input(self.calc_center(depthroi), depthFrame)
                            spatials_center_rectangle, centroid_center_rectangle = self.hostSpatials.calc_spatials(depthFrame, depthroi_checked_input)

                            # Get disparity frame for nicer depth visualization
                            roi = x1, y1, x2, y2
                            rgb_center = self.calc_center(roi)
                            depth_center = self.calc_center(depthroi)


                            print(f"calc center: roi:{self.calc_center(roi)}, depthroi:{self.calc_center(depthroi)}")


                            print(f"depthroi: {depthroi}")
                            print(f"depthroi_checked: {depthroi_checked_input}")
                            print(f"roi:{roi}")
                            #print(f"roi_checked_input:{roi_checked_input}")
                            #--> activte when 800
                            #text.rectangle(disp, (depthroi[0]*2, depthroi[1]*2), (depthroi[2]*2, depthroi[3]*2))
                            text.rectangle(disp, (depthroi[0], depthroi[1]), (depthroi[2], depthroi[3]))
                            #cv2.rectangle(frame, (roi_checked_input[0], roi_checked_input[1]), (roi_checked_input[2], roi_checked_input[3]), bgr, 2)
                            #text.rectangle(disp, (x1, y1), (x1, y1))
                            text.putText(disp, "X: " + (
                                "{:.1f}m".format(spatials_center_rectangle['x'] / 1000) if not math.isnan(spatials_center_rectangle['x']) else "--"),
                                         (depthroi_checked_input[0] + 30, depthroi_checked_input[1] + 20))
                            text.putText(disp, "Y: " + (
                                "{:.1f}m".format(spatials_center_rectangle['y'] / 1000) if not math.isnan(spatials_center_rectangle['y']) else "--"),
                                         (depthroi_checked_input[0] + 30, depthroi_checked_input[1] + 35))
                            text.putText(disp, "Z: " + (
                                "{:.1f}m".format(spatials_center_rectangle['z'] / 1000) if not math.isnan(spatials_center_rectangle['z']) else "--"),
                                         (depthroi_checked_input[0] + 30, depthroi_checked_input[1] + 50))

                            text.rectangle(disp, (depthroi_checked_input[0], depthroi_checked_input[1]), (depthroi_checked_input[2], depthroi_checked_input[3]))


                            text.putText(disp, "X: " + (
                                "{:.1f}m".format(spatials['x'] / 1000) if not math.isnan(spatials['x']) else "--"),
                                         (depthroi[0] + 10, depthroi[1] + 20))
                            text.putText(disp, "Y: " + (
                                "{:.1f}m".format(spatials['y'] / 1000) if not math.isnan(spatials['y']) else "--"),
                                         (depthroi[0] + 10, depthroi[1] + 35))
                            text.putText(disp, "Z: " + (
                                "{:.1f}m".format(spatials['z'] / 1000) if not math.isnan(spatials['z']) else "--"),
                                         (depthroi[0] + 10, depthroi[1] + 50))

                            print(f"spatials x:{spatials['x']}")
                            if not math.isnan(spatials['x']) and not math.isnan(spatials_center_rectangle['x']):
                                z = spatials['z'] / 1000 if spatials['z'] < spatials_center_rectangle['z'] else spatials_center_rectangle['z'] / 1000
                                x = spatials_center_rectangle['x'] / 1000 if z == spatials_center_rectangle['z'] / 1000 else spatials['x'] / 1000
                                y = spatials_center_rectangle['y'] / 1000 if z == spatials_center_rectangle['z'] / 1000 else spatials['y'] / 1000

                                print(f"z:{z}, y:{y}, x:{x}")

                                text.putText(frame, "X: " + (
                                    "{:.1f}m".format(x) if not math.isnan(x) else "--"),
                                             (x1 + 10, y1 + 20))
                                text.putText(frame, "Y: " + (
                                    "{:.1f}m".format(y) if not math.isnan(y) else "--"),
                                             (x1 + 10, y1 + 35))
                                text.putText(frame, "Z: " + (
                                    "{:.1f}m".format(z) if not math.isnan(z) else "--"),
                                             (x1 + 10, y1 + 50))

                                map_x = self.calc_map_x(x)
                                map_y = self.calc_map_z(z)

                                pixel_x = int(x / self.mapthreshold * self.mapwidth + self.mapwidth / 2)
                                pixel_y = int(self.mapheight - z / self.mapthreshold * self.mapheight)
                                print(f"pixel : {pixel_x}, {pixel_y}")
                                print(f"map: x:{map_x}, y:{map_y}")

                            # data = [label, smallest spatials, bird view coords, roi, depthroi, innerdisproi]
                            data = [self.class_to_label(labels[i]), [x,y,z], [pixel_x, pixel_y], roi, depthroi, depthroi_checked_input]
                            self.logginglist.append(data)
                            color = (0, 0, 255)

                            text.circle(self.birdframe, (pixel_x, pixel_y), 2, color, 2)

                            # border rectangle coordinates
                            x_factor = self.mapwidth / self.mapthreshold
                            y_factor = self.mapheight / self.mapthreshold

                            x_pixel_distance = x_factor * self.minobjectdistance
                            y_pixel_distance = y_factor * self.minobjectdistance

                            print(f"xfactor:{x_factor} xpixel:{x_pixel_distance}")

                            mapx1 = (int(pixel_x - x_pixel_distance), int(pixel_y - y_pixel_distance))
                            mapx2 = (int(pixel_x - x_pixel_distance), int(pixel_y + y_pixel_distance))
                            mapy1 = (int(pixel_x + x_pixel_distance), int(pixel_y - y_pixel_distance))
                            mapy2 = (int(pixel_x + x_pixel_distance), int(pixel_y + y_pixel_distance))

                            text.rectangle(self.birdframe, (mapx1[0], mapx1[1]), (mapy2[0], mapy2[1]))

                            self.obstaclelist.append([mapx1[0]//20, mapy1[0]//20, mapx1[1]//20, mapx2[1]//20])

                            print("not changed",mapx1[1], mapx2[1], mapx1[0], mapy1[0])
                            print("changed",mapx1[1]//10, mapx2[1]//10, mapx1[0]//10, mapy1[0]//10)
                            print(f"points:{mapx1, mapx2, mapy1, mapy2}")

                            #save.save([x,y,z])
                    if process.is_alive() == False:
                        print("obstaclelist", self.obstaclelist)
                        process = CustomProcess(self.obstaclelist, 50)
                        process.start()

                        print('Waiting for the child process to finish')

                        print(f'Parent got: {process.obstaclelist}')
                        print(process.path)
                        pathlistx = [process.path[i] * (1000 // 50) + 20 for i in range(len(process.path) // 2)]
                        pathlisty = [process.path[i] * (1000 // 50) + 20 for i in
                                     range(len(process.path) // 2, len(process.path))]
                        print(f"pathlistx:{pathlistx}, len:{len(pathlistx)}")
                        print(f"pathlisty:{pathlisty}, len:{len(pathlisty)}")
                        pathlist = []
                        for i in range(len(pathlistx)):
                            pathlist.append([pathlistx[i], pathlisty[i]])
                        pathlist = np.array(pathlist, dtype=np.int32)
                        print(f"pathlist:{pathlist}")
                        self.pathlist = pathlist
                    print("pathlistdebug", self.pathlist)
                    if len(self.pathlist) > 0:
                        cv2.polylines(self.birdframe, [self.pathlist], False, (0, 255, 255))
                    self.obstaclelist = []
                    end_time = time()
                    fps = 1 / np.round(end_time - start_time, 2)
                    print(f"Frames Per Second : {fps}")

                    text.putText(frame, f'FPS: {int(fps)}', (900, 70))
                    if len(self.logginglist) > 0:
                        filename = time()
                        filepath = f'C:/Users/johan/images/{filename}.jpg'
                        filepathdepth = f'C:/Users/johan/images/d{filename}.jpg'
                        filepathmap = f'C:/Users/johan/images/m{filename}.jpg'
                        logging.info(f"{self.logginglist}|{filepath}")
                        cv2.imwrite(filepath, frame)
                        cv2.imwrite(filepathdepth, disp)
                        cv2.imwrite(filepathmap, self.birdframe)
                        self.logginglist = []
                    cv2.imshow("birdframe", self.birdframe)
                    cv2.imshow("depth", disp)
                    cv2.imshow('YOLOv5 Detection', frame)
                    cv2.imshow("monoLeft", monoFrame)
                    cv2.imshow("isp", isp)
                    cv2.imshow("manip", manip)
                    cv2.imshow("manip depth", manipdepth)


                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        break


# Create a new object and execute.
if __name__ == '__main__':
    detector = CAS(capture_index=0, model_name="yolov5l")
    detector()
#models/yolov5s/bestl.pt