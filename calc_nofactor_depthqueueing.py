import math
import numpy as np
import depthai as dai
from depthqueueing import DepthQueueModifier

class HostSpatialsCalc:
    # We need device object to get calibration data
    def __init__(self, device):
        calibData = device.readCalibration()
        # Required information for calculating spatial coordinates on the host
        self.monoHFOV = np.deg2rad(calibData.getFov(dai.CameraBoardSocket.LEFT))

        # Values
        self.DELTA = 50
        self.THRESH_LOW = 200 # 20cm
        self.THRESH_HIGH = 30000 # 30m
        self.depthqueue = DepthQueueModifier()

    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low
    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low
    def setDeltaRoi(self, delta):
        self.DELTA = delta

    def _check_input(self, roi, frame): # Check if input is ROI or point. If point, convert to ROI
        if len(roi) == 4: return roi
        if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
        # Limit the point so ROI won't be outside the frame
        #self.DELTA = 50 # Take 10x10 depth pixels around point for depth averaging
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x-self.DELTA,y-self.DELTA,x+self.DELTA,y+self.DELTA)

    def _calc_angle(self, frame, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (frame.shape[1] / 2.0))
    '''
    # roi has to be list of ints
    def calc_spatials(self, depthFrame, roi, averaging_method=np.median):
        roi = self._check_input(roi, depthFrame)  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi
        print(f"roi: {roi}, {xmin}, {ymin}, {xmax}, {ymax}")

        print(f"shape depthframe: {depthFrame.shape}")

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])
        print(f"averageDepth before main:{averageDepth}")
        #depth queueing

        meanqueuedepth = self.depthqueue.run(averageDepth)
        print(f'mean depth after print {meanqueuedepth}')

        averageDepth = meanqueuedepth
        print(f"averageDepth after:{averageDepth}")




        centroid = {  # Get centroid of the ROI
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        midW = int(depthFrame.shape[1] / 2)  # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2)  # middle of the depth img height
        #print(f"midw:{midW}, midH:{midH}")
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        angle_x = self._calc_angle(depthFrame, bb_x_pos)
        angle_y = self._calc_angle(depthFrame, bb_y_pos)

        spatials = {
            'z': averageDepth,
            'x': averageDepth * math.tan(angle_x),
            'y': -averageDepth * math.tan(angle_y)
        }
        return spatials, centroid
        '''
    # roi has to be list of ints
    def calc_averagedepth(self, depthFrame, roi, averaging_method=np.median):
        roi = self._check_input(roi, depthFrame)  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi
        print(f"roi: {roi}, {xmin}, {ymin}, {xmax}, {ymax}")

        print(f"shape depthframe: {depthFrame.shape}")

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])
        print(f"averageDepth before main:{averageDepth}")
        #depth queueing

        return averageDepth

    # roi has to be list of ints
    def calc_spatials(self, depthFrame, roi, averageDepth):
        roi = self._check_input(roi, depthFrame)  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi
        print(f"roi: {roi}, {xmin}, {ymin}, {xmax}, {ymax}")

        print(f"shape depthframe: {depthFrame.shape}")

        centroid = {  # Get centroid of the ROI
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        midW = int(depthFrame.shape[1] / 2)  # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2)  # middle of the depth img height
        #print(f"midw:{midW}, midH:{midH}")
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        angle_x = self._calc_angle(depthFrame, bb_x_pos)
        angle_y = self._calc_angle(depthFrame, bb_y_pos)

        spatials = {
            'z': averageDepth,
            'x': averageDepth * math.tan(angle_x),
            'y': -averageDepth * math.tan(angle_y)
        }
        return spatials, centroid