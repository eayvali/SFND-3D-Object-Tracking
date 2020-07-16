#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:18:33 2020

@author: elif.ayvali
"""

import numpy as np
import cv2 as cv
from recordtype import recordtype #mutable namedtuple
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import copy

####################################################################################################
########################----------YOLO V3 OpenCV Implementation------------#########################
####################################################################################################    
class ObjectDetectionOpenCV:
    #https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    def __init__(self,params,labelsPath,weightsPath,configPath,bBox):
        #bBox=recordtype('BoundingBox',['boxID','trackID','roi','classID', 'confidence','lidarPoints','keypoints','kptMatches'])
        self.BoundingBox=bBox
        self.params=params
        # load the COCO class labels our YOLO model was trained on
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),    dtype="uint8")
        # derive the paths to the YOLO weights and model configuration
        self.net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
        

        # determine only the *output* layer names that we need from YOLO
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def predict(self,image):
        # load our input image and grab its spatial dimensions
        (H, W) = image.shape[:2]
        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=False, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        
        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.params["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences,self.params["confidence"],self.params["threshold"])              
        # ensure at least one detection exists
        bBoxes=[]
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                # cv.rectangle(image, (x, y), (x + w, y + h), color, 2)

                #['boxID','trackID','roi','classID', 'confidence']
                [x,y,w,h]=boxes[i]
                x_shrink=int(x+self.params["shrink"]*w/2.0)
                y_shrink=int(y+self.params["shrink"]*h/2.0)
                w_shrink=int(w*(1-self.params["shrink"]))
                h_shrink=int(h*(1-self.params["shrink"]))
                cv.rectangle(image, (x_shrink, y_shrink), (x_shrink + w_shrink, y_shrink + h_shrink), color, 1)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv.putText(image, text, (x_shrink+10, y_shrink - 5), cv.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
                self.BoundingBox.boxID=len(bBoxes)
                self.BoundingBox.classID=classIDs[i]
                self.BoundingBox.confidence=confidences[i]
                self.BoundingBox.roi=[x_shrink, y_shrink, w_shrink, h_shrink]
                bBoxes.append(copy.copy(self.BoundingBox))
        return bBoxes,image        
    
            
            
####################################################################################################
########################----------YOLO V3 KERAS Implementation-------------#########################
####################################################################################################

# modified from https://github.com/experiencor/keras-yolo3
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

#  preparethe image
def load_image_pixels(img, shape):
    #shape:width_height
    height, width = img.shape[:2]
    # load the image with the required size
    image = cv.resize(img, dsize=shape)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = np.expand_dims(image, 0)
    return image, width, height

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores

 
    
    
class ObjectDetection:
        # load yolov3 model and perform object detection
        # https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
    def __init__(self,params,labelsPath,modelPath,bBox):
        #bBox=recordtype('BoundingBox',['boxID','trackID','roi','classID', 'confidence','lidarPoints','keypoints','kptMatches'])
        self.BoundingBox=bBox
        self.params=params       
        # load the COCO class labels our YOLO model was trained on
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),    dtype="uint8")
        
        #load YOLO model
        self.model =load_model(modelPath)
        self.img_original=[]
        
    def predict(self,img):   
        self.img_original=img.copy()
        # define the expected input shape for the model
        input_w, input_h = 416, 416
        # load and prepare image
        image, image_w, image_h = load_image_pixels(img, (input_w, input_h))
        # make prediction
        yhat = self.model.predict(image)
        # define the anchors
        anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = list()
        # loop over eac 
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], anchors[i], self.params["threshold"], input_h, input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)


        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        # boxes = []
        # confidences = []
        # classIDs = []
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], anchors[i], self.params["threshold"], input_h, input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        # suppress non-maximal boxes
        do_nms(boxes,self.params["confidence"])
        v_boxes, v_labels, v_scores = get_boxes(boxes, self.LABELS, self.params["threshold"])        
        # plot each box
        bBoxes=[]
        for i in range(len(v_boxes)):
            box = v_boxes[i]
            # get coordinates
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1            
            # draw a bounding box rectangle and label on the image           
            color = [int(c) for c in self.COLORS[self.LABELS.index(v_labels[i])]] 
            cv.rectangle(self.img_original, (x1, y1), (x1 + width, y1 + height), color, 2)
            text =  "%s (%.3f)" % (v_labels[i], v_scores[i])
            cv.putText(self.img_original, text, (x1+10, y1 - 5), cv.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
            #Export as Udacity bBoxes data Structure
            #['boxID','trackID','roi','classID', 'confidence']
            self.BoundingBox.boxID = len(v_boxes)
            self.BoundingBox.classID = v_labels[i]
            self.BoundingBox.confidence = v_scores[i]
            self.BoundingBox.roi = [x1, y1, int(width), int(height)]
            bBoxes.append(self.BoundingBox)
        return bBoxes,self.img_original  
        
            
