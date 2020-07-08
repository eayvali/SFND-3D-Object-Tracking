#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:33:46 2020

@author: elif.ayvali
"""
import numpy as np
import cv2 as cv
from recordtype import recordtype #mutable namedtuple


class ObjectDetectionOpenCV:
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
        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
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
                cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv.putText(image, text, (x+10, y - 5), cv.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
                #['boxID','trackID','roi','classID', 'confidence']
                self.BoundingBox.boxID=len(bBoxes)
                self.BoundingBox.classID=classIDs[i]
                self.BoundingBox.confidence=confidences[i]
                self.BoundingBox.roi=boxes[i]
                bBoxes.append(self.BoundingBox)
        return bBoxes,image        

class CameraProcessing:
    def computeFeatureDescriptors(img1,img2):
        #Initiate fast keypoint detector, brief descriptor
        star = cv.xfeatures2d.StarDetector_create()
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        brief=cv.xfeatures2d.BriefDescriptorExtractor_create()
        kp1=star.detect(img1,None)
        kp2=star.detect(img2,None)
        kp1,des1= brief.compute(img1, kp1) 
        kp2,des2= brief.compute(img2, kp2)         
        return kp1,kp2,des1,des2
    
    def matchFeatureDescriptor(kp1,kp2,des1,des2):
       
        #Define Flann descriptor matcher   
        FLANN_INDEX_LSH = 6 
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6, key_size=12,
                                multi_probe_level=1)
        search_params = dict() # dict(checks=50)  
        descriptor_matcher=cv.FlannBasedMatcher(index_params, search_params)    
        # Find the 2 best matches for each descriptor.
        matches = descriptor_matcher.knnMatch(des1,des2, 2)
        # Filter the matches based on the distance ratio test.
        good_matches = [match[0] for match in matches if len(match) > 1 and \
                    match[0].distance < 0.6 * match[1].distance]
            
        # Select the good keypoints 
        good_kp1 = [kp1[match.queryIdx] for match in good_matches] #current frame
        good_kp2 = [kp2[match.trainIdx] for match in good_matches] #matching frame        
        return good_matches,good_kp1,good_kp2
        
    
    

class LidarProcessing:
    
    def projectLidarToCam(lidar_pts): 
        
        #lidar_pts: 4xn, column:[x,y,z,1]
        #velodyne to left camera extrinsic
        T_velo_to_cam=np.array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03 ],
                                [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02 ],
                                [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01 ],
                                [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00 ]])
        #cam intrinsic
        R_rect=np.array([[ 9.999239e-01, 9.837760e-03, -7.445048e-03, 0.000000e+00 ],
                         [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.000000e+00 ],
                         [ 7.402527e-03, 4.351614e-03,  9.999631e-01, 0.000000e+00 ],
                         [ 0.000000e+00, 0.000000e+00,  0.000000e+00, 1.000000e+00 ]])
        #P_rect_01 in calib_cam_to_cam.txt: left camera
        P_rect=np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                         [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                         [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])
        
        #Project lidar to left camera image: Eq(7) in IJRR paper   
        lidar_cam= np.matmul(np.linalg.multi_dot([P_rect,R_rect,T_velo_to_cam]), lidar_pts.T)  #lidar_pts.T: 4xn
        #scale adjustment lidar_cam[:1,i]/lidar_cam[2,i]
        p_lidar=lidar_cam/lidar_cam[2,:]
        return np.delete(p_lidar, 2, axis=0) #only keep x,y

    def velo_points_filter_kitti(points, v_fov, h_fov, max_d):
        """ extract points corresponding to FOV setting
            output: nx4 filtered lidar homogeneous coordinates
            reference: kitti_foundation.py , github@windowsub0406"""
        
        # Projecting to 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
        if h_fov[0] < -90:
            h_fov = (-90,) + h_fov[1:]
        if h_fov[1] > 90:
            h_fov = h_fov[:1] + (90,)
        
        x_lim = LidarProcessing.__fov_setting(x, x, y, z, dist, h_fov, v_fov)[:,None]
        y_lim = LidarProcessing.__fov_setting(y, x, y, z, dist, h_fov, v_fov)[:,None]
        z_lim = LidarProcessing.__fov_setting(z, x, y, z, dist, h_fov, v_fov)[:,None]   
        # Stack arrays in sequence horizontally
        xyz_ = np.hstack((x_lim, y_lim, z_lim))
         # transform to homogeneous coordinate  
        lidar_pts=np.insert(xyz_, 3, values=1, axis=1)  
        # need dist info for points color
        dist_lim = LidarProcessing.__fov_setting(dist, x, y, z, dist, h_fov, v_fov)
        color = LidarProcessing.__depth_color(dist_lim, 0, max_d)    
        return lidar_pts, color    
    
    def crop_lidar_points_udacity(lidar_pts,crop_range,max_d):
        #input: nx4 [x,y,z,r], output: mx4
        minZ,maxZ,minX,maxX,maxY,minR=crop_range
        idx=(lidar_pts[:,0]>=minX) & (lidar_pts[:,0]<=maxX) & (lidar_pts[:,2]>=minZ) \
            & (lidar_pts[:,2]<=maxZ)  & (np.abs(lidar_pts[:,1])<=maxY) & (lidar_pts[:,3]>=minR) 
        cropped_lidar_pts=lidar_pts[idx,:]
        
        #compute color of each point based on distance
        #maxVal=np.max(lidar_pts[:,0]) # max along the driving direction X
        maxVal=max_d
        lidar_bgr_color=[]
        for i in range(cropped_lidar_pts.shape[0]):
            val= cropped_lidar_pts[i, 0]  #distance in driving direction
            red=min(255,int(255*abs((val-maxVal)/maxVal)))
            green=min(255,int(255*(1-abs((val-maxVal)/maxVal))))        
            lidar_bgr_color.append( (0,green,red) ) #opencv
        return cropped_lidar_pts, lidar_bgr_color    
            
    def __depth_color(val, min_d=0, max_d=120):
        """ 
        print Color(HSV's H value) corresponding to distance(m) 
        close distance = red , far distance = blue
        """
        np.clip(val, 0, max_d, out=val) # max distance is 120m 
        return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8) 
    
    def __in_h_range_points(points, m, n, fov):
        """ extract horizontal in-range points """
        return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), \
                              np.arctan2(n,m) < (-fov[0] * np.pi / 180))
    
    def __in_v_range_points(points, m, n, fov):
        """ extract vertical in-range points """
        return np.logical_and(np.arctan2(n,m) < (fov[1] * np.pi / 180), \
                              np.arctan2(n,m) > (fov[0] * np.pi / 180))
    
    def __fov_setting(points, x, y, z, dist, h_fov, v_fov):
        """ filter points based on h,v FOV 
             reference: kitti_foundation.py , github@windowsub0406
        """
        
        if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
            return points
        
        if h_fov[1] == 180 and h_fov[0] == -180:
            return points[LidarProcessing.__in_v_range_points(points, dist, z, v_fov)]
        elif v_fov[1] == 2.0 and v_fov[0] == -24.9:        
            return points[LidarProcessing.__in_h_range_points(points, x, y, h_fov)]
        else:
            h_points = LidarProcessing.__in_h_range_points(points, x, y, h_fov)
            v_points = LidarProcessing.__in_v_range_points(points, dist, z, v_fov)
            return points[np.logical_and(h_points, v_points)]
      
            
            
