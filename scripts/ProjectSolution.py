#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:14:34 2020
@author: elif.ayvali

BoundingBox {
    boxID        : bounding box around a classified object (contains both 2D and 3D data)
    trackID      : unique identifier for the track to which this bounding box belongs
    roi          : cv.Rect 2D region-of-interest in image coordinates
    classID      : ID based on class file provided to YOLO framework
    confidence   : classification trust
    lidarPoints  : 4D (n,4) lidar coordinates [x,y,z,r]
    lidarPixels  : 2D (n,2) image coordinates of lidar points within roi
    keypoints    : 2D keypoints within camera image
    kptMatches   : keypoint matches between previous and current frame
}
DataFrame {
    cameraImg    : color camera image
    keypoints    : 2D keypoints within camera image
    descriptors  : keypoint descriptors
    kptMatches   : keypoint matches between previous and current frame
    lidarPoints  : 4D (n,4) lidar coordinates [x,y,z,r]
    boundingBoxes: array of BoundingBox
    bbMatches    : bounding box matches between previous and current frame
          }

"""
import cv2 as cv
from recordtype import recordtype #mutable namedtuple
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from SensorProcessor import LidarProcessing,CameraProcessing
from YOLO import ObjectDetection,ObjectDetectionOpenCV
from utils import View

###########################################################################
#-----------------Define Data Structures and Paths------------------------#
###########################################################################       
#Define data structures
DataFrame=recordtype('DataFrame',['cameraImg','keypoints', 'descriptors' ,'kptMatches','lidarPoints','boundingBoxes','bbMatches'])
BoundingBox=recordtype('BoundingBox',['boxID','trackID','roi','classID', 'confidence', 'lidarPoints','lidarPixels','keypoints','kptMatches'])

#Data location
#Images
img_folderpath='../data/KITTI/2011_09_26/image_02/data/*.png'
img_filepaths=sorted(glob.glob(img_folderpath))
#Lidar
lidar_folderpath='../data/KITTI/2011_09_26/velodyne_points/data/*.bin'
lidar_filepaths=sorted(glob.glob(lidar_folderpath))
#Yolo Network
labelsPath = "../data/YOLO/coco.names"
weightsPath = "../data/YOLO/yolov3.weights"
configPath = "../data/YOLO/yolov3.cfg"
modelPath = "../data/YOLO/yolov3_model.h5"



#Loop over all data sequence
dataBuffer=[]
idx_start=0
idx_step=2
idx_end=12#len(img_filepaths)
view=View()
LP=LidarProcessing()
CP=CameraProcessing()

#YOLO
params=dict()
params["confidence"]=0.2
params["threshold"]=0.4
params["shrink"]=0.1#shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
params["framerate"]=10.0 / idx_step
#LiDAR
#Kitti
x_range,y_range,z_range,scale=(-20, 20),(-20, 20),(-2, 2),10
v_fov, h_fov,max_d = (-24.9, 2.0), (-90, 90),70
#Udacity
minZ = -1.5; maxZ = -0.9; minX = 2.0; maxX = 20.0; maxY = 2.0; minR = 0.1; # focus on ego lane


for idx in range(idx_start,idx_end,idx_step):
    ###########################################################################
    #-----------------------------Process Camera------------------------------#
    ###########################################################################   
    bBox=BoundingBox('None','None','None','None', 'None','None','None','None','None')
    frame=DataFrame('None','None', 'None' ,'None','None','None','None')

    img = cv.imread(img_filepaths[idx]) 
    # plt.figure(figsize=(16, 16))
    # plt.title('Front camera')
    # plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    # plt.show() 


    # YOLO OpenCV Implementation
    YOLO=ObjectDetectionOpenCV(params,labelsPath,weightsPath,configPath,bBox) 
    #YOLO Keras Implementation
    # YOLO=ObjectDetection(params,labelsPath,modelPath,bBox) 
    bBoxes,img_boxes=YOLO.predict(img.copy())  #returns list of bBox  
    print('Boxes found:',len(bBoxes))

    print("#1 : LOAD IMAGE INTO BUFFER done")
    print("#2 : DETECT & CLASSIFY OBJECTS done")
    
    # plt.figure(figsize=(16, 16))
    # plt.title('Yolo Object Classification')
    # plt.imshow(cv.cvtColor(img_boxes, cv.COLOR_BGR2RGB))
    # plt.show()
    
    #Compute Keypoints and Descriptprs
    img_gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp,des=CP.computeFeatureDescriptors(img_gray)
    print("#5-6 : DETECT & EXTRACT KEYPOINTS done") 
    #Associate keypoint
    


    ###########################################################################
    #------------------------------Process Lidar------------------------------#
    ###########################################################################
    velodyne_scan = np.fromfile(lidar_filepaths[idx], dtype=np.float32)
    #import lidar data
    lidar_scan=velodyne_scan.reshape((-1, 4))      
    img_lidar_topview=view.lidar_top_view(lidar_scan)
    #remove Lidar points based on distance properties
    # minZ = -1.5; maxZ = -0.9; minX = 2.0; maxX = 20.0; maxY = 2.0; minR = 0.1; # focus on ego lane
    crop_range=(minZ,maxZ,minX,maxX,maxY,minR)
    lidar_pts,lidar_color=LP.crop_lidar_points(lidar_scan,crop_range,max_d)
    #project lidar points to camera images
    lidar_px=LP.projectLidarToCam(lidar_pts) 
    #Overlay lidar on camera image
    img_lidar_fusion=view.lidar_overlay(lidar_px, lidar_color, img)       
    #-------------------Cluster Lidar----------------------#
    #Clustor lidar pixels with Boundingbox ROI    
    bBoxes_clustered=LP.cluster_lidar_with_ROI(bBoxes,lidar_px,lidar_pts)
    print( "#4 : CLUSTER LIDAR POINT CLOUD done")       
    img_clustered_lidar_topview=view.lidar_3d_objects(bBoxes_clustered)    
    img_clustered_lidar_cam_overlay=view.lidar_overlay_objects(bBoxes_clustered,img_boxes,max_d)    

    
    # plt.figure(figsize=(16, 16))
    # plt.title('Top-View Perspective of LiDAR data ')
    # plt.imshow(img_lidar_topview)
    # plt.axis('off')
    # plt.show()
    
    # plt.figure(figsize=(16, 16))
    # plt.title(' LiDAR fusion ')
    # plt.imshow(img_lidar_fusion)
    # plt.axis('off')
    # plt.show()
        
    # plt.figure(figsize=(16, 16))
    # plt.title('Clustered LiDAR Top View')
    # plt.imshow(img_clustered_lidar_topview)
    # plt.axis('off')
    # plt.show()   
    
    # plt.figure(figsize=(16, 16))
    # plt.title('Clustered LiDAR Overlay ')
    # plt.imshow(img_clustered_lidar_cam_overlay)
    # plt.axis('off')
    # plt.show()  
    
    #Save data frame
    frame.cameraImg=img_boxes.copy()
    frame.boundingBoxes=bBoxes.copy()
    frame.keypoints=kp.copy()
    frame.descriptors=des.copy()
    frame.lidarPoints=lidar_pts.copy()    
    dataBuffer.append(frame)  


    ###########################################################################
    #-------------------------Descriptor Matching-----------------------------#
    ###########################################################################       
    if len(dataBuffer)>1:    
        #grayscale images for keypoint
        last_frame=dataBuffer[-2]
        current_frame=dataBuffer[-1]   
        img_current=current_frame.cameraImg.copy()
        img_last=last_frame.cameraImg.copy()
        good_matches,good_kp_current,good_kp_last=CP.matchFeatureDescriptor(current_frame,last_frame)
        cv.drawKeypoints(img_last,good_kp_last,img_last,(0,0,255),cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS )
        cv.drawKeypoints(img_current, good_kp_current,img_current,(0,0,255),cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS )
        current_frame.kptMatches=good_matches
        num_good_matches = len(good_matches)
        # img_match=cv.drawMatches(img_current,good_kp_current,img_last,good_kp_last,good_matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.figure()
        # plt.title('Keypoint Matching')
        # plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))   
        # plt.show()        
 
   
        print("#7 : MATCH KEYPOINT DESCRIPTORS done" )   
        #Associate bounding boxes netween current and previous frames: map(last_boxID,current_boxID)      
        #store matches in the current data frame
        current_frame.bbMatches=CP.matchBoundingBoxes(good_kp_current,good_kp_last,current_frame.boundingBoxes,last_frame.boundingBoxes)  
        print('boundingBox matches:',current_frame.bbMatches)
        print("FP.1 : Match 3D Objects done")
        
        #consider boxID index
        for box in current_frame.boundingBoxes:
            color = (list(np.random.choice(range(256), size=3)))  
            color_kp =[int(color[0]), int(color[1]), int(color[2])]  
            CP.clusterKptMatchesWithROI(box,good_kp_current,good_kp_last,good_matches)
            cv.drawKeypoints(img_current, box.keypoints,img_current,color_kp,cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
        for box in last_frame.boundingBoxes:
            color = (list(np.random.choice(range(256), size=3)))  
            color_kp =[int(color[0]), int(color[1]), int(color[2])]  
            CP.clusterKptMatchesWithROI(box,good_kp_current,good_kp_last,good_matches)
            cv.drawKeypoints(img_last, box.keypoints,img_last,color_kp,cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print("FP.3 : Associate Keypoint Correspondences with Bounding Boxes done")
        
        plt.figure(figsize=(16, 16))
        plt.imshow(cv.cvtColor(img_last, cv.COLOR_BGR2RGB))   
        plt.title('Associate KeyPoints with Bounding Boxes: Last Frame')
        plt.show()        
        plt.figure(figsize=(16, 16))
        plt.title('Associate KeyPoints with Bounding Boxes: Current Frame')
        plt.imshow(cv.cvtColor(img_current, cv.COLOR_BGR2RGB)) 
        plt.show()              
        print("#8 : TRACK 3D OBJECT BOUNDING BOXES done")
        
        #Compute TTC for each object: 
        TTC=[]
        for match in current_frame.bbMatches:
            last_boxID=match[0]
            current_boxID=match[1]  
            currBox=current_frame.boundingBoxes[current_boxID]
            lastBox=last_frame.boundingBoxes[last_boxID]
            TTC_cam=CP.computeTTCcamera(currBox,params["framerate"])
            TTC_lidar=LP.computeTTCLidar(currBox,lastBox,params["framerate"])
            TTC.append((current_boxID,TTC_cam,TTC_lidar)) 
        
        img_TTC_camera=view.TTC_camera_view(current_frame,TTC)
        print("FP.4 : Compute Camera-based TTC")
      
        plt.figure(figsize=(16, 16))
        plt.title('TTC_camera')
        plt.imshow(cv.cvtColor(img_TTC_camera, cv.COLOR_BGR2RGB))
        plt.show()
              
        
    

