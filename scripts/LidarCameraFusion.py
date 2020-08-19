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
import copy
import numpy as np

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
idx_step=3
idx_end=6#len(img_filepaths)
view=View()
LP=LidarProcessing()
CP=CameraProcessing()

#YOLO
params=dict()
params["confidence"]=0.2
params["threshold"]=0.4
params["shrink"]=0.1


#LiDAR
#Kitti
x_range,y_range,z_range,scale=(-20, 20),(-20, 20),(-2, 2),10
v_fov, h_fov,max_d = (-24.9, 2.0), (-90, 90),70
#Udacity
# minZ = -1.5; maxZ = -0.9; minX = 2.0; maxX = 20.0; maxY = 2.0; minR = 0.1; # focus on ego lane
minZ = -1.4; maxZ = 1e2; minX = 0.0; maxX = 25.0; maxY = 6.0; minR = 0.01;max_d=20;


for idx in range(idx_start,idx_end,idx_step):
    bBox=BoundingBox('None','None','None','None', 'None','None','None','None','None')
    frame=DataFrame('None','None', 'None' ,'None','None','None','None')
    ###########################################################################
    #-----------------------------Process Camera------------------------------#
    ###########################################################################   
    img = cv.imread(img_filepaths[idx]) 
    print('Processing ',img_filepaths[idx])
    # plt.figure(figsize=(16, 16))
    # plt.title('Front camera')
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # plt.show() 


    # YOLO OpenCV Implementation
    YOLO=ObjectDetectionOpenCV(params,labelsPath,weightsPath,configPath,bBox) 
    #YOLO Keras Implementation
    # YOLO=ObjectDetection(params,labelsPath,modelPath,bBox) 
    bBoxes,img_boxes=YOLO.predict(img.copy())  #returns list of bBox   
    print("#1 : LOAD IMAGE INTO BUFFER done")
    print("#2 : DETECT & CLASSIFY OBJECTS done")
    
    # plt.figure(figsize=(16, 16))
    # plt.title('Yolo Object Classification')
    # plt.imshow(cv.cvtColor(img_boxes, cv.COLOR_BGR2RGB))
    # plt.show()

    #Compute Keypoints and Descriptprs
    img_gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp,des=CP.computeFeatureDescriptors(img_gray)  

    
    
    ###########################################################################
    #------------------------------Process Lidar------------------------------#
    ###########################################################################
    velodyne_scan = np.fromfile(lidar_filepaths[idx], dtype=np.float32)
    #import lidar data
    lidar_scan=velodyne_scan.reshape((-1, 4))    
    #-----------------KITTI-------------------#
    #display topview
    img_lidar_topview_kitti=view.lidar_top_view_kitti(lidar_scan, x_range, y_range, z_range, scale)#lidar_pts:nx4
    #remove lidar points outside of the field of view and compute color range based on max distance
    lidar_pts_kitti,lidar_color_kitti=LP.velo_points_filter_kitti(lidar_scan,v_fov,h_fov,max_d)
    print("#3 : CROP LIDAR POINTS done")
    #project lidar points to camera images
    lidar_px_kitti=LP.projectLidarToCam(lidar_pts_kitti) 
    #Overlay lidar on camera image
    img_lidar_fusion_kitti=view.lidar_overlay_kitti(lidar_px_kitti, lidar_color_kitti, img.copy())      

 
    #-----------------Udacity-------------------#
    img_lidar_topview=view.lidar_top_view(lidar_scan)
    #remove Lidar points based on distance properties
    # minZ = -1.5; maxZ = -0.9; minX = 2.0; maxX = 20.0; maxY = 2.0; minR = 0.1; # focus on ego lane
    crop_range=(minZ,maxZ,minX,maxX,maxY,minR)
    lidar_pts,lidar_color=LP.crop_lidar_points(lidar_scan,crop_range,max_d)
    #project lidar points to camera images
    lidar_px=LP.projectLidarToCam(lidar_pts) 
    #Overlay lidar on camera image
    img_lidar_fusion=view.lidar_overlay(lidar_px, lidar_color, img.copy())       
    

    #-------------------Cluster Lidar----------------------#
    #Clustor lidar pixels with Boundingbox ROI    
    bBoxes_clustered=LP.cluster_lidar_with_ROI(bBoxes,lidar_px,lidar_pts)
    print( "#4 : CLUSTER LIDAR POINT CLOUD done")       
    img_clustered_lidar_topview=view.lidar_3d_objects(bBoxes_clustered)    
    img_clustered_lidar_cam_overlay=view.lidar_overlay_objects(bBoxes_clustered,img_boxes,max_d)    

    #Save data frame
    frame.cameraImg=img.copy()
    frame.boundingBoxes=bBoxes.copy()
    frame.keypoints=kp.copy()
    frame.descriptors=des.copy()
    frame.lidarPoints=lidar_pts.copy()    
    dataBuffer.append(frame)  

    # plt.figure(figsize=(16, 16))
    # plt.title('Top-View Perspective of LiDAR data (Kitti)')
    # plt.imshow(img_lidar_topview_kitti)
    # plt.show()
    
    # plt.figure(figsize=(16, 16))
    # plt.title('Lidar Fusion (Kitti)')
    # plt.imshow(img_lidar_fusion_kitti)
    # plt.show()
    
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
    
    plt.figure(figsize=(16, 16))
    plt.title('Clustered LiDAR Overlay ')
    plt.imshow(img_clustered_lidar_cam_overlay)
    plt.axis('off')
    plt.show()   



    ###########################################################################
    #-------------------------Descriptor Matching-----------------------------#
    ###########################################################################       
    if len(dataBuffer)>1:        
        last_frame=dataBuffer[-2]
        current_frame=dataBuffer[-1]
        good_matches,good_kp_last,good_kp_current=CP.matchFeatureDescriptor(current_frame,last_frame)        
        num_good_matches = len(good_matches)
        print("#7 : MATCH KEYPOINT DESCRIPTORS done" )  
        img_current=current_frame.cameraImg.copy()
        img_last=last_frame.cameraImg.copy()
        cv.drawKeypoints(img_last, good_kp_last,img_last,(0, 0, 255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.drawKeypoints(img_current, good_kp_current,img_current,(0, 0, 255),cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
        plt.figure(figsize=(16, 16))
        plt.title('last_frame')
        plt.imshow(cv.cvtColor(img_last, cv.COLOR_BGR2RGB))
        plt.show()        
        plt.figure(figsize=(16, 16))
        plt.title('current_frame')
        plt.imshow(cv.cvtColor(img_current, cv.COLOR_BGR2RGB))
        plt.show()
        
        
        
        
    

