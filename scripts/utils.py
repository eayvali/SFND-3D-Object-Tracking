#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:33:46 2020

@author: elif.ayvali
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class View:   
            
    def lidar_overlay_cv_kitti(lidar_px, color, image):
        """ project converted velodyne points into camera image
            color is a scalar value
            reference: kitti_foundation.py , github@windowsub0406
        """
    
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)    
        for i in range(lidar_px.shape[1]):
            cv.circle(hsv_image, (np.int32(lidar_px[0,i]),np.int32(lidar_px[1,i])),3, (int(color[i]),255,255),-1)
    
        return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    def lidar_overlay_plt_kitti(lidar_px, color, image):
        """ project converted velodyne points into camera image 
            color is a scalar value   
            reference: kitti_foundation.py , github@windowsub0406
        """
        
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)    
        for i in range(lidar_px.shape[1]):
            cv.circle(hsv_image, (np.int32(lidar_px[0,i]),np.int32(lidar_px[1,i])),3, (int(color[i]),255,255),-1)    
        return cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)
    
    def lidar_overlay_plt_udacity(lidar_px, color, image):
        """ project converted velodyne points into camera image
            color is in (b,g,r) format"""    
    
        for i in range(lidar_px.shape[1]):
            image=cv.circle(image, (np.int32(lidar_px[0,i]),np.int32(lidar_px[1,i])),4, color[i],-1)    
        return cv.cvtColor(image, cv.COLOR_BGR2RGB) #to display as plt        
   
    
    def lidar_top_view_kitti(points, x_range, y_range, z_range, scale):
        ''' reference: kitti_foundation.py , github@windowsub0406 '''
        # Projecting to 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2)
        
        # extract in-range points
        x_lim = View.__in_range_points(x, x, y, z, x_range, y_range, z_range)
        y_lim = View.__in_range_points(y, x, y, z, x_range, y_range, z_range)
        dist_lim = View.__in_range_points(dist, x, y, z, x_range, y_range, z_range)
        
        # * x,y,z range are based on lidar coordinates
        x_size = int((y_range[1] - y_range[0]))
        y_size = int((x_range[1] - x_range[0]))
        
        # convert 3D lidar coordinates(vehicle coordinates) to 2D image coordinates
        # Velodyne coordinates info : http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        # scale - for high resolution
        x_img = -(y_lim * scale).astype(np.int32)
        y_img = -(x_lim * scale).astype(np.int32)
    
        # shift negative points to positive points (shift minimum value to 0)
        x_img += int(np.trunc(y_range[1] * scale))
        y_img += int(np.trunc(x_range[1] * scale))
    
        # normalize distance value & convert to depth map
        max_dist = np.sqrt((max(x_range)**2) + (max(y_range)**2))
        dist_lim = View.__normalize_depth(dist_lim, min_v=0, max_v=max_dist)
        
        # array to img
        img = np.zeros([y_size * scale + 1, x_size * scale + 1], dtype=np.uint8)
        img[y_img, x_img] = dist_lim
        return img    
    
    def __normalize_depth(val, min_v, max_v):
        """ 
        print 'normalized depth value' 
        normalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
        """
        return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)
    
    def __in_range_points(points, x, y, z, x_range, y_range, z_range):
        """ extract in-range points """
        return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], \
                                             y < y_range[1], z > z_range[0], z < z_range[1]))]

    def lidar_top_view_udacity(points):
        #(width,height) (col,row) (x,y) image frame
        worldSize=(10,20)     #(width,height) of sensor field 
        imageSize=(1000,2000) #corresponding top view image in pixel
        #points[x,y,z,r]:valodyne frame
        topviewImg=np.zeros((imageSize[0],imageSize[1],3), np.uint8)
        # Projecting to 2D
        x = (-points[:, 0]*imageSize[1]/worldSize[1])+imageSize[1]
        y = (-points[:, 1]*imageSize[0]/worldSize[0])+imageSize[0]/2
        z = points[:, 2]
        
        #only plot above the road
        minz=-1.40
        for i in range(points.shape[0]):
            if points[i, 2]>minz:
                val= points[i, 0]  #distance in driving direction
                maxVal=worldSize[1]
                red=min(255,int(255*abs((val-maxVal)/maxVal)))
                green=min(255,int(255*(1-abs((val-maxVal)/maxVal))))
                # print( (np.int32(x[i]),np.int32(y[i])))
                topviewImg=cv.circle(topviewImg, (np.int32(x[i]),np.int32(y[i])),4, (0,green,red),-1)    
        #plot distance markers
        lineSpacing=2
        nMarkers=int(np.floor(worldSize[1]/lineSpacing))
        for i in range(nMarkers):
            x=np.int32((-(i*lineSpacing)*imageSize[1]/worldSize[1])+imageSize[1])
            topviewImg=cv.line(topviewImg, (x,0),(x,imageSize[0]),(255,0,0),2) 
            cv.putText(topviewImg, '|2 meters|',(25,35),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        return cv.cvtColor(topviewImg, cv.COLOR_BGR2RGB) #to display as plt

class Plot:
    
    def __init__(self):
        
        
        img_init=np.zeros((300,1000), np.uint8)
        self.fig=plt.figure()
        gs=self.fig.add_gridspec(3,2)
        
        axs1=self.fig.add_subplot(gs[:2,0])
        self.plot1=axs1.imshow(img_init,vmin=0,vmax=255, aspect='auto')#defining vmin,vmax is necessary
        axs1.set_title('Top-View Perspective of LiDAR data (Kitti)', fontsize=10)
        axs1.axis('off')
        
        
        axs2=self.fig.add_subplot(gs[0,1])
        self.plot2=axs2.imshow(img_init,vmin=0,vmax=255, aspect='auto')
        axs2.set_title('Top-View Perspective of LiDAR data (Udacity)', fontsize=10)
        axs2.axis('off')
        
        axs3=self.fig.add_subplot(gs[1,1])
        self.plot3=axs3.imshow(img_init,vmin=0,vmax=255, aspect='auto')
        axs3.set_title('LiDAR fusion (Udacity)', fontsize=10)
        axs3.axis('off')        
        
        axs4=self.fig.add_subplot(gs[2,0])
        self.plot4=axs4.imshow(img_init,vmin=0,vmax=255, aspect='auto')
        axs4.set_title('Good Keypoints Last Frame', fontsize=10)
        axs4.axis('off')

        axs5=self.fig.add_subplot(gs[2,1])
        self.plot5=axs5.imshow(img_init,vmin=0,vmax=255, aspect='auto')
        axs5.set_title('Good Keypoints Current Frame', fontsize=10)
        axs5.axis('off')
        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
                
        
    def get_figure(self):
        return self.fig
   
    
    def update(self,img_last,img_current,img_lidar_topview ,img_lidar_fusion,img_lidar_topview_udacity,img_lidar_fusion_udacity):
        self.plot1.set_data(img_lidar_topview)
        self.plot2.set_data(img_lidar_topview_udacity)
        self.plot3.set_data(img_lidar_fusion_udacity)
        self.plot4.set_data(cv.cvtColor(img_last, cv.COLOR_BGR2RGB))
        self.plot5.set_data(cv.cvtColor(img_current, cv.COLOR_BGR2RGB))
        
            
    
    