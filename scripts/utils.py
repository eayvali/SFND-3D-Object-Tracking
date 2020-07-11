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
    
    def __init__(self):
        #(width,height) (col,row) (x,y) image frame
        self.worldSize=(10,20)     #(width,height) of sensor field 
        self.imageSize=(1000,2000) #corresponding top view image in pixel
            
    def lidar_overlay_cv_kitti(self, lidar_px, color, image):
        """ project converted velodyne points into camera image
            color is a scalar value
            reference: kitti_foundation.py , github@windowsub0406
        """
    
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)    
        for i in range(lidar_px.shape[1]):
            cv.circle(hsv_image, (np.int32(lidar_px[0,i]),np.int32(lidar_px[1,i])),3, (int(color[i]),255,255),-1)
    
        return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    def lidar_overlay_kitti(self, lidar_px, color, image):
        """ project converted velodyne points into camera image 
            color is a scalar value   
            reference: kitti_foundation.py , github@windowsub0406
        """
        
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)    
        for i in range(lidar_px.shape[1]):
            cv.circle(hsv_image, (np.int32(lidar_px[0,i]),np.int32(lidar_px[1,i])),3, (int(color[i]),255,255),-1)    
        return cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)
    
    def lidar_overlay(self, lidar_px, color, image):
        """ project converted velodyne points into camera image
            color is in (b,g,r) format"""    
    
        for i in range(lidar_px.shape[1]):
            image=cv.circle(image, (np.int32(lidar_px[0,i]),np.int32(lidar_px[1,i])),4, color[i],-1)    
        return cv.cvtColor(image, cv.COLOR_BGR2RGB) #to display as plt        
   
       
    def lidar_top_view_kitti(self, points, x_range, y_range, z_range, scale):
        ''' reference: kitti_foundation.py , github@windowsub0406 '''
        # Projecting to 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x ** 2 + y ** 2)
        
        # extract in-range points
        x_lim = self.__in_range_points(x, x, y, z, x_range, y_range, z_range)
        y_lim = self.__in_range_points(y, x, y, z, x_range, y_range, z_range)
        dist_lim = self.__in_range_points(dist, x, y, z, x_range, y_range, z_range)
        
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
        dist_lim = self.__normalize_depth(dist_lim, min_v=0, max_v=max_dist)
        
        # array to img
        img = np.zeros([y_size * scale + 1, x_size * scale + 1], dtype=np.uint8)
        img[y_img, x_img] = dist_lim
        return img    
    
    def __normalize_depth(self, val, min_v, max_v):
        """ 
        print 'normalized depth value' 
        normalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
        """
        return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)
    
    def __in_range_points(self, points, x, y, z, x_range, y_range, z_range):
        """ extract in-range points """
        return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], \
                                             y < y_range[1], z > z_range[0], z < z_range[1]))]

    def lidar_top_view(self, points):
        #points[x,y,z,r]:valodyne frame
        topviewImg=np.zeros((self.imageSize[0],self.imageSize[1],3), np.uint8)
        # Projecting to 2D
        x = (-points[:, 0]*self.imageSize[1]/self.worldSize[1])+self.imageSize[1]
        y = (-points[:, 1]*self.imageSize[0]/self.worldSize[0])+self.imageSize[0]/2
        # z = points[:, 2]#not used
        
        #only plot above the road
        minz=-1.40
        for i in range(points.shape[0]):
            if points[i, 2]>minz:
                val= points[i, 0]  #distance in driving direction
                maxVal=self.worldSize[1]
                red=min(255,int(255*abs((val-maxVal)/maxVal)))
                green=min(255,int(255*(1-abs((val-maxVal)/maxVal))))
                # print( (np.int32(x[i]),np.int32(y[i])))
                topviewImg=cv.circle(topviewImg, (np.int32(x[i]),np.int32(y[i])),4, (0,green,red),-1)    
        #plot distance markers
        lineSpacing=2
        nMarkers=int(np.floor(self.worldSize[1]/lineSpacing))
        for i in range(nMarkers):
            x=np.int32((-(i*lineSpacing)*self.imageSize[1]/self.worldSize[1])+self.imageSize[1])
            topviewImg=cv.line(topviewImg, (x,0),(x,self.imageSize[0]),(255,0,0),3) 
            cv.putText(topviewImg, '|2 meters|',(35,50),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),4)
        return cv.cvtColor(topviewImg, cv.COLOR_BGR2RGB) #to display as plt

    def lidar_3d_objects(self, bBoxes):
        topviewImg=np.zeros((self.imageSize[0],self.imageSize[1],3), np.uint8)
        color_options= np.random.choice(range(256), size=(len(bBoxes),3))
        for idx,box in enumerate(bBoxes):
            (top,left,bottom,right)=(1e4,1e4,0.0,0.0)
            color_box=(int(color_options[idx,0]),int(color_options[idx,1]),int(color_options[idx,2]))
            #convert box to top view
            for (px,py,_,_) in box.lidarPoints:#(n,4)
                # Projecting to 2D top image
                x = int((-px*self.imageSize[1]/self.worldSize[1])+self.imageSize[1])
                y = int((-py*self.imageSize[0]/self.worldSize[0])+self.imageSize[0]/2)         
                #find enclosing rectangle
                top=min(top,y)
                left=min(left,x)
                bottom=max(bottom,y)
                right=max(right,x)
                topviewImg=cv.circle(topviewImg, (x,y),4, color_box, -1) 
            topviewImg=cv.rectangle(topviewImg, (int(left),int(top)), (int(right),int(bottom)),color_box,3) 
        #plot distance markers
        lineSpacing=2
        nMarkers=int(np.floor(self.worldSize[1]/lineSpacing))
        for i in range(nMarkers):
            x=np.int32((-(i*lineSpacing)*self.imageSize[1]/self.worldSize[1])+self.imageSize[1])
            topviewImg=cv.line(topviewImg, (x,0),(x,self.imageSize[0]),(255,0,0),3) 
            cv.putText(topviewImg, '|2 meters|',(35,50),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),4)
        return cv.cvtColor(topviewImg, cv.COLOR_BGR2RGB) #to display as plt

    def lidar_overlay_objects(self,bBoxes,img,max_d): 
        for idx,box in enumerate(bBoxes):
            for idx in range(box.lidarPixels.shape[1]):#(2,n)
                #compute color of each point based on distance
                val= box.lidarPoints[idx, 0]  #distance in driving direction
                red=min(255,int(255*abs((val-max_d)/max_d)))
                green=min(255,int(255*(1-abs((val-max_d)/max_d))))        
                color_px= (0,green,red) #opencv
                img=cv.circle(img, (np.int32(box.lidarPixels[0,idx]),np.int32(box.lidarPixels[1,idx])),4, color_px,-1)             
        return cv.cvtColor(img, cv.COLOR_BGR2RGB) #to display as plt       


        
class Plot:
    
    def __init__(self):        
        
        img_init=np.zeros((300,1000), np.uint8)
        self.fig=plt.figure()
        gs=self.fig.add_gridspec(3,2)
        
        axs1=self.fig.add_subplot(gs[:2,0])
        self.plot1=axs1.imshow(img_init,vmin=0,vmax=255, aspect='auto')#defining vmin,vmax is necessary
        axs1.set_title('Top-View of LiDAR data (KITTI)', fontsize=10)
        axs1.axis('off')
        
        
        axs2=self.fig.add_subplot(gs[0,1])
        self.plot2=axs2.imshow(img_init,vmin=0,vmax=255)
        axs2.set_title('Top-View of filtered LiDAR data', fontsize=10)
        axs2.axis('off')
        
        axs3=self.fig.add_subplot(gs[1,1])
        self.plot3=axs3.imshow(img_init,vmin=0,vmax=25)
        axs3.set_title('LiDAR fusion', fontsize=10)
        axs3.axis('off')        
        
        axs4=self.fig.add_subplot(gs[2,0])
        self.plot4=axs4.imshow(img_init,vmin=0,vmax=255)
        axs4.set_title('Good Keypoints Last Frame', fontsize=10)
        axs4.axis('off')

        axs5=self.fig.add_subplot(gs[2,1])
        self.plot5=axs5.imshow(img_init,vmin=0,vmax=255)
        axs5.set_title('Good Keypoints Current Frame', fontsize=10)
        axs5.axis('off')
        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()

                
        
    def get_figure(self):
        return self.fig
   
    
    def update(self,img_last,img_current,img_lidar_topview ,img_lidar_fusion,img_lidar_topview_kitti):
        self.plot1.set_data(img_lidar_topview_kitti)
        self.plot2.set_data(img_lidar_topview)
        self.plot3.set_data(img_lidar_fusion)
        self.plot4.set_data(cv.cvtColor(img_last, cv.COLOR_BGR2RGB))
        self.plot5.set_data(cv.cvtColor(img_current, cv.COLOR_BGR2RGB))
        
            
    
    