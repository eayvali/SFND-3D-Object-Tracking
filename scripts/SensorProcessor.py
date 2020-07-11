#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:33:46 2020

@author: elif.ayvali
"""
import numpy as np
import cv2 as cv
    
    

class CameraProcessing:
    def __init__(self):
        pass
            
    def computeFeatureDescriptors(self,img1,img2):
        #Initiate fast keypoint detector, brief descriptor
        star = cv.xfeatures2d.StarDetector_create()
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        brief=cv.xfeatures2d.BriefDescriptorExtractor_create()
        kp1=star.detect(img1,None)
        kp2=star.detect(img2,None)
        self.kp1,self.des1= brief.compute(img1, kp1) 
        self.kp2,self.des2= brief.compute(img2, kp2)         
    
    def matchFeatureDescriptor(self):
       
        #Define Flann descriptor matcher   
        FLANN_INDEX_LSH = 6 
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6, key_size=12,
                                multi_probe_level=1)
        search_params = dict() # dict(checks=50)  
        descriptor_matcher=cv.FlannBasedMatcher(index_params, search_params)    
        # Find the 2 best matches for each descriptor.
        matches = descriptor_matcher.knnMatch(self.des1,self.des2, 2)
        # Filter the matches based on the distance ratio test.
        good_matches = [match[0] for match in matches if len(match) > 1 and \
                    match[0].distance < 0.6 * match[1].distance]
            
        # Select the good keypoints 
        good_kp1 = [self.kp1[match.queryIdx] for match in good_matches] #current frame
        good_kp2 = [self.kp2[match.trainIdx] for match in good_matches] #matching frame        
        return good_matches,good_kp1,good_kp2
    
        
class LidarProcessing:
    
    def __init__(self):
        
         #velodyne to left camera extrinsic
        self.T_velo_to_cam=np.array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03 ],
                                [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02 ],
                                [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01 ],
                                [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00 ]])
        #cam intrinsic
        self.R_rect=np.array([[ 9.999239e-01, 9.837760e-03, -7.445048e-03, 0.000000e+00 ],
                         [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.000000e+00 ],
                         [ 7.402527e-03, 4.351614e-03,  9.999631e-01, 0.000000e+00 ],
                         [ 0.000000e+00, 0.000000e+00,  0.000000e+00, 1.000000e+00 ]])
        #P_rect_01 in calib_cam_to_cam.txt: left camera
        self.P_rect=np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                         [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                         [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])
               
    
    def projectLidarToCam(self,lidar_pts): 
        
        #lidar_pts: nx4, column:[x,y,z,1]
        #Project lidar to left camera image: Eq(7) in IJRR paper   
        lidar_cam= np.matmul(np.linalg.multi_dot([self.P_rect,self.R_rect,self.T_velo_to_cam]), lidar_pts.T)  #lidar_pts.T: 4xn
        #scale adjustment lidar_cam[:1,i]/lidar_cam[2,i]
        p_lidar=lidar_cam/lidar_cam[2,:] #pixel coordinates
        return np.delete(p_lidar, 2, axis=0) #only keep x,y #2xn

    def cluster_lidar_with_ROI(self, bBoxes,lidar_px,lidar_pts):  
                #lidar_px: (2,n)
        #lidar_pts=(n,4)
        #bBoxes is a list,it will be modified here
        select_bool=np.ones(lidar_px.shape[1],dtype=bool) #1: can be selected        
        for box in bBoxes:
            inside_box,select_bool=self.__box_contains(box.roi,np.copy(lidar_px), select_bool)
            lidar_px_in_box=np.copy(lidar_px[:,inside_box])
            lidar_pts_in_box=np.copy(lidar_pts[inside_box,:])            
            box.lidarPixels=lidar_px_in_box
            box.lidarPoints=lidar_pts_in_box
        return bBoxes
            
    def __box_contains(self,roi, lidar_px,select_bool):
        '''roi=:[x,y,w,h]
           px_inside_roi = lidar_px[inside_box]
           px_outside_roi = points[~inside_box]'''
        bound_x = np.logical_and(lidar_px[0, :] > roi[0], lidar_px[0 ,:] < roi[0]+roi[2] )
        bound_y = np.logical_and(lidar_px[1, :] > roi[1], lidar_px[1 ,:] <  roi[1]+roi[3])    
        inside_box = np.logical_and(np.logical_and(bound_x, bound_y) , select_bool)
        #If inside and not selected
        select_bool[inside_box] = select_bool[inside_box] *0  
        return inside_box,select_bool

    def velo_points_filter_kitti(self,points, v_fov, h_fov, max_d):
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
        
        x_lim = self.__fov_setting(x, x, y, z, dist, h_fov, v_fov)[:,None]
        y_lim = self.__fov_setting(y, x, y, z, dist, h_fov, v_fov)[:,None]
        z_lim = self.__fov_setting(z, x, y, z, dist, h_fov, v_fov)[:,None]   
        # Stack arrays in sequence horizontally
        xyz_ = np.hstack((x_lim, y_lim, z_lim))
         # transform to homogeneous coordinate  
        lidar_pts=np.insert(xyz_, 3, values=1, axis=1)  
        # need dist info for points color
        dist_lim = self.__fov_setting(dist, x, y, z, dist, h_fov, v_fov)
        color = self.__depth_color(dist_lim, 0, max_d)    
        return lidar_pts, color    
    
    def crop_lidar_points(self,lidar_pts,crop_range,max_d):
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
    

    def __depth_color(self,val, min_d=0, max_d=120):
        """ 
        print Color(HSV's H value) corresponding to distance(m) 
        close distance = red , far distance = blue
        """
        np.clip(val, 0, max_d, out=val) # max distance is 120m 
        return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8) 
    
    def __in_h_range_points(self,points, m, n, fov):
        """ extract horizontal in-range points """
        return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), \
                              np.arctan2(n,m) < (-fov[0] * np.pi / 180))
    
    def __in_v_range_points(self,points, m, n, fov):
        """ extract vertical in-range points """
        return np.logical_and(np.arctan2(n,m) < (fov[1] * np.pi / 180), \
                              np.arctan2(n,m) > (fov[0] * np.pi / 180))
    
    def __fov_setting(self,points, x, y, z, dist, h_fov, v_fov):
        """ filter points based on h,v FOV 
             reference: kitti_foundation.py , github@windowsub0406
        """
        
        if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
            return points
        
        if h_fov[1] == 180 and h_fov[0] == -180:
            return points[self.__in_v_range_points(points, dist, z, v_fov)]
        elif v_fov[1] == 2.0 and v_fov[0] == -24.9:        
            return points[self.__in_h_range_points(points, x, y, h_fov)]
        else:
            h_points = self.__in_h_range_points(points, x, y, h_fov)
            v_points = self.__in_v_range_points(points, dist, z, v_fov)
            return points[np.logical_and(h_points, v_points)]
      


    