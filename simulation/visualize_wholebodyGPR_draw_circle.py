#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('.')
import numpy as np
import math
import time
import seaborn as sns
import random
import rospy

from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from jacobi.ur5e_robot_Jacob_tool_length0145 import *
from jacobi.ur5e_car_kinematics_class_2 import UR5e_car_kinematics
import matplotlib.pyplot as plt
from trajectory_msgs.msg import *
from control_msgs.msg import *
from geometry_msgs.msg import Twist,Point
from gazebo_msgs.msg import LinkStates
from nav_msgs.msg import Odometry
import ikfast.ikfastpy as ikfastpy
from scipy import signal
from visualization_msgs.msg import Marker  


ur5_kin = ikfastpy.PyKinematics()
ur5e_car = UR5e_car_kinematics()
n_joints = ur5_kin.getDOF()
ur_sub_record = []
ur_end_record = []
car_sub_record = []
car_coor_record=[]
ur_coor_record=[]
wb_record = []
#UR state subsciber callback function
#FK to end pose 
Stop_flag = False
def ur_sub_callback(state):
    global Stop_flag
    ur_sub_record.append(list(state.actual.positions))
    ee_pose = ur5_kin.forward(ur_sub_record[-1])
    ee_pose = np.asarray(ee_pose).reshape(3,4)

    R = ee_pose[0:3,0:3]
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    if Stop_flag == True:
        sys.exit()
    ur_end_record.append([ee_pose[0,3],ee_pose[1,3],ee_pose[2,3],x,y,z])


#car state subscriber callback function
def car_sub_callback(odom):
    global Stop_flag

    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    if Stop_flag == True:
        sys.exit()
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])

#read the demonstration data
def readTxt():
    data = np.loadtxt('data/car_xytheta_dot_record9.txt')
    xy_path = data[:,0:2]
    xy_dot_path = np.concatenate((data[:,3].reshape(-1,1)*np.cos(data[:,2].reshape(-1,1)),data[:,3].reshape(-1,1)*np.sin(data[:,2]).reshape(-1,1)),axis=1)
    xy_theta_path = data[:,0:3]
    xy_theta_dot_path = np.concatenate((xy_dot_path,data[:,5].reshape(-1,1)),axis=1)

    return xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path


#窗口均值滤波
def mean_filter(data,windows_size=4,is_but=False):
    rows, cols = data.shape
    filtedData = np.zeros((rows,cols))
    windows_size = 4
    if is_but:
        b, a = signal.butter(8, 0.1, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
        for i in range(cols):
            data[:,i] = signal.filtfilt(b, a, data[:,i])  #data为要过滤的信号
    
    windows = np.zeros((windows_size,cols))
    for i in range(windows_size):
        windows[i,:] = data[0,:]
    
    for i in range(rows):
        windows[i%windows_size,:] = data[i,:]
        filtedData[i,:] = np.sum(windows,axis=0)/windows_size
    return filtedData

#down sample a numpy array with times
def down_sample(data,times):

    rows, cols = data.shape
    down_data = np.zeros((int(rows//times),cols))
    for i in range(int(rows//times)):
        down_data[i,:] = data[i*times,:]

    return down_data
class MyGPR():
    def __init__(self,data_in,data_out,normalize = False,n_restarts_optimizer=5):
        self.data_in = data_in
        self.data_out = data_out
        
        #kernel1 =  10*RBF(length_scale=1e-3, length_scale_bounds=(1e-15, 1e2))+0.000001*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
        #self.gpr1 = GaussianProcessRegressor(kernel=kernel1,random_state=0,n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize)
        kernel1 = 1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+0.1*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
        self.gpr1 = GaussianProcessRegressor(kernel=kernel1,random_state=0,n_restarts_optimizer=5,normalize_y=False)
        print('fitting')
        self.gpr1.fit(data_in,data_out)
        print('fitted')
    def get_diff_varian_matrix(self,xmin,xmax,ymin,ymax,demonstrate_point,step=400):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.step = step
        self.demonstrate_point = demonstrate_point


        x_step = (xmax - xmin)/step
        y_step = (ymax - ymin)/step
        xx = np.arange(xmin,xmax,x_step)
        yy = np.arange(ymin,ymax,y_step)
        xxx,yyy = np.meshgrid(xx,yy)
        predict_point = np.concatenate((xxx.reshape(-1,1),yyy.reshape(-1,1)),1)
        result,std = self.gpr1.predict(predict_point,return_std=True)


        self.diff_std = np.zeros((step,step,2))
        std_matrix = std[:].reshape(step,step)
        

        for i in range(step-1):
            for j in range(step-1):
                self.diff_std[i,j,0] = (std_matrix[i,j+1] - std_matrix[i,j])
                self.diff_std[i,j,1] = (std_matrix[i+1,j] - std_matrix[i,j])
        self.diff_std[:,:,0] /= x_step
        self.diff_std[:,:,1] /= y_step

        return self.diff_std
    
    #given a point return the next desired step
    def get_next_step(self,predict_point):
        result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)
        distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        if distance > 2:
            distance = 5.4
        else:
            distance *= 2.7
        return np.array((result[0,0] - self.diff_std[remap_i,remap_j,0]*distance ,  result[0,1] - self.diff_std[remap_i,remap_j,1]*distance)).reshape(1,-1)
 

    def plot_field(self,xmin,xmax,ymin,ymax,demonstrate_point,step=200,field_density = 20):
        diff_std = self.get_diff_varian_matrix(xmin,xmax,ymin,ymax,demonstrate_point,step=step)
        
        x_step = (xmax - xmin)/step
        y_step = (ymax - ymin)/step
        xx = np.arange(xmin,xmax,x_step)
        yy = np.arange(ymin,ymax,y_step)
        xxx,yyy = np.meshgrid(xx,yy)
        predict_point = np.concatenate((xxx.reshape(-1,1),yyy.reshape(-1,1)),1)        
        
        Weight = np.zeros((step*step))
        for i in range(predict_point.shape[0]):
            distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[i,0])**2+(self.demonstrate_point[:,1]-predict_point[i,1])**2))
            if distance > 2:
                distance = 5.4
            else:
                distance *= 2.7
            self.dis_cof = 0.5/np.max(self.diff_std)
            distance *= self.dis_cof
            Weight[i] = distance
        Weight = Weight.reshape(step,step)

        
        plt.figure()
        plt.subplot(221)
        sns.heatmap(np.sqrt(abs(diff_std[:-1,:-1,0])**2+abs(diff_std[:-1,:-1,1])**2)*Weight[:-1,:-1],cmap="Blues",cbar=False,xticklabels=False,yticklabels=False).invert_yaxis()
        plt.subplot(222)
        sns.heatmap(np.sqrt(abs(diff_std[:-1,:-1,0])**2+abs(diff_std[:-1,:-1,1])**2),cmap="Blues",cbar=False,xticklabels=False,yticklabels=False).invert_yaxis()


        x_step = (xmax - xmin)/field_density
        y_step = (ymax - ymin)/field_density
        xx = np.arange(xmin,xmax,x_step)
        yy = np.arange(ymin,ymax,y_step)
        xxx,yyy = np.meshgrid(xx,yy)
        predict_point = np.concatenate((xxx.reshape(-1,1),yyy.reshape(-1,1)),1)
        result,std = self.gpr1.predict(predict_point,return_std=True)
        
        plt.subplot(223)
        plt.scatter(predict_point[:,0],predict_point[:,1])

        for i in range(field_density**2):
            plt.arrow(predict_point[i,0],predict_point[i,1],result[i,0],result[i,1],head_width=0.01)
        plt.plot(demonstrate_point[:,0],demonstrate_point[:,1])
        
        plt.subplot(224)
        plt.scatter(predict_point[:,0],predict_point[:,1])

        for i in range(field_density**2): 
            remap_i = int(i//field_density) * int(step//field_density)
            remap_j = int(i%field_density) * int(step//field_density)
            res = self.get_next_step(predict_point[i,:].reshape(1,-1))
            plt.arrow(predict_point[i,0],predict_point[i,1], res[0,0], res[0,1],head_width=0.01)        
        plt.plot(demonstrate_point[:,0],demonstrate_point[:,1])

    
    def find_nearest_point(self,point):
        distance = np.linalg.norm((self.data_out-point),axis=1)
        #print('arm',np.argmin(distance))
        return self.data_in[np.argmin(distance)]




class BEZIER():
    def __init__(self):
        self.nSection=10000#根据经验，1米的路程分大于10000000份（7个零）
        self.step_previous=0
        self.f_t_x=[]
        self.f_t_y=[]
        self.length_of_t =[0]* self.nSection
        self.bezier_total_length = 0
    def bezier_tracjectory_generator(self, p0x,p0y,initA,p3x,p3y,endA):
        L=sqrt((p0x-p3x)**2+(p0y-p3y)**2)
        section=L/2.5
        p1x=p0x+section*cos(initA)
        p1y=p0y+section*sin(initA)
        p2x=p3x-section*cos(endA)
        p2y=p3y-section*sin(endA)
        n=self.nSection
        self.f_t_x=[]
        self.f_t_y=[]
        for ii in range(0,n+1):
            t=ii/n
            bezier=[p3x*t**3 - p0x*(t - 1)**3 + 3*p1x*t*(t - 1)**2 - p2x*t**2*(3*t - 3),
                    p3y*t**3 - p0y*(t - 1)**3 + 3*p1y*t*(t - 1)**2 - p2y*t**2*(3*t - 3)]
            self.f_t_x.append(bezier[0])
            self.f_t_y.append(bezier[1])
        bezier_total_length=0.0
        for ii in range(0,n):
            bezier_total_length=bezier_total_length+sqrt((self.f_t_x[ii+1]-self.f_t_x[ii])**2+(self.f_t_y[ii+1]-self.f_t_y[ii])**2)
            self.length_of_t[ii] = bezier_total_length
        self.bezier_total_length = bezier_total_length
        return p1x,p1y,p2x,p2y, bezier_total_length
    
    def find_nearest_t_of_s(self,s):
        step = self.step_previous  
        while step < self.nSection and self.length_of_t[step] < s:
            step += 1
        self.step_previous = step
        return step


    def bezier_tangent_at_t(self, t,p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y):
        tangent_x=3*p1x*(t - 1)**2 - 3*p0x*(t - 1)**2 - 3*p2x*t**2 + 3*p3x*t**2 + 3*p1x*t*(2*t - 2) - 2*p2x*t*(3*t - 3)
        tangent_y=3*p1y*(t - 1)**2 - 3*p0y*(t - 1)**2 - 3*p2y*t**2 + 3*p3y*t**2 + 3*p1y*t*(2*t - 2) - 2*p2y*t*(3*t - 3)
        length=sqrt(tangent_x**2+tangent_y**2)
        norm_direction=[tangent_x/length, tangent_y/length]
        angle=atan2(norm_direction[1],norm_direction[0])
        return norm_direction, angle
    def plot_bezier(self):
        plt.plot(self.f_t_x,self.f_t_y)


    def get_result(self,p0x,p0y,initA,p3x,p3y,endA,step_size=0.002):
        p1x,p1y,p2x,p2y,total_length = self.bezier_tracjectory_generator(p0x,p0y,initA,p3x,p3y,endA)
        step = int(total_length//step_size)+1
        angle_record = []
        w_record = []
        t_record = []
        for i in range(step):
            t = self.find_nearest_t_of_s(step_size*i)
            t_record.append(t)
            _,angle = self.bezier_tangent_at_t(t/self.nSection,p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y)
            angle_record.append(angle)

        for i in range(len(angle_record)-1):
            w_record.append(angle_record[i+1] - angle_record[i])

        
        return w_record




if __name__ == "__main__":

    #init ros publisher 
    rospy.init_node("UR5Control")
    ur_pub = rospy.Publisher ("/arm_controller/command", JointTrajectory,queue_size=0)  
    base_pub = rospy.Publisher ("/mobile_base_controller/cmd_vel", Twist,queue_size=0) 
    ur_sub = rospy.Subscriber("/arm_controller/state",JointTrajectoryControllerState,ur_sub_callback)
    base_sub = rospy.Subscriber("/mobile_base_controller/odom",Odometry,car_sub_callback)
    refer_rviz = rospy.Publisher ("/reference",Marker,queue_size=0) 
    real_rviz = rospy.Publisher ("/real",Marker,queue_size=0) 
    # gazebo_sub = rospy.Subscriber("/gazebo/link_states",LinkStates,gazebo_sub_callback)
    ur_real_rviz = rospy.Publisher ("/ur_real",Marker,queue_size=0) 
    ur_refer_rviz = rospy.Publisher ("/ur_refer",Marker,queue_size=0) 
    time.sleep(0.5)     #need to wait for a short time. If not, the node cannot work
    #prepare for publish
    msg = JointTrajectory()
    vel = Twist()
    np.set_printoptions(threshold=np.inf)   #for convenient showing
    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint','wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point1 = JointTrajectoryPoint()

    #set the car and UR initial state
    vel.linear.x = 0
    base_pub.publish(vel)
    joint_init=[2.1394758224487305+pi/2, -2.140566965142721, 1.9454715887652796, -2.9405275783934535, 4.144657135009766-2*pi, -3.131549660359518]
    joint_init=np.array([-0, -90, -90+30, -90-30, 90, -90])/57.3
    joint_init=np.array([-0, -90, -90, -90, 90, -90])/57.3

    point1.positions = joint_init
    point1.time_from_start = rospy.Duration(1,0)
    msg.points = [point1]
    msg.header.seq = 0
    ur_pub.publish(msg)
    time.sleep(1.5)

    def reach(theta,distance):
        vel = Twist()
        vel.angular.z = 0
        vel.linear.x = 0
        base_pub.publish(vel)
        time.sleep(0.1)
        angular = pi/2/5
        linear = 0.2
        #给一个角度，然后给定了速度，那我就自然知道了总时间，进而知道总步长
        dt = 0.1
        for i in range(int(theta/dt/angular)):
            vel.angular.z = angular
            vel.linear.x = 0
            base_pub.publish(vel)
            time.sleep(dt)
        vel.angular.z = 0
        vel.linear.x = 0
        base_pub.publish(vel)
        time.sleep(0.1)

        for i in range(int(distance/dt/linear)):
            vel.angular.z = 0
            vel.linear.x = linear
            base_pub.publish(vel)
            time.sleep(dt)        
        vel.angular.z = 0
        vel.linear.x = 0
        base_pub.publish(vel)
        time.sleep(0.1)
        for i in range(int(theta/dt/angular)):
            vel.angular.z = -angular
            vel.linear.x = 0
            base_pub.publish(vel)
            time.sleep(dt)       
        vel.angular.z = 0
        vel.linear.x = 0
        base_pub.publish(vel)
        time.sleep(0.1) 
    
    #reach(0,1.5)
    
    #read demonstation data
    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path = readTxt()
    down_times = int(xy_path.shape[0]//40 + 1)
    xy_dot_path_filtered = mean_filter(xy_dot_path)
    xy_dot_path_filtered_down = down_sample(xy_dot_path_filtered,down_times)
    xy_path_down = down_sample(xy_path,down_times)
    
    
    #generate UR desired path
    arm_xyztheta_path_noise = [] 
    arm_xyztheta_path = [] 
    center_x = 0.15
    center_y = 0
    r = 0.15
    # for ii in range(1,len(xy_path)+1):
    #     i = ii/len(xy_path)*2*pi
    #     arm_xyztheta_path_noise.append(np.array((center_x-r*cos(i),center_y+r*sin(i),0,0)) + 0.00001*np.random.normal(0,2,4))
    #     arm_xyztheta_path.append(np.array((center_x-r*cos(i),center_y+r*sin(i),0,0)))
    # arm_xyztheta_path_down = down_sample(np.array(arm_xyztheta_path),down_times)
    # arm_xyztheta_path_noise_down = down_sample(np.array(arm_xyztheta_path_noise),down_times)

    for ii in range(1,len(xy_path)+1):
        i = ii/len(xy_path)*0.3
        th = ii/len(xy_path)*2*pi
        arm_xyztheta_path_noise.append(np.array((0.15*sin(th),i,0,0)) + 0.00001*np.random.normal(0,2,4))
        arm_xyztheta_path.append(np.array((0.15*sin(th),i,0,0)))
    arm_xyztheta_path_down = down_sample(np.array(arm_xyztheta_path),down_times)
    arm_xyztheta_path_noise_down = down_sample(np.array(arm_xyztheta_path_noise),down_times)

    #Fit the GPR
    car_gpr = MyGPR(xy_path_down,np.array(xy_dot_path_filtered_down))
    car2arm_gpr = MyGPR(xy_path_down,arm_xyztheta_path_noise_down,normalize=False)
    #After fitting, it is better to change the data to undown-sample for corrdination term 
    # car2arm_gpr.data_in = xy_path
    # car2arm_gpr.data_out = arm_xyztheta_path
    # print(arm_xyztheta_path)
    # coordinate_result = car2arm_gpr.find_nearest_point(np.array((0,0,0,0)).reshape(1,-1))
    # print('coor',coordinate_result)

    result1 = car2arm_gpr.gpr1.predict(xy_path,return_std=False)
    #plot the attractive field
    car_gpr.plot_field(-1,1,-1,1,xy_path)
    #plt.show()
    #test the UR GPR
    # plt.figure()
    result1 = car2arm_gpr.gpr1.predict(xy_path_down,return_std=False)
    # result2 = car2arm_gpr.gpr1.predict(xy_path_down+np.array((0,0.05)),return_std=False)
    # plt.plot(result1[:,0],result1[:,1])
    # plt.plot(result2[:,0],result2[:,1])
    #plt.show()


    def visualize(pub,points,r,g,b,a,scale):
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = rospy.Time()
        marker.id = 0
        marker.ns = "/"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.color.a = a
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.scale.x = scale
        marker.points=points
        pub.publish(marker)
    point = Point()
    points = []
    ur_point = Point()
    ur_points = []
    for i in range(xy_path.shape[0]):
        #marker.pose.position.x += 0.2
        point = Point()
        point.x = xy_path[i][0]
        point.y = xy_path[i][1]
        points.append(point)
        visualize(refer_rviz,points,1,0,0,0.7,0.02)
        time.sleep(0.002)

    points = []
    ur_points = []

    #move slowly to the first disired point
    end_init_state = ur_end_record[-1]
    end_state = ur_end_record[-1]
    arm_state = ur_sub_record[-1]
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    robot_target = car2arm_gpr.gpr1.predict(np.array((current_x,current_y)).reshape(1,-1))
    wholebody_init_state = ur5e_car.get_end_effector_posture([current_x,current_y,current_theta] + arm_state)
    wholebody_init_state = wholebody_init_state[:,3]

    time.sleep(2)


    #设置当前车子状态
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    arm_state = ur_sub_record[-1]
    end_state = ur_end_record[-1]
    car_sub_record.clear()
    ur_sub_record.clear()
    ur_end_record.clear()
    step = 0
    bz_flag = False
    bz_index = 0
    w_record = []
    time.sleep(0.2)
    coor_lin = 0
    coor_ang = 0
    coor_flag = 0
    soft_index = 0

    while True:    
        #if near the target, break the loop
        if np.sqrt((current_x-xy_path_down[-1,0])**2+(current_y-xy_path_down[-1,1])**2) < 0.01:
            break
        t0 = time.time()
        step +=1
        
        #get the car next step
        result = car_gpr.get_next_step(np.array((current_x,current_y)).reshape(1,-1))
        target_angle = np.arctan2(result[0,1],result[0,0])
        delta_theta = np.arctan2(result[0,1],result[0,0])-current_theta
        #Bezier步长
        step_size = 0.2    


        #####coordination
        nearest_path = xy_path[np.argmin(np.linalg.norm((xy_path-np.array((current_x,current_y)).reshape(1,-1)),axis=1))]

        if (step >=2500 and step <= 3200) and coor_flag == 0:
            coor_flag  = 2
            coordinate_index = np.argmin(np.linalg.norm((xy_path-np.array((current_x,current_y)).reshape(1,-1)),axis=1))-50
            soft_index = 500


        if (step >=4000 and step <= 4500) and coor_flag == 0:
            coor_flag  = 1
            coordinate_result = np.array((current_x,current_y))
            #coordinate_result = xy_path[coordinate_index+int(soft_index/1)]
            


        if (step >=5500 and step <= 6500) and coor_flag == 0:
            coor_flag  = 3
            coordinate_index = np.argmin(np.linalg.norm((xy_path-np.array((current_x,current_y)).reshape(1,-1)),axis=1))+50
            soft_index = 50
            #coordinate_result = xy_path[coordinate_index-int(soft_index/1)]

        if not ((step >=2500 and step <= 3200) or (step >=4000 and step <= 4500) or (step >=5500 and step <= 6500)):
            coor_flag  = 0
            distance = np.linalg.norm(xy_path-np.array((current_x,current_y)).reshape(1,-1),axis=1)
            coordinate_result = np.array(xy_path[np.argmin(distance)])
            soft_index=0
            coordinate_index = np.argmin(distance)
        distance1 = np.sqrt((current_x-coordinate_result[0])**2+(current_y-coordinate_result[1])**2)
        if step % 11==0:
            print('current:',current_x,' ',current_y,'corrdinate:',coordinate_result,'distance1:',distance1)
            car_coor_record.append(np.argmin(distance))
            if coor_flag == 3:
                ur_coor_record.append(coordinate_index-int(soft_index/1))
            else:
                ur_coor_record.append(coordinate_index+int(soft_index/10))
            print('car',np.argmin(np.linalg.norm((xy_path-np.array((current_x,current_y)).reshape(1,-1)),axis=1)))
            print(step)

        if(distance1 > 0.00):
            #启动协调过程
            result = car_gpr.gpr1.predict(np.array((coordinate_result[0],coordinate_result[1])).reshape(1,-1))
            target_angle = atan2(result[0,1],result[0,0])
            delta_theta = target_angle - current_theta
            
            if(abs(atan2(coordinate_result[1] - current_y,coordinate_result[0] - current_x) - current_theta) < 0.5):#正着走
                #print(coor_flag)
                coor_lin = distance1*5
                print(coor_lin,np.sqrt(result[0,0]**2+result[0,1]**2)*0.5)
                coor_ang = (target_angle-current_theta)*3


            elif(abs(atan2(-coordinate_result[1] + current_y,-coordinate_result[0] + current_x) - current_theta) < 0.5):#反着走   
                #print(coor_flag)
                coor_lin = -distance1*5
                print(coor_lin,np.sqrt(result[0,0]**2+result[0,1]**2)*0.5)
                coor_ang = (target_angle-current_theta)*3



        
        vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0.5 +coor_lin
        vel.angular.z = (np.arctan2(result[0,1],result[0,0])-current_theta)*5+coor_ang  
        coor_lin = 0
        coor_ang = 0
        if abs(vel.angular.z) > 0.2:
            vel.angular.z = 0.2*np.sign(vel.angular.z)
        if not(step >= 1000 and step <=2000):
            base_pub.publish(vel)


        #################################################################################################################################        
        #机械臂控制
        wholebody_state = ur5e_car.get_end_effector_posture([current_x,current_y,current_theta] + arm_state)
        wholebody_state = wholebody_state[:,3]
        distance = np.linalg.norm(xy_path-np.array((current_x,current_y)).reshape(1,-1),axis=1)
        if step %10 == 0:
            print('arm',np.argmin(distance))
        
        #robot_target = car2arm_gpr.gpr1.predict(xy_path[np.argmin(distance)].reshape(1,-1))
        
        if coor_flag == 2:
            #print(soft_index)
            if soft_index >= 0:
                soft_index -=1
                coordinate_result = xy_path[coordinate_index+int(soft_index/10)]
        if coor_flag == 3:
            #print(soft_index)
            if soft_index >= 0:
                soft_index -=1
                coordinate_result = xy_path[coordinate_index-int(soft_index/1)]
        robot_target = car2arm_gpr.gpr1.predict(coordinate_result.reshape(1,-1))
        #robot_target = car2arm_gpr.gpr1.predict(np.array((current_x,current_y)).reshape(1,-1))
        #robot_target = car2arm_gpr.gpr1.predict(xy_path[np.argmin(distance)].reshape(1,-1))
        velocity_X = np.array((robot_target[0,0]+wholebody_init_state[0]-wholebody_state[0],robot_target[0,1]+wholebody_init_state[1]-wholebody_state[1],robot_target[0,2]+wholebody_init_state[2]-wholebody_state[2],0,0,0))


        wholeBodyState = car_sub_record[-1][0:3]+ur_sub_record[-1]
        wholebody_J = ur5e_car.get_jacobian_lx(wholeBodyState)
        Weight=np.array([[999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        z=[vel.linear.x*0.01, vel.angular.z*0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #z=[0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        J_transpose=np.transpose(wholebody_J)
        Weignt_inverse=np.linalg.pinv(Weight)
        J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, np.linalg.pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
        joint_v=np.matmul(J_sharp,velocity_X)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)
        point1.positions = arm_state + joint_v[2:].ravel()
        point1.time_from_start = rospy.Duration(0.01,0)
        msg.points = [point1]
        if coor_flag != 1:
            ur_pub.publish(msg)
        
        wholebody_state = ur5e_car.get_end_effector_posture([current_x,current_y,current_theta] + arm_state)
        wholebody_state = wholebody_state[:,3]
        wb_record.append([wholebody_state[0],wholebody_state[1],wholebody_state[2]])
        point = Point()
        point.x = current_x
        point.y = current_y
        points.append(point)
        


        wholebody_state = ur5e_car.get_end_effector_posture([current_x,current_y,current_theta] + arm_state)
        ur_point = Point()
        ur_point.x = wholebody_state[0,3]
        ur_point.y = wholebody_state[1,3]
        ur_point.z = wholebody_state[2,3]
        ur_points.append(ur_point)

        visualize(ur_real_rviz,ur_points,0,0,1,0.7,0.02)
        visualize(real_rviz,points,0,1,0,0.7,0.04)


        if time.time()- t0 < 0.008:
            time.sleep(0.01-time.time()+t0)

        step +=1
        end_state = ur_end_record[-1]
        arm_state = ur_sub_record[-1]
        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]




    Stop_flag = True      
    wb_record = np.array(wb_record)
    ur_end_record = np.array(ur_end_record)
    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)
    # plt.figure()
    # plt.plot([x[0] for x in car_sub_record],[x[1] for x in car_sub_record])
    # plt.plot(xy_path[:,0],xy_path[:,1])
    # plt.figure()

    # wb_record=mean_filter(wb_record,4,True)
    # plt.plot(wb_record[:,0]-wb_record[0,0],wb_record[:,1]-wb_record[0,1])
    # plt.plot(result1[:,0],result1[:,1])
    # plt.legend(('1','2','3','4','5','6'))
    # plt.figure()
    # plt.plot(car_coor_record)
    # plt.plot(ur_coor_record)

    np.savetxt('data/2adrawsin_car_coor.txt',np.array(car_coor_record))
    np.savetxt('data/2adrawsin_ur_coor.txt',np.array(ur_coor_record))
    np.savetxt('data/2adrawsin_wb.txt',np.array(wb_record[:,:]-wb_record[0,:]))
    np.savetxt('data/2adrawsin_car.txt',np.array(car_sub_record))
    np.savetxt('data/2adrawsin_car_ref.txt',np.array(xy_path))
    np.savetxt('data/2adrawsin_wb_ref.txt',result1)

    plt.show()
    plt.pause(0)


