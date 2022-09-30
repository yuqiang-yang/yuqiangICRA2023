#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('.')
import matplotlib.pyplot as plt
from trajectory_msgs.msg import *
from control_msgs.msg import *
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from geometry_msgs.msg import Twist,Point
from control_msgs.msg import JointTrajectoryControllerState
from nav_msgs.msg import Odometry
import numpy as np
import rospy
import time
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from visualization_msgs.msg import Marker  
from gazebo_msgs.msg import LinkStates
from jacobi.ur5e_car_kinematics_class_2 import UR5e_car_kinematics

ur5e_car = UR5e_car_kinematics()

#Data Record
ur_sub_record = []
ur_end_record = []
car_sub_record = []
start_x = 0
start_y = 0
gazebo_refresh = True
# def gazebo_sub_callback(state):

#     print('1')
#     #gazebo_refresh = False
#     x = state.pose[1].orientation.x
#     y = state.pose[1].orientation.y
#     z = state.pose[1].orientation.z
#     w = state.pose[1].orientation.w
#     theta = atan2(2*(w*z+x*y),1-2*(y**2+z**2))
#     # car_sub_record.append([state.pose[1].position.x,state.pose[1].position.y,theta])
#     ur_sub_record.append([state.pose[1].position.x,state.pose[1].position.y,0])

#         # ur_sub_record.append([state.pose[-2].position.x,state.pose[-2].position.y,state.pose[-2].position.z])

def ur_sub_callback(state):
    ur_sub_record.append(list(state.actual.positions))
    # ee_pose = ur5_kin.forward(ur_sub_record[-1])
    # ee_pose = np.asarray(ee_pose).reshape(3,4)

    # R = ee_pose[0:3,0:3]
    # sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    # singular = sy < 1e-6

    # if  not singular :
    #     x = math.atan2(R[2,1] , R[2,2])
    #     y = math.atan2(-R[2,0], sy)
    #     z = math.atan2(R[1,0], R[0,0])
    # else :
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0
    # if Stop_flag == True:
    #     sys.exit()
    # ur_end_record.append([ee_pose[0,3],ee_pose[1,3],ee_pose[2,3],x,y,z])


#mobile base state subsciber callback function
def car_sub_callback(odom):
    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])


#read demonstration data
def readTxt(path):
    data = np.loadtxt(path)
    xy_path = data[:,0:2]
    xy_dot_path = np.concatenate((data[:,3].reshape(-1,1)*np.cos(data[:,2].reshape(-1,1)),data[:,3].reshape(-1,1)*np.sin(data[:,2]).reshape(-1,1)),axis=1)
    xy_theta_path = data[:,0:3]
    xy_theta_dot_path = np.concatenate((xy_dot_path,data[:,5].reshape(-1,1)),axis=1)

    return xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path

#given n*m numpy data,return the data filtered
def mean_filter(data,windows_size = 4):
    rows, cols = data.shape
    filtedData = np.zeros((rows,cols))
    windows = np.zeros((windows_size,cols))
    for i in range(windows_size):
        windows[i,:] = data[0,:]
    
    for i in range(rows):
        windows[i%windows_size,:] = data[i,:]
        filtedData[i,:] = np.sum(windows,axis=0)/windows_size
    return filtedData

#given n*m numpy data,return n/times * m numpy data
def down_sample(data,times):
    rows, cols = data.shape
    down_data = np.zeros((int(rows//times),cols))
    for i in range(int(rows//times)):
        down_data[i,:] = data[i*times,:]
    return down_data

#Gaussian Process Regression implementation
class MyGPR():
    def __init__(self,data_in,data_out):
        self.data_in = data_in
        self.data_out = data_out
        #kernel1 = 1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+1*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
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
        self.std_matrix = std_matrix
        for i in range(step-1):
            for j in range(step-1):
                self.diff_std[i,j,0] = (std_matrix[i,j+1] - std_matrix[i,j])
                self.diff_std[i,j,1] = (std_matrix[i+1,j] - std_matrix[i,j])
        self.diff_std[:,:,0] /= x_step
        self.diff_std[:,:,1] /= y_step
        self.long_dis_stable = (np.max(abs(self.diff_std)))/4

        return self.diff_std
    def get_next_step_jens(self,predict_point):
        result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)

        return np.array((result[0,0] - (self.diff_std[remap_i,remap_j,0])*1.85 ,  result[0,1] - (self.diff_std[remap_i,remap_j,1])*1.7)).reshape(1,-1)
 
    def get_next_step(self,predict_point):
        result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)
        distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        min_index = np.argmin(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        long_dis_stable_direction = -(self.demonstrate_point[min_index] - predict_point)/np.linalg.norm(self.demonstrate_point[min_index] - predict_point)

        if distance > 2:
            distance = 5.4
        else:
            distance *= 2.7
        self.dis_cof = 0.1/np.max(self.diff_std)
        distance *= self.dis_cof
        return np.array((result[0,0] - (self.diff_std[remap_i,remap_j,0]+long_dis_stable_direction[0,0]*self.long_dis_stable)*distance ,  result[0,1] - (self.diff_std[remap_i,remap_j,1]+long_dis_stable_direction[0,1]*self.long_dis_stable)*distance)).reshape(1,-1)
    def get_varian(self,predict_point):
        #result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)
        distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        min_index = np.argmin(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        long_dis_stable_direction = -(self.demonstrate_point[min_index] - predict_point)/np.linalg.norm(self.demonstrate_point[min_index] - predict_point)

        if distance > 2:
            distance = 5.4
        else:
            distance *= 2.7
        self.dis_cof = 0.1/np.max(self.diff_std)
        distance *= self.dis_cof
        return np.array((- (self.diff_std[remap_i,remap_j,0]+long_dis_stable_direction[0,0]*self.long_dis_stable)*distance ,   - (self.diff_std[remap_i,remap_j,1]+long_dis_stable_direction[0,1]*self.long_dis_stable)*distance)).reshape(1,-1)

    def plot_field(self,xmin,xmax,ymin,ymax,demonstrate_point,step=75,field_density = 7):
        diff_std = self.get_diff_varian_matrix(xmin,xmax,ymin,ymax,demonstrate_point,step=step)
        
    def find_nearest_point(self,point):
        distance = np.linalg.norm((self.data_out-point),axis=1)
        print('arm',np.argmin(distance))
        return self.data_in[np.argmin(distance)]

def time_optimal_for_1D_tracking(x1_tf, x2_tf, x1_t0, x2_t0, abs_acc):
    string=''
    if abs(x1_tf-x1_t0)+abs(x2_tf-x2_t0)<0.001:
        abs_acc=0.001
    if x2_t0>x2_tf:#,走abs_acc<0
        string=string+'x2_t0>x2_tf  |'
        t_minus=(x2_t0-x2_tf)/(-abs_acc)
        temp=0.5*(-abs_acc)*t_minus**2+x2_t0*t_minus+x1_tf
        if x1_t0<temp:
            acc=abs_acc
            string=string+'x1_t0<temp  |加速'+str(acc)
        else:#x1_t0>=temp
            acc=-abs_acc
            string=string+'x1_t0>=temp  |减速'+str(acc)
    elif x2_t0<x2_tf:#,走abs_acc>0
        string=string+'x2_t0<x2_tf  |'
        t_plus=(x2_t0-x2_tf)/abs_acc
        temp=0.5*abs_acc*t_plus**2+x2_t0*t_plus+x1_tf
        if x1_t0<=temp:
            acc=abs_acc
            string=string+'x1_t0<=temp  |加速'+str(acc)
            
        else:
            acc=-abs_acc
            string=string+'x1_t0>temp  |减速'+str(acc)
    else:#x2_t0==x2_tf
        string=string+'x2_t0==x2_tf  |'
        if x1_t0<x1_tf:
            acc=abs_acc
            string=string+'x1_t0<x1_tf  |加速'+str(acc)
            
        elif x1_t0>x1_tf:
            acc=-abs_acc
            string=string+'x1_t0>x1_tf  |减速'+str(acc)
            
        else:#x1_t0==x1_tf
            acc=0.0
            string=string+'x1_t0==x1_tf  |不变'+str(acc)
    return acc,string

if __name__ == "__main__":
    #init ros node
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
    time.sleep(0.5)
    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path = readTxt('data/car_xytheta_dot_record1.txt')
    down_times = int(xy_path.shape[0]//40 + 1)
    print('down_times',down_times)
    xy_dot_path_filtered = mean_filter(xy_dot_path)
    xy_dot_path_filtered_down = down_sample(xy_dot_path,down_times)
    xy_path_down = down_sample(xy_path,down_times)
    vel = Twist()
    joint_init=np.array([-0, -90, -90, -90, 90, -90])/57.3
    msg = JointTrajectory()
    vel.angular.z = 0
    vel.linear.x = 0
    base_pub.publish(vel)
    point1 = JointTrajectoryPoint()
    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint','wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


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
    start_theta = 0
    start_dis = 2
    reach(start_theta,start_dis)
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

    #read data and pre-process
    points = []


    gpr = MyGPR(xy_path_down,xy_dot_path_filtered_down)
    p=gpr.plot_field(-2,5,-2,5,xy_path)



    #for gazebo ros controller
    msg = JointTrajectory()
    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()


    step = 0
    dt = 0.01
    bz_flag = False
    car_angular_vel = 30
    car_angular_acc = 0.5
    vel.linear.x = 0
    base_pub.publish(vel)
    time.sleep(0.5)


    #limits 
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    arm_state = ur_sub_record[-1]

    car_sub_record.clear()
    ur_sub_record.clear()
    plt.figure()
    step =0
    while True:
        step += 1
        #if the car is near the target, break the loop
        if np.sqrt((current_x-2.9)**2+(current_y-2.9)**2) < 0.2:
            break
        t0 = time.time()
        
        result = gpr.get_next_step(np.array((current_x,current_y)).reshape(1,-1))

        target_angle = np.arctan2(result[0,1],result[0,0])
        delta_theta = np.arctan2(result[0,1],result[0,0])-current_theta
        print(current_y,current_y,np.sqrt((current_x-start_dis*cos(start_theta))**2+(current_y-start_dis*sin(start_theta))**2))
        # w0 = car_sub_record[-1][5]
        w0 = 0
        #if delta angle is small, run directly.
        if np.abs(delta_theta) > car_angular_vel*dt and np.sqrt((current_x-start_dis*cos(start_theta))**2+(current_y-start_dis*sin(start_theta))**2) > 0.2:        
            bz_flag = True
        elif(np.abs(delta_theta) > 5*dt):
            bz_flag = True
        #if the angle is large, turn to the disired angle first
        if bz_flag ==True:
            bz_flag = False
            vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0.0
            if np.sqrt((current_x-start_dis*cos(start_theta))**2+(current_y-start_dis*sin(start_theta))**2) > 0.2:
                vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0.5
            #acc,_ = time_optimal_for_1D_tracking(delta_theta, 0, 0, w0, car_angular_acc)
            vel.angular.z = delta_theta*2.5
            #print('Bezier','acc:',acc,'delta:',delta_theta,'w0',w0)
        else:    
            acc,_ = time_optimal_for_1D_tracking(delta_theta, delta_theta*0, 0, w0, car_angular_acc)

            vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0.5
            vel.angular.z = delta_theta*5
        #vel.angular.z = w0+acc*dt
        if(abs(vel.angular.z) > 0.2):
            vel.angular.z = np.sign(vel.angular.z)*0.2
        base_pub.publish(vel)
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

        #visualize(ur_real_rviz,ur_points,0,0,1,0.7,0.04)
        visualize(real_rviz,points,0,1,0,0.7,0.04)

        time.sleep(dt)
        arm_state = ur_sub_record[-1]

        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    print('eeeeeeeeeee')
    #stop the car
    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)

    plt.plot([x[0] for x in car_sub_record],[x[1] for x in car_sub_record])
    plt.plot(xy_path[:,0],xy_path[:,1])

    #np.savetxt('data/car_xytheta_dot_record3.txt',np.array(car_sub_record)[:,:])

    #plt.show()
    plt.pause(0)


