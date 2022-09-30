#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from trajectory_msgs.msg import *
from control_msgs.msg import *
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import LinkStates

from control_msgs.msg import JointTrajectoryControllerState
from nav_msgs.msg import Odometry
import numpy as np
import rospy
import time
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import seaborn as sns
 
#Data Record
ur_sub_record = []
car_sub_record = []
gazebo_sub_record = []
def gazebo_sub_callback(state):
    x = state.pose[1].orientation.x
    y = state.pose[1].orientation.y
    z = state.pose[1].orientation.z
    w = state.pose[1].orientation.w
    theta = atan2(2*(w*z+x*y),1-2*(y**2+z**2))

    gazebo_sub_record.append([state.pose[1].position.x,state.pose[1].position.y,theta,state.pose[1].position.y,theta,state.twist[1].angular.z])

#ur state subsciber callback function
def ur_sub_callback(state):
    ur_sub_record.append(list(state.actual.positions))

#mobile base state subsciber callback function
def car_sub_callback(odom):
    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    car_sub_record.append([odom.pose.pose.position.x+1, odom.pose.pose.position.y+1, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])

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

        return self.diff_std

    def get_next_step(self,predict_point):
        result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)
        distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        if distance > 2:
            distance = 5.4
        else:
            distance *= 2.7
        self.dis_cof = 0.05/np.max(self.diff_std)
        distance *= self.dis_cof
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
            self.dis_cof = 0.2/np.max(self.diff_std)
            distance *= self.dis_cof
            Weight[i] = distance
        Weight = Weight.reshape(step,step)

        plt.figure()
        sns.heatmap(self.std_matrix,cmap="Blues",cbar=False,xticklabels=False,yticklabels=False).invert_yaxis()
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
        print('arm',np.argmin(distance))
        return self.data_in[np.argmin(distance)]

#Bezier Curve 
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


def time_optimal_for_1D_tracking(x1_tf, x2_tf, x1_t0, x2_t0, abs_acc):
    string=''
    if abs(x1_tf-x1_t0)+abs(x2_tf-x2_t0)<0.001:
        abs_acc=0.01
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
    #base_sub = rospy.Subscriber("/mobile_base_controller/odom",Odometry,car_sub_callback)
    gazebo_sub = rospy.Subscriber("/gazebo/link_states",LinkStates,gazebo_sub_callback)
    #read data and pre-process
    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path = readTxt('data/car_xytheta_dot_record1.txt')
    down_times = int(xy_path.shape[0]//40 + 1)
    print('down_times',down_times)
    xy_dot_path_filtered = mean_filter(xy_dot_path)
    xy_dot_path_filtered_down = down_sample(xy_dot_path,down_times)
    xy_path_down = down_sample(xy_path,down_times)


    gpr = MyGPR(xy_path_down,xy_dot_path_filtered_down)
    gpr.plot_field(-2,5,-2,5,xy_path)
    #plt.show()

    #for gazebo ros controller
    msg = JointTrajectory()
    vel = Twist()
    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()


    step = 0
    dt = 0.01
    bz_flag = False
    car_angular_vel = 0.48
    car_angular_acc = 0.5
    vel.linear.x = 0
    base_pub.publish(vel)
    time.sleep(0.5)

    a0 = 0
    #limits 
    current_x,current_y,current_theta = gazebo_sub_record[-1][0],gazebo_sub_record[-1][1],gazebo_sub_record[-1][2]
    car_sub_record.clear()
    ur_sub_record.clear()
    gazebo_sub_record.clear()
    plt.figure()
    w0_record = []

    while True:
        

        t0 = time.time()
        

        result = gpr.get_next_step(np.array((current_x,current_y)).reshape(1,-1))

        target_angle = 90/57.3
        target_w0 = -0.1
        delta_theta = target_angle-current_theta
        w0 = vel.angular.z
        w0_record.append([current_theta,w0])
        #w0 = gazebo_sub_record[-1][3]

        #if delta angle is small, run directly.
        if np.abs(delta_theta) >= 0:        
            bz_flag = True

        #if the angle is large, turn to the disired angle first
        if bz_flag ==True:
            bz_flag = False
            vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0.0
            acc,_ = time_optimal_for_1D_tracking(target_angle, target_w0, current_theta, w0, car_angular_acc)
            vel.linear.x = 0
            print('Bezier','acc:',acc,'delta:',delta_theta,'w0',w0)
        else:    
            acc,_ = time_optimal_for_1D_tracking(delta_theta, target_w0, 0, w0, car_angular_acc)
            vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0

        #if the car is near the target, break the loop
        if np.abs(delta_theta) < 0.01 and abs(w0-target_w0) < 0.05:
            print(w0,delta_theta)
            break
        vel.angular.z = w0+acc*dt
        a0 += vel.angular.z*dt
        base_pub.publish(vel)
        time.sleep(dt)
        current_x,current_y,current_theta = gazebo_sub_record[-1][0],gazebo_sub_record[-1][1],gazebo_sub_record[-1][2]
    #stop the car
    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)

    plt.plot([x[0] for x in w0_record],[x[1] for x in w0_record])
    #plt.plot(xy_path[:,0],xy_path[:,1])

    #np.savetxt('data/car_xytheta_dot_record3.txt',np.array(car_sub_record)[:,:])

    plt.show()
    plt.pause(0)


