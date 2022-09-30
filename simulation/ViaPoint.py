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
from matplotlib import projections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trajectory_msgs.msg import *
from control_msgs.msg import *
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import LinkStates
from nav_msgs.msg import Odometry
import ikfast.ikfastpy as ikfastpy


ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()
ur_sub_record = []
ur_end_record = []
car_sub_record = []

#UR state subsciber callback function
#FK to end pose 
def ur_sub_callback(state):
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
    ur_end_record.append([ee_pose[0,3],ee_pose[1,3],ee_pose[2,3],x,y,z])


#car state subscriber callback function
def car_sub_callback(odom):
    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])

#read the demonstration data
def readTxt():
    data = np.loadtxt('data/car_xytheta_dot_record1.txt')
    xy_path = data[:,0:2]
    xy_dot_path = np.concatenate((data[:,3].reshape(-1,1)*np.cos(data[:,2].reshape(-1,1)),data[:,3].reshape(-1,1)*np.sin(data[:,2]).reshape(-1,1)),axis=1)
    xy_theta_path = data[:,0:3]
    xy_theta_dot_path = np.concatenate((xy_dot_path,data[:,5].reshape(-1,1)),axis=1)

    return xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path


def mean_filter(data):
    rows, cols = data.shape
    filtedData = np.zeros((rows,cols))
    windows_size = 4
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
        kernel1 = 1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+10*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
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
            self.dis_cof = 0.2/np.max(self.diff_std)
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
        print('arm',np.argmin(distance))
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
    time.sleep(0.5)     #need to wait for a short time. If not, the node cannot work

    #prepare for publish
    msg = JointTrajectory()
    vel = Twist()
    np.set_printoptions(threshold=np.inf)   #for convenient showing
    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint','wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()

    #set the car and UR initial state
    vel.linear.x = 0
    base_pub.publish(vel)
    joint_init=[2.1394758224487305, -2.140566965142721, 1.9454715887652796, -2.9405275783934535, 4.144657135009766-2*pi, -3.131549660359518]

    point.positions = joint_init
    point.time_from_start = rospy.Duration(1,0)
    msg.points = [point]
    msg.header.seq = 0
    ur_pub.publish(msg)
    time.sleep(1.5)


    
    #read demonstation data
    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path = readTxt()
    down_times = int(xy_path.shape[0]//20 + 1)
    xy_dot_path_filtered = mean_filter(xy_dot_path)
    xy_dot_path_filtered_down = down_sample(xy_dot_path_filtered,down_times)
    xy_path_down = down_sample(xy_path,down_times)
    
    
    #generate UR desired path
    arm_xyztheta_path_noise = [] 
    arm_xyztheta_path = [] 
    for ii in range(1,len(xy_path)+1):
        i = ii/down_times
        arm_xyztheta_path_noise.append([0.0015*i,0.0*i,-0.10*np.cos(2*3.14159/20*i),0.001*i] + 0.0002*np.random.normal(0,2,4))
        arm_xyztheta_path.append([0.0015*i,0.0*i,-0.10*np.cos(2*3.14159/20*i),0.001*i])
    arm_xyztheta_path_down = down_sample(np.array(arm_xyztheta_path),down_times)
    arm_xyztheta_path_noise_down = down_sample(np.array(arm_xyztheta_path_noise),down_times)



    #Fit the GPR
    car_gpr = MyGPR(xy_path_down,xy_dot_path_filtered_down)
    car2arm_gpr = MyGPR(xy_path_down,arm_xyztheta_path_noise_down,normalize=False)

    #via point1
    kernel2 = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3))+0.0001*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
    xy_via = xy_path_down[0].reshape(1,-1)#np.array((0,0)).reshape(1,-1)
    xy_via = np.vstack((xy_via,xy_path_down[5].reshape(1,-1)))
    xy_via = np.vstack((xy_via,xy_path_down[10].reshape(1,-1)))
    xy_via = np.vstack((xy_via,xy_path_down[15].reshape(1,-1)))
    xy_via = np.vstack((xy_via,xy_path_down[18].reshape(1,-1)))
    arm_via = arm_xyztheta_path_down[0].reshape(1,-1)
    arm_via = np.vstack((arm_via,arm_xyztheta_path_down[5].reshape(1,-1)))
    via_ppp = arm_xyztheta_path_down[10]+np.array((0,0,-0.03,0))
    arm_via = np.vstack((arm_via,via_ppp.reshape(1,-1)))
    #arm_via = np.vstack((arm_via,arm_xyztheta_path_down[10].reshape(1,-1)))
    arm_via = np.vstack((arm_via,arm_xyztheta_path_down[15].reshape(1,-1)))
    arm_via = np.vstack((arm_via,arm_xyztheta_path_down[18].reshape(1,-1)))
    noise = np.array((1e-5,1e-5,1e-7,1e-5,1e-5))
    gpr2 = GaussianProcessRegressor(kernel=kernel2,random_state=0,n_restarts_optimizer=5,alpha=noise,normalize_y=False)
    gpr2.fit(xy_via,arm_via)
    #After fitting, it is better to change the data to undown-sample for corrdination term 
    distance = np.linalg.norm(xy_path-xy_via[2].reshape(1,-1),axis=1)
    std_via = 1e-5+0.05*distance**2
    result2,std2 = gpr2.predict(xy_path,return_std=True)
    std2 /= 10
    std2 += std_via

    ######
    fig = plt.figure()

    ax1 = Axes3D(fig)
    ax1.scatter3D(arm_via[2,0],arm_via[2,1],arm_via[2,2],linewidth=5,color='red')
    ###
    #via point2
    kernel2 = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3))+0.0001*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
    xy_via = xy_path_down[0].reshape(1,-1)#np.array((0,0)).reshape(1,-1)
    xy_via = np.vstack((xy_via,xy_path_down[5].reshape(1,-1)))
    xy_via = np.vstack((xy_via,xy_path_down[10].reshape(1,-1)))
    xy_via = np.vstack((xy_via,xy_path_down[15].reshape(1,-1)))
    xy_via = np.vstack((xy_via,xy_path_down[18].reshape(1,-1)))
    arm_via = arm_xyztheta_path_down[0].reshape(1,-1)
    via_ppp = arm_xyztheta_path_down[5]+np.array((0,0,0.05,0))
    arm_via = np.vstack((arm_via,via_ppp.reshape(1,-1)))
    arm_via = np.vstack((arm_via,arm_xyztheta_path_down[10].reshape(1,-1)))
    arm_via = np.vstack((arm_via,arm_xyztheta_path_down[15].reshape(1,-1)))
    arm_via = np.vstack((arm_via,arm_xyztheta_path_down[18].reshape(1,-1)))
    noise = np.array((1e-5,1e-5,1e-5,1e-5,1e-5))
    gpr3 = GaussianProcessRegressor(kernel=kernel2,random_state=0,n_restarts_optimizer=5,alpha=noise,normalize_y=False)
    gpr3.fit(xy_via,arm_via)
    #After fitting, it is better to change the data to undown-sample for corrdination term 
    distance = np.linalg.norm(xy_path-xy_via[1].reshape(1,-1),axis=1)
    std_via3 = 1e-5+0.05*distance**2
    result3,std3 = gpr3.predict(xy_path,return_std=True)
    std3 /= 10
    std3 += std_via3




    #plot the attractive field
    #car_gpr.plot_field(-2,5,-2,5,xy_path)
    #plt.show()
    result1,std1 = car2arm_gpr.gpr1.predict((xy_path),return_std=True)
    #print(std2)
    #result2 = car2arm_gpr.gpr1.predict(xy_path_down+np.array((0,0.1)),return_std=False)
    #print(std1.shape)
    arm_xyztheta_path = np.array(arm_xyztheta_path)
    #print(arm_xyztheta_path.shape)
    arm_xyztheta_path_down = np.array(arm_xyztheta_path_down)
    #ax1.plot3D(result1[:,0]+xy_path[:,1],result1[:,1]-xy_path[:,0],result1[:,2])
    #ax1.plot3D(result2[:,0],result2[:,1],result2[:,2])
    #ax1.plot3D(arm_xyztheta_path[:,0]+xy_path[:,1],arm_xyztheta_path[:,1]-xy_path[:,0],arm_xyztheta_path[:,2])
    ax1.scatter3D(arm_via[1,0],arm_via[1,1],arm_via[1,2],linewidth=5,color='springgreen')

    ax1.plot3D(arm_xyztheta_path[:,0],arm_xyztheta_path[:,1],arm_xyztheta_path[:,2],linewidth=3)
    #ax1.scatter3D(arm_via[:,0],arm_via[:,1],arm_via[:,2])
    print(arm_via[1,0],arm_via[1,1])
    print('std1',std1)
    std1 += 1e-8
    std2 += 1e-8
    std3 += 1e-8
    blend_std = 1/(1/std1+1/std2+1/std3)
    result4 = np.zeros((result2.shape[0],result2.shape[1]))
    for i in range(4):
        result4[:,i] = blend_std*(result1[:,i]/std1 + result2[:,i]/std2 + result3[:,i]/std3)
    #ax1.plot3D(result2[:,0]+xy_path[:,1],result2[:,1]-xy_path[:,0],result2[:,2])
    #ax1.plot3D(result1[:,0],result1[:,1],result1[:,2])
    ax1.plot3D(result4[:,0],result4[:,1],result4[:,2],linewidth=3)
    #ax1.plot3D(result3[:,0],result3[:,1],result3[:,2])
    plt.legend(('reference traj.','generalization traj.','strong via point','weak via point'),fontsize=14)
    fontdict={'family':'Times New Roman','size':14}

    plt.xlabel('x/m',fontdict= fontdict,labelpad=10)
    plt.ylabel('y/m',fontdict=fontdict,labelpad=10)
    ax1.set_zlabel('z/m',fontdict=fontdict,labelpad=10)

    #plt.xticks(fontproperties='Times New Roman',size=18)
    #plt.yticks(fontproperties='Times New Roman',size=18)  
    ax1.tick_params('z',labelsize=14)  
    ax1.tick_params('x',labelsize=14)  
    ax1.tick_params('y',labelsize=14)  
    ax1.grid(False)
    plt.show()
    #move slowly to the first disired point
    end_init_state = ur_end_record[-1]
    end_state = ur_end_record[-1]
    arm_state = ur_sub_record[-1]
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    robot_target = car2arm_gpr.gpr1.predict(np.array((current_x,current_y)).reshape(1,-1))

    #Speed Line
    def UR_SpeedL(ur5_end_vel,arm_state,time_from_start=0.01):
        J = ur5e_robot_Jacob_tool_length(arm_state)
        manipulability=sqrt(np.linalg.det(np.matmul(J,np.transpose(J))))
        if(manipulability < 0.01):
            print('manipulability:',manipulability)
        joint_vel = np.matmul(np.linalg.pinv(J),ur5_end_vel)
        point.positions = arm_state + joint_vel.ravel()
        point.time_from_start = rospy.Duration(1,0)
        msg.points = [point]
        ur_pub.publish(msg)

    ur5_end_vel = np.array((robot_target[0,0]+end_init_state[0]-end_state[0],robot_target[0,1]+end_init_state[1]-end_state[1],robot_target[0,2]+end_init_state[2]-end_state[2],0,0,robot_target[0,3]+end_init_state[5]-end_state[5])).reshape(-1,1)
    UR_SpeedL(ur5_end_vel,arm_state,time_from_start=1)
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

    while True:    
        #if near the target, break the loop
        if np.sqrt((current_x-2.8)**2+(current_y-2.8)**2) < 0.1:
            break
        t0 = time.time()

        #get the car next step
        result = car_gpr.get_next_step(np.array((current_x,current_y)).reshape(1,-1))
        target_angle = np.arctan2(result[0,1],result[0,0])
        delta_theta = np.arctan2(result[0,1],result[0,0])-current_theta
        
        #Bezier步长
        step_size = 0.2    
        if step < 0:#给扰动
            vel.linear.x = 0.5
            vel.angular.z = 0
            base_pub.publish(vel)
        else:
            #要限制最大的角速度为1rad/s，由于周期是0.1s，那么假如角度超过了0.1rad，就分开几段
            if np.abs(delta_theta) > 0.1 and bz_flag == False:        #假如角度相差不大，就直接走，否则走Bezier
                bz_flag = True
                bezier = BEZIER()
                #w_record = bezier.get_result(current_x,current_y,current_theta,current_x+step_size*np.cos(target_angle),current_y+step_size*np.sin(target_angle),current_theta+delta_theta,step_size=0.01)
                w_record = []
                while delta_theta > 0:
                    if delta_theta > 0.1:
                        w_record.append(0.1)
                    else:
                        w_record.append(delta_theta)
                    delta_theta -= 0.1

                if delta_theta < 0:
                    while delta_theta < 0:
                        if delta_theta < -0.1:
                            w_record.append(-0.1)
                        else:
                            w_record.append(delta_theta)
                        delta_theta += 0.1
                              
                print('delta theta:',delta_theta,'w:',len(w_record),'bezier')

            if bz_flag ==True:
                if bz_index < len(w_record):
                    vel.linear.x = 0.1
                    vel.angular.z = w_record[bz_index]*10.2
                    bz_index += 1
                else:
                    bz_flag = False
                    bz_index = 0
                    vel.linear.x = 0
                    vel.angular.z = 0
                base_pub.publish(vel)
            else:    
                #给乘2是为了加快整段的运行，实测精度差不多
                vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*2
                vel.angular.z = (np.arctan2(result[0,1],result[0,0])-current_theta)*10.2    #10.2是由延时0.1s决定的
                print(vel)
                base_pub.publish(vel)
        time.sleep(0.05)

#########################################################################################################################################        
        #机械臂控制
        robot_target = car2arm_gpr.gpr1.predict(np.array((current_x,current_y)).reshape(1,-1))
        ur5_end_vel = np.array((robot_target[0,0]+end_init_state[0]-end_state[0],robot_target[0,1]+end_init_state[1]-end_state[1],robot_target[0,2]+end_init_state[2]-end_state[2],0,0,robot_target[0,3]+end_init_state[5]-end_state[5])).reshape(-1,1)
        UR_SpeedL(ur5_end_vel,arm_state)
        if time.time()- t0 < 0.1:
            time.sleep(0.1-time.time()+t0)

        step +=1
        end_state = ur_end_record[-1]
        arm_state = ur_sub_record[-1]
        #current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
#########################################################################################################
        #在仿真中，机械臂的控制是位置控制，和延时多久没关系，但是车子是速度控制，所以要保证命令的间隔要在0.1s左右
        #协调控制，调整车子位姿.要更新好最新的位姿再开始控制  
        #要减去初始值才是输入GPR的输入吧
        coordinate_result = car2arm_gpr.find_nearest_point(np.array((end_state[0]-end_init_state[0],end_state[1]-end_init_state[1],end_state[2]-end_init_state[2],end_state[5]-end_init_state[5])).reshape(1,-1))
        print('car',np.argmin(np.linalg.norm((xy_path-np.array((current_x,current_y)).reshape(1,-1)),axis=1)))
        nearest_path = xy_path[np.argmin(np.linalg.norm((xy_path-np.array((current_x,current_y)).reshape(1,-1)),axis=1))]
        coordinate_result = coordinate_result.reshape(1,-1)
        nearest_path = nearest_path.reshape(1,-1)
        distance = np.sqrt((current_x-coordinate_result[0,0])**2+(current_y-coordinate_result[0,1])**2)
        print('current:',current_x,' ',current_y,'corrdinate:',coordinate_result,'distance:',distance)
        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]

        if(distance > 0.2 and bz_flag == False):
            #启动协调过程
            result = car_gpr.gpr1.predict(np.array((coordinate_result[0,0],coordinate_result[0,1])).reshape(1,-1))
            target_angle = atan2(result[0,1],result[0,0])
            delta_theta = target_angle - current_theta
            if(abs(atan2(coordinate_result[0,1] - current_y,coordinate_result[0,0] - current_x) - current_theta) < 0.1):#正着走
                print('corr1')
                vel.linear.x = distance*3
                vel.angular.z = (target_angle-current_theta)*3
                base_pub.publish(vel)
                time.sleep(0.2)

            elif(abs(atan2(-coordinate_result[0,1] + current_y,-coordinate_result[0,0] + current_x) - current_theta) < 0.1):#反着走   
                print('corr2')
                vel.linear.x = -distance*3
                vel.angular.z = (target_angle-current_theta)*3
                base_pub.publish(vel)
                time.sleep(0.2)

            elif(bz_flag):
                bezier = BEZIER()
                #w_record = bezier.get_result(current_x,current_y,current_theta,current_x+step_size*np.cos(target_angle),current_y+step_size*np.sin(target_angle),current_theta+delta_theta,step_size=0.01)
                w_record = []
                while delta_theta > 0:
                    if delta_theta > 0.1:
                        w_record.append(0.1)
                    else:
                        w_record.append(delta_theta)
                    delta_theta -= 0.1

                if delta_theta < 0:
                    while delta_theta < 0:
                        if delta_theta < -0.1:
                            w_record.append(-0.1)
                        else:
                            w_record.append(delta_theta)
                        delta_theta += 0.1
                              
                print('delta theta:',delta_theta,'w:',len(w_record),'bezier222')

    

    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)
    plt.figure()
    plt.plot([x[0] for x in car_sub_record],[x[1] for x in car_sub_record])
    plt.plot(xy_path[:,0],xy_path[:,1])
    plt.figure()

    plt.plot([x[0]-ur_end_record[0][0] for x in ur_end_record],[x[1]-ur_end_record[0][1] for x in ur_end_record])
    plt.plot([x[0]-ur_end_record[0][0] for x in ur_end_record],[x[2]-ur_end_record[0][2] for x in ur_end_record])
    plt.plot([x[0]-ur_end_record[0][0] for x in ur_end_record],[x[5]-ur_end_record[0][5] for x in ur_end_record])
    plt.plot([x[0] for x in arm_xyztheta_path],[x[1] for x in arm_xyztheta_path])
    plt.plot([x[0] for x in arm_xyztheta_path],[x[2] for x in arm_xyztheta_path])
    plt.plot([x[0] for x in arm_xyztheta_path],[x[3] for x in arm_xyztheta_path])
    plt.legend(('1','2','3','4','5','6'))

    plt.show()
    plt.pause(0)


