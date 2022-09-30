#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from trajectory_msgs.msg import *
from control_msgs.msg import *
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from geometry_msgs.msg import Twist
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
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])

#read demonstration data
def readTxt():
    data = np.loadtxt('data/car_xytheta_dot_record1.txt')
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


    def get_result(self,p0x,p0y,initA,p3x,p3y,endA,step_size=0.01):
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
    #init ros node
    rospy.init_node("UR5Control")
    ur_pub = rospy.Publisher ("/arm_controller/command", JointTrajectory,queue_size=0)  
    base_pub = rospy.Publisher ("/mobile_base_controller/cmd_vel", Twist,queue_size=0) 
    ur_sub = rospy.Subscriber("/arm_controller/state",JointTrajectoryControllerState,ur_sub_callback)
    base_sub = rospy.Subscriber("/mobile_base_controller/odom",Odometry,car_sub_callback)
    print('init success')
    time.sleep(0.5)


    #prepare for the ros publish 
    msg = JointTrajectory()
    vel = Twist()
    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()


    #Bezier Culve generate
    bezier1 = BEZIER()
    p0x,p0y,p3x,p3y = 0,0,0.5,0
    w_record = bezier1.get_result(p0x,p0y,0,p3x,p3y,0,step_size=0.005)
    bezier1.plot_bezier()

    step = 1
    vel.linear.x = 0
    base_pub.publish(vel)

    time.sleep(0.5)

    #set the UR to initial pose
    joint_init=np.array([-90, -90, -90, -90, 90, -90])/57.3
    point.positions = joint_init
    point.time_from_start = rospy.Duration(1,0)
    msg.points = [point]
    msg.header.seq = step
    ur_pub.publish(msg)
    time.sleep(2)

    #car rotate to a specific angel
    # for i in range(20):
    #     vel.angular.z = pi/2/4
    #     base_pub.publish(vel)
    #     time.sleep(0.2)
    # vel.angular.z = 0
    # base_pub.publish(vel)


    #clear the record data
    ur5_current_joint_position = ur_sub_record[-1]
    car_sub_record.clear()
    ur_sub_record.clear()
    time.sleep(0.1)



    while True:
        tt0 = time.time()
        
        if step >= int((len(w_record))):
            break

        vel.linear.x = 0.05#0.1015
        vel.angular.z = w_record[step]*10.4
        base_pub.publish(vel)

        step +=1
        time.sleep(0.1+tt0-time.time())
        ur5_current_joint_position = ur_sub_record[-1]

    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)
    time.sleep(0.2)



    plt.plot([x[0] for x in car_sub_record],[x[1] for x in car_sub_record])


    np.savetxt('data/car_xytheta_dot_record10.txt',np.array(car_sub_record)[20:,:])

    plt.show()
    plt.pause(0)


