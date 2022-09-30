#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('.')
import rospy
import time
import random
import rtde_receive
import rtde_control
import threading
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from trajectory_msgs.msg import *
import seaborn as sns
from jacobi.ur5e_robot_Jacob_tool_length0145 import *
from jacobi.ur5e_car_kinematics_class_2 import UR5e_car_kinematics

Jiting = False  #emergency stop flag
IP = "192.168.176.224"

rtde_r = rtde_receive.RTDEReceiveInterface(IP)
for i in range(3):
    try:
        rtde_c = rtde_control.RTDEControlInterface(IP)
        break
    except Exception:
        time.sleep(3)
        print('keep trying to connect RTDE Control')
        if i == 2:
            sys.exit()


ur_end_record = []
car_record = []
ur_joint_record = []
ur_force_record = []
car_sub_record = []
JiTing=False
def press_enter_to_JiTing():#不是完全的急停
    global JiTing
    key=input()
    JiTing=True
    key=input()
    JiTing=True
    sys.exit()  #exit this input thread
listener=threading.Thread(target=press_enter_to_JiTing)
listener.start()



#car state subsciber callback function
def car_sub_callback(odom):
    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])


#End Effector Admmitance Control Implementation
def admittance_control(ef,m,d,x_dot,dt,speed_limit=0.2):
    k1=(ef-d*x_dot)/m
    k2=(ef-d*(x_dot+dt*k1))/m
    y_t0=x_dot
    y_t1=y_t0+0.5*(k1+k2)*dt

    for i in range(3):
        if abs(y_t1[i]) > speed_limit:
            y_t1[i]=abs(y_t1[i])/y_t1[i]*speed_limit
    return y_t1


if __name__ == "__main__":
    print('start the main function')

    #init ros node
    rospy.init_node("UR5Control")
    base_pub = rospy.Publisher ("/mobile_base/cmd_vel", Twist,queue_size=0) 
    base_sub = rospy.Subscriber("/mobile_base/odom",Odometry,car_sub_callback)
    ur5e_car = UR5e_car_kinematics()    #wholeBody Jacobi

    time.sleep(0.1) #For the node working

    #init the variable for the program
    vel = Twist()
    dt = 0.01
    step = 0
    np.set_printoptions(threshold=np.inf)
    V_pre = np.zeros(3)     #Admittance Velocity
    W_pre = np.zeros(3)     
    wholebody_pre_vel = np.zeros(6)
    car_pre_linear_vel = 0
    car_pre_angular_vel = 0
    
    #stop the car and move the UR to the initial pose
    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)
    joint_init=np.array([101.14, -86.93, 71.42, -74.12, -90.3, -76.5])/57.3
    joint_init=np.array([103.59, -97.73, 82.98, -80.83, -92.25, -73.92])/57.3

    rtde_c.moveJ(joint_init,0.5,0.1)

    #read the FT sensor data
    windows_size = 3
    F_raw = np.zeros((windows_size,6))
    rtde_c.zeroFtSensor()
    for i in range(windows_size):
        F_raw[i] = rtde_r.getActualTCPForce()
        time.sleep(0.05)


    #set the max velocity and acc limits 
    wholebody_linear_acc = 0.02
    wholebody_angular_acc = 0.05
    car_linear_acc = 0.02
    car_angular_acc = 0.05
    car_linear_vel = 0.4
    car_angular_vel = 0.48

    #set the force sensor limits
    force_min = 3
    force_max = 12
    torge_min = 0.05
    torge_max = 3

    #Manually set the gripper to the door bar
    rtde_c.teachMode()
    while JiTing == False:
        print(rtde_r.getActualTCPForce())
        time.sleep(0.5)

    JiTing = False
    rtde_c.endTeachMode()

    JiTing = False
    while JiTing == False:
        if rtde_c.isProgramRunning() == False:
            print('RTDE C error')
            break
        #read wholebody state
        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
        joint = rtde_r.getActualQ()
        F_raw[step % windows_size] = rtde_r.getActualTCPForce()
        F_filtered = np.mean(F_raw,axis=0)

        #if the car is running, record the wholebody state
        if abs(car_sub_record[-1][3]) > 0.005:
            car_record.append(car_sub_record[-1])
            ur_joint_record.append(joint)
            ur_end_record.append(rtde_r.getActualTCPPose())
            ur_force_record.append(F_filtered)
        
        #force coordinate transfomation  Base->World
        B_0_R = np.array([[0,-1,0],
                        [1,0,0],
                        [0,0,1]])
        W_B_R = np.array([[cos(current_theta),-sin(current_theta),0],
        [sin(current_theta),cos(current_theta),0],
        [0,0,1]])
        W_0_R = np.matmul(W_B_R,B_0_R)
        W_Force = np.matmul(W_0_R,F_filtered[:3])   #Force in world frame
        W_Torge = np.matmul(W_0_R,F_filtered[3:])   #Torge in world frame

        for i in range(3):
            if abs(W_Force[i]) < force_min:
                W_Force[i] = 0
            elif abs(W_Force[i]) > force_max:
                W_Force[i] = force_max*np.sign(W_Force[i])
        
        for i in range(3):
            if abs(W_Torge[i]) < torge_min:
                W_Torge[i] = 0
            if abs(W_Torge[i]) > torge_max:
                W_Torge[i] = torge_max*np.sign(W_Torge[i])
        
        #Admittance Controller
        V_pre = admittance_control(W_Force,3.2,100,V_pre,0.01,speed_limit=0.2)
        W_pre = admittance_control(W_Torge,0.01,0.5,W_pre,0.01,speed_limit=0.4)
        W_pre[0]=0  #disable the roll and pitch
        W_pre[1]=0
        wholebody_vel = np.hstack((V_pre,W_pre))
        
        #wholebody speed limits
        for i in range(3):
            if abs(wholebody_vel[i] - wholebody_pre_vel[i]) >  wholebody_linear_acc*dt:
                wholebody_vel[i] = wholebody_pre_vel[i] + np.sign(wholebody_vel[i] - wholebody_pre_vel[i])*wholebody_linear_acc*dt
        for i in range(3):
            if abs(wholebody_vel[i+3] - wholebody_pre_vel[i+3]) > wholebody_linear_acc*dt:
                wholebody_vel[i+3] = wholebody_pre_vel[i+3] + np.sign(wholebody_vel[i+3] - wholebody_pre_vel[i+3])*wholebody_linear_acc*dt
        
        #differential inverse kinematics
        wholebody_J = ur5e_car.get_jacobian_lx([current_x,current_y,current_theta] + joint)
        Weight=np.array([[1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])
        z=[0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        J_transpose=np.transpose(wholebody_J)
        Weignt_inverse=np.linalg.pinv(Weight)
        J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, np.linalg.pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
        joint_v=np.matmul(J_sharp,wholebody_vel)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)

        #car speed limits
        if abs(joint_v[0] - car_pre_linear_vel) > car_linear_acc*dt:
            joint_v[0] = car_pre_linear_vel + car_linear_acc*dt*np.sign(joint_v[0] - car_pre_linear_vel)
        if abs(joint_v[1] - car_pre_angular_vel) > car_angular_acc*dt:
            joint_v[1] = car_pre_angular_vel + car_angular_acc*dt*np.sign(joint_v[1] - car_pre_angular_vel) 
        
        if abs(joint_v[0]) > car_linear_vel:
            joint_v[0] = car_linear_vel*np.sign(joint_v[0])      
        if abs(joint_v[1]) > car_angular_vel:
            joint_v[1] = car_angular_vel*np.sign(joint_v[1])                     
        vel.linear.x = joint_v[0]
        vel.angular.z = joint_v[1]


        if step%10 ==0:
            print('filterd',F_filtered)
            print('vel',wholebody_vel)
        car_pre_linear_vel = vel.linear.x
        car_pre_angular_vel = vel.angular.z
        rtde_c.speedJ(joint_v[2:].tolist(),0.5,0.002)
        base_pub.publish(vel)

        wholebody_pre_vel = wholebody_vel
        time.sleep(dt)
        step += 1      
    rtde_c.speedStop()
    ur_joint_record = np.array(ur_joint_record)
   
    np.savetxt('car_open_door40.txt',np.array(car_record)[:,:])
    np.savetxt('ur_open_door40.txt',np.array(ur_end_record)[:,:])

    print('end')
    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)





