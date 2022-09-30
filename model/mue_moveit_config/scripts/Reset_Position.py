#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from pyparsing import Forward
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from math import cos, acos, sin, sqrt, exp, atan, atan2, pi
from numpy.linalg import norm, inv, det, cond, pinv
import math
import time
from ur5e_car_kinematics_class_2 import UR5e_car_kinematics
import matplotlib.pyplot as plt
from matplotlib import pyplot
car_sub_record = []
#车子状态接收回调函数
def car_sub_callback(odom):
    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])

if __name__ == "__main__":
    #初始化ros节点
    rospy.init_node("UR5Control")
    base_pub = rospy.Publisher ("/mobile_base_controller/cmd_vel", Twist,queue_size=0) 
    base_sub = rospy.Subscriber("/mobile_base_controller/odom",Odometry,car_sub_callback)

    vel = Twist()
    np.set_printoptions(threshold=np.inf)
    vel.linear.x = 0
    base_pub.publish(vel)
    bz_flag = False
    Forward = True
    time.sleep(0.1)
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]

    target_angle = atan2(current_y,current_x)
    print('target angle',target_angle)
    #第一阶段 转弯
    while True:
        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
        print('step1:','current theta',current_theta)

        if abs(target_angle + pi - current_theta) < 0.01 or abs(target_angle - current_theta) < 0.01:
            break

        if abs(target_angle - current_theta) <= abs(target_angle + pi - current_theta):
            if abs(target_angle - current_theta) < 0.1:
                ang = abs(target_angle - current_theta)
            else:
                ang = 0.1
            vel.linear.x = 0
            vel.angular.z = ang*np.sign(target_angle - current_theta)
            Forward = False
        else:
            if abs(target_angle+ pi - current_theta) < 0.1:
                ang = abs(target_angle + pi - current_theta)
            else:
                ang = 0.1
            vel.linear.x = 0
            vel.angular.z = ang*np.sign(target_angle + pi - current_theta)
            Forward = True

        
        base_pub.publish(vel)
        time.sleep(0.1)
    #第二阶段  位移
    while True:
        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
        distance = np.sqrt((current_x)**2+(current_y)**2)

        print('step2:','distance:',distance)
        if distance < 0.05:
            break
        if Forward:
            if distance < 0.1:
                vel.linear.x = distance
            else:
                vel.linear.x = 0.1
            vel.angular.z = 0
        else:
            if distance < 0.1:
                vel.linear.x = -distance
            else:
                vel.linear.x = -0.1
            vel.angular.z = 0
        base_pub.publish(vel)
        time.sleep(0.1)

    #第三阶段 回正
    target_angle = 0
    while True:
        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
        print('step3:','current theta',current_theta)

        if abs(target_angle + pi - current_theta) < 0.01 or abs(target_angle - current_theta) < 0.01:
            break

        if abs(target_angle - current_theta) <= abs(target_angle + pi - current_theta):
            if abs(target_angle - current_theta) < 0.1:
                ang = abs(target_angle - current_theta)
            else:
                ang = 0.1
            vel.linear.x = 0
            vel.angular.z = ang*np.sign(target_angle - current_theta)
        else:
            if abs(target_angle+ pi - current_theta) < 0.1:
                ang = abs(target_angle + pi - current_theta)
            else:
                ang = 0.1
            vel.linear.x = 0
            vel.angular.z = ang*np.sign(target_angle + pi - current_theta)

        
        base_pub.publish(vel)
        time.sleep(0.1)
   
