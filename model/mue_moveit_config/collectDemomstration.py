#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from email.mime import base
from trajectory_msgs.msg import *
from control_msgs.msg import *
import rospy
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
import time
from geometry_msgs.msg import Twist
from control_msgs.msg import JointTrajectoryControllerState
from nav_msgs.msg import Odometry
import numpy as np
import math

#图像显示设置
ur_sub_cnt = 0
ur_fig_refresh = True
car_sub_cnt = 0
car_fig_refresh = True
ur_sub_record = []
car_sub_xytheta_dot_record = []



#机械臂状态接收回调函数
def ur_sub_callback(state):
    global ur_sub_cnt
    global ur_fig_refresh
    ur_sub_cnt += 1
    if ur_sub_cnt > 0 :
        ur_sub_cnt = 0
        ur_fig_refresh = True
        ur_sub_record.append(list(state.actual.positions))
#车子状态接收回调函数
def car_sub_callback(odom):
    global car_sub_cnt
    global car_fig_refresh
    car_sub_cnt +=1
    if car_sub_cnt > 0:
        car_sub_cnt = 0
        car_fig_refresh = True
        x = odom.pose.pose.orientation.x
        y = odom.pose.pose.orientation.y
        z = odom.pose.pose.orientation.z
        w = odom.pose.pose.orientation.w
        theta = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
        car_sub_xytheta_dot_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])


#输入n*3的一个位置数组，输出一个n*3的差分向量，用于速度
def get_velocity(position):
    rows,cols = position.shape
    velocity = np.zeros((rows,cols))
    for i in range(rows - 1):
        velocity[i] = position[i+1] - position[i]
    velocity[rows-1] = velocity[rows-2]

    return velocity


if __name__ == "__main__":
    #初始化ros节点
    rospy.init_node("UR5Control")
    ur_pub = rospy.Publisher ("/arm_controller/command", JointTrajectory,queue_size=10)  
    base_pub = rospy.Publisher ("/mobile_base_controller/cmd_vel", Twist,queue_size=10)  
    ur_sub = rospy.Subscriber("/arm_controller/state",JointTrajectoryControllerState,ur_sub_callback)
    base_sub = rospy.Subscriber("/mobile_base_controller/odom",Odometry,car_sub_callback)
    
    
    msg = JointTrajectory()
    vel = Twist()

    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()
    step = 0
    while True:
        tt0 = time.time()
        if step > 200:
            break
        #print(step)

        if(step >= 0 and step < 100):

            vel.linear.x = 3.1415926/2/10
            vel.angular.z = -3.1415926/2/10
            base_pub.publish(vel)
        elif(step >=100 and step <= 200):
            vel.linear.x = 3.1415926/2/10
            vel.angular.z = 3.1415926/2/10
            base_pub.publish(vel)
        step +=1
        print(step)
        time.sleep(0.1+tt0-time.time())
    np.savetxt('car_xytheta_dot_record.txt',np.array(car_sub_xytheta_dot_record)[20:,:])
