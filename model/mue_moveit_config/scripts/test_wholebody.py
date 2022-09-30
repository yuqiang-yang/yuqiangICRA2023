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
from gazebo_msgs.msg import LinkStates

from nav_msgs.msg import Odometry
import numpy as np
import math
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import seaborn as sns
from ur5e_robot_Jacob_tool_length0145 import *
import random
import ikfastpy
from ur5e_car_kinematics_class_2 import UR5e_car_kinematics
ur5e_car = UR5e_car_kinematics()
ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()
ur_sub_record = []
ur_end_record = []
car_sub_record = []
gazebo_sub_record = []
car_refresh = True
gazebo_refresh = True
#记录车子和臂在初始位置的六维位姿，而后通过GP算出来的值都相对于这个值进行偏移，初始位置这个值是固定的。
#####可以实现一个反复扫的任务，这个任务感觉是有价值的。类似AGV的效果，但是不需要传感器
#避障也可以是扰动
#机械臂状态接收回调函数

gazebo_sub_cnt = 0
def gazebo_sub_callback(state):
    global gazebo_refresh
    if gazebo_refresh:
        gazebo_refresh = False
        gazebo_sub_record.append([state.pose[-2].position.x,state.pose[-2].position.y,state.pose[-2].position.z])
        #print(state.pose[-2].position)
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


#车子状态接收回调函数
def car_sub_callback(odom):
    global car_refresh
    if car_refresh:
        car_refresh = False
        x = odom.pose.pose.orientation.x
        y = odom.pose.pose.orientation.y
        z = odom.pose.pose.orientation.z
        w = odom.pose.pose.orientation.w
        theta = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
        car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])



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

    #初始化ros节点
    rospy.init_node("UR5Control")
    ur_pub = rospy.Publisher ("/arm_controller/command", JointTrajectory,queue_size=0)  
    base_pub = rospy.Publisher ("/mobile_base_controller/cmd_vel", Twist,queue_size=0) 
    ur_sub = rospy.Subscriber("/arm_controller/state",JointTrajectoryControllerState,ur_sub_callback)
    base_sub = rospy.Subscriber("/mobile_base_controller/odom",Odometry,car_sub_callback)
    gazebo_sub = rospy.Subscriber("/gazebo/link_states",LinkStates,gazebo_sub_callback)
    time.sleep(0.3)
    #准备发送数据
    msg = JointTrajectory()
    vel = Twist()

    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint','wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()

    #先给车发送一个指令，不然第一次指令无法响应
    vel.linear.x = 0
    base_pub.publish(vel)
    joint_init=[2.1394758224487305, -2.140566965142721, 1.9454715887652796, -2.9405275783934535, 4.144657135009766-2*pi, -3.131549660359518]
    #joint_init = [ 0.21819526 ,-1.00794291, -1.94259298, -0.16764699, -0.21731699,  3.12555647]
    this_joint = joint_init
    point.positions = joint_init
    point.time_from_start = rospy.Duration(0.5,0)
    msg.points = [point]
    msg.header.seq = 0
    ur_pub.publish(msg)
    time.sleep(1)
    point = JointTrajectoryPoint()

    #设置当前车子状态
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    arm_state = ur_sub_record[-1]
    end_state = ur_end_record[-1]
    #end_init_state = ur_end_record[-1]
    car_sub_record.clear()
    ur_sub_record.clear()
    ur_end_record.clear()
    gazebo_sub_record.clear()
    step = 0
    bz_flag = False
    bz_index = 0
    car_refresh = True
    gazebo_refresh = True
    time.sleep(0.5)
####################################################################################################################################################
    while True:
        tt0 = time.time()
        if step >= 100: 
            break
        if step %1 == 0:
            car_refresh = True
            gazebo_refresh = True
        t0 = time.time()
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
        velocity_X=[0.55, 0, 0, 0,0,0]
        z=[0.5, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        J_transpose=np.transpose(wholebody_J)
        Weignt_inverse=np.linalg.pinv(Weight)

        J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, np.linalg.pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
        joint_v=np.matmul(J_sharp,velocity_X)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)
        
        
        vel.linear.x = joint_v[0]
        vel.angular.z = joint_v[1]
        base_pub.publish(vel)
        
        point.positions = arm_state + joint_v[2:]*0.1
        print(joint_v)
        point.velocities = joint_v[2:]
        point.time_from_start = rospy.Duration(0.01,0)
        msg.points = [point]
        msg.header.seq = step
        ur_pub.publish(msg)



        step +=1
        if time.time()-t0 <= 0.1:
            time.sleep(0.1-time.time()+t0)
        else:
            print(time.time()-t0)
        end_state = ur_end_record[-1]
        arm_state = ur_sub_record[-1]
        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]

    

    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)
    plt.figure()
    plt.plot([g[0]-gazebo_sub_record[0][0] for g in gazebo_sub_record])
    plt.plot([g[1]-gazebo_sub_record[0][1] for g in gazebo_sub_record])
    plt.plot([g[2]-gazebo_sub_record[0][2] for g in gazebo_sub_record])
    plt.plot([c[0]-car_sub_record[0][0] for c in car_sub_record])
    plt.plot([c[1]-car_sub_record[0][1] for c in car_sub_record])
    plt.plot([c[2]-car_sub_record[0][2] for c in car_sub_record])



    plt.legend(('1','2','3','4','5','6'))
    #np.savetxt('car_xytheta_dot_record2.txt',np.array(car_sub_record)[:,:])

    plt.show()
    plt.pause(0)


