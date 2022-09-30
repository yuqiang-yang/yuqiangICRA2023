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
from mpl_toolkits.mplot3d import Axes3D
from ur5e_robot_Jacob_tool_length0145 import *
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
#输入dt step 速度和Z  输出理论和仿真轨迹
def find_error(dt,mystep,velocity_X,z):
    global ur5_kin
    joint=[2.139, -2.14056, 1.9454, -2.940527, 4.1446571, -3.13154]
    ur_end_record = []
    theroy_end_record = []

    car_x,car_y,car_theta = 0,0,0
    step = 0
    ee_pose0 = ur5_kin.forward(np.array(joint))
    ee_pose0 = np.asarray(ee_pose0).reshape(3,4) # 3x4 rigid transformation matrix
    theroy_end = [0,0,0,0,0,0]
    theroy_end_record.append([0,0,0,0,0,0])
    ur_end_record.append([0,0,0])
    while True:
        if step >= mystep:
            break

        t0 = time.time()
        wholeBodyState = [car_x,car_y,car_theta] + joint #car_sub_record[-1][0:3]+ur_sub_record[-1]
        #print(wholeBodyState[3:])
        wholebody_J = ur5e_car.get_jacobian_lx(wholeBodyState)
        Weight=np.array([[99999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 99999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        J_transpose=np.transpose(wholebody_J)
        Weignt_inverse=np.linalg.pinv(Weight)
        J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, np.linalg.pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
        joint_v=np.matmul(J_sharp,velocity_X)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)
        
        # J_robot = ur5e_robot_Jacob_tool_length(joint)
        # joint_v = np.matmul(np.linalg.pinv(J_robot),velocity_X)
        # joint_v = np.hstack((np.array((0,0)),joint_v))
        #print(joint_v)

        car_x = car_x+joint_v[0]*dt
        car_theta = car_theta+joint_v[1]*dt
        joint[0] = joint[0] + joint_v[2]*dt
        joint[1] = joint[1] + joint_v[3]*dt
        joint[2] = joint[2] + joint_v[4]*dt
        joint[3] = joint[3] + joint_v[5]*dt
        joint[4] = joint[4] + joint_v[6]*dt
        joint[5] = joint[5] + joint_v[7]*dt
        ee_pose = ur5_kin.forward(np.array(joint))
        ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
        theroy_end = theroy_end.copy()
        for i in range(6):
            theroy_end[i] = theroy_end[i] + velocity_X[i]*dt
        
        theroy_end_record.append(theroy_end)
        ur_end_record.append([-ee_pose[1,3]+ee_pose0[1,3]+car_x,ee_pose[0,3]-ee_pose0[0,3],ee_pose[2,3]-ee_pose0[2,3]])
        #ur_end_record.append([ee_pose[0,3]-ee_pose0[0,3],ee_pose[1,3]-ee_pose0[1,3],ee_pose[2,3]-ee_pose0[2,3]])
        step +=1    


    return np.array(theroy_end_record),np.array(ur_end_record)
if __name__ == "__main__":
    dt = 0.01
    step = 2/dt
    theroy_end_record,ur_end_record = find_error(dt,step,[0.2,0,0,0,0,0],[0,0,0,0,0,0,0,0])


    dt = 0.025
    step = 2/dt
    theroy_end_record1,ur_end_record1 = find_error(dt,step,[0.2,0,0,0,0,0],[0,0,0,0,0,0,0,0])


    dt = 0.05
    step = 2/dt
    theroy_end_record2,ur_end_record2 = find_error(dt,step,[0.2,0,0,0,0,0],[0,0,0,0,0,0,0,0])


    dt = 0.1
    step = 2/dt
    theroy_end_record3,ur_end_record3 = find_error(dt,step,[0.2,0,0,0,0,0],[0,0,0,0,0,0,0,0])

    plt.figure()
    plt.subplot(121)
    plt.plot(np.linalg.norm(theroy_end_record[:,:3]-ur_end_record,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record1[:,:3]-ur_end_record1,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record2[:,:3]-ur_end_record2,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record3[:,:3]-ur_end_record3,axis=1))
    plt.legend(('dt=0.01','dt=0.025','dt=0.05','dt=0.1'))
    plt.title('unweighted,dt->error,run 0.2m/s in x directions for 2 seconds')
    plt.subplot(122)
    plt.plot(theroy_end_record[:,:3])
    plt.plot(ur_end_record)
    plt.title('unweighted,result path')
    plt.legend(('sim x','sim y','sim z','real x','real y','real z'))

    dt = 0.025
    vel = 0.01
    step = 20/vel
    theroy_end_record,ur_end_record = find_error(dt,step,[vel,0,0,0,0,0],[0,0,0,0,0,0,0,0])


    vel = 0.02
    step = 20/vel
    theroy_end_record1,ur_end_record1 = find_error(dt,step,[vel,0,0,0,0,0],[0,0,0,0,0,0,0,0])

    vel = 0.1
    step = 20/vel
    theroy_end_record2,ur_end_record2 = find_error(dt,step,[vel,0,0,0,0,0],[0,0,0,0,0,0,0,0])


    vel = 0.5
    step = 20/vel
    theroy_end_record3,ur_end_record3 = find_error(dt,step,[vel,0,0,0,0,0],[0,0,0,0,0,0,0,0])

    plt.figure()
    plt.subplot(121)
    plt.plot(np.linalg.norm(theroy_end_record[:,:3]-ur_end_record,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record1[:,:3]-ur_end_record1,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record2[:,:3]-ur_end_record2,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record3[:,:3]-ur_end_record3,axis=1))
    plt.legend(('vel=0.01','vel=0.02','vel=0.1','dt=vel=0.5'))
    plt.title('unweighted,vel->error,run 0.05m in x directions')
    plt.subplot(122)
    plt.plot(theroy_end_record[:,:3])
    plt.plot(ur_end_record)
    plt.title('unweighted,result path')
    plt.legend(('sim x','sim y','sim z','real x','real y','real z'))





    dt = 0.01
    step = 2/dt
    theroy_end_record,ur_end_record = find_error(dt,step,[0,0,-0.2,0,0,0],[0,0,0,0,0,0,0,0])


    dt = 0.025
    step = 2/dt
    theroy_end_record1,ur_end_record1 = find_error(dt,step,[0,0,-0.2,0,0,0],[0,0,0,0,0,0,0,0])


    dt = 0.05
    step = 2/dt
    theroy_end_record2,ur_end_record2 = find_error(dt,step,[0,0,-0.2,0,0,0],[0,0,0,0,0,0,0,0])


    dt = 0.1
    step = 2/dt
    theroy_end_record3,ur_end_record3 = find_error(dt,step,[0,0,-0.2,0,0,0],[0,0,0,0,0,0,0,0])

    plt.figure()
    plt.subplot(121)
    plt.plot(np.linalg.norm(theroy_end_record[:,:3]-ur_end_record,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record1[:,:3]-ur_end_record1,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record2[:,:3]-ur_end_record2,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record3[:,:3]-ur_end_record3,axis=1))
    plt.legend(('dt=0.01','dt=0.025','dt=0.05','dt=0.1'))
    plt.title('unweighted,dt->error,run 0.2m/s in z directions for 2 seconds')
    plt.subplot(122)
    plt.plot(theroy_end_record[:,:3])
    plt.plot(ur_end_record)
    plt.title('unweighted,result path')
    plt.legend(('sim x','sim y','sim z','real x','real y','real z'))

    






    dt = 0.025
    vel = 0.01
    step = 20/vel
    theroy_end_record,ur_end_record = find_error(dt,step,[0,0,-vel,0,0,0],[0,0,0,0,0,0,0,0])



    vel = 0.02
    step = 20/vel
    theroy_end_record1,ur_end_record1 = find_error(dt,step,[0,0,-vel,0,0,0],[0,0,0,0,0,0,0,0])

    vel = 0.1
    step = 20/vel
    theroy_end_record2,ur_end_record2 = find_error(dt,step,[0,0,-vel,0,0,0],[0,0,0,0,0,0,0,0])


    vel = 0.5
    step = 20/vel
    theroy_end_record3,ur_end_record3 = find_error(dt,step,[0,0,-vel,0,0,0],[0,0,0,0,0,0,0,0])

    plt.figure()
    plt.subplot(121)
    plt.plot(np.linalg.norm(theroy_end_record[:,:3]-ur_end_record,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record1[:,:3]-ur_end_record1,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record2[:,:3]-ur_end_record2,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record3[:,:3]-ur_end_record3,axis=1))
    plt.legend(('vel=0.01','vel=0.02','vel=0.1','vel=0.5'))
    plt.title('unweighted,vel->error,run 0.5m in z directions')
    plt.subplot(122)
    plt.plot(theroy_end_record[:,:3])
    plt.plot(ur_end_record)
    plt.title('unweighted,result path')
    plt.legend(('sim x','sim y','sim z','real x','real y','real z'))





    dt = 0.01
    step = 2/dt
    theroy_end_record,ur_end_record = find_error(dt,step,[0,0,0,0,0,0],[-0.2,0,0,0,0,0,0,0])


    dt = 0.025
    step = 2/dt
    theroy_end_record1,ur_end_record1 = find_error(dt,step,[0,0,0,0,0,0],[-0.2,0,0,0,0,0,0,0])


    dt = 0.05
    step = 2/dt
    theroy_end_record2,ur_end_record2 = find_error(dt,step,[0,0,0,0,0,0],[-0.2,0,0,0,0,0,0,0])


    dt = 0.1
    step = 2/dt
    theroy_end_record3,ur_end_record3 = find_error(dt,step,[0,0,0,0,0,0],[-0.2,0,0,0,0,0,0,0])

    plt.figure()
    plt.subplot(121)
    plt.plot(np.linalg.norm(theroy_end_record[:,:3]-ur_end_record,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record1[:,:3]-ur_end_record1,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record2[:,:3]-ur_end_record2,axis=1))
    plt.plot(np.linalg.norm(theroy_end_record3[:,:3]-ur_end_record3,axis=1))
    plt.legend(('dt=0.01','dt=0.025','dt=0.05','dt=0.1'))
    plt.title('unweighted,dt->error,stop for 2 seconds(car 0.2m/s)')
    plt.subplot(122)
    plt.plot(theroy_end_record[:,:3])
    plt.plot(ur_end_record)
    plt.title('unweighted,result path')
    plt.legend(('sim x','sim y','sim z','real x','real y','real z'))


    plt.show()


