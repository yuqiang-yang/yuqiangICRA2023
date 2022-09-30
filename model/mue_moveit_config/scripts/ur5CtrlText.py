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
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
pi = 3.1415926
from ur5e_robot_Jacob_tool_length0145 import *
#图像显示设置
ur_sub_cnt = 0
ur_fig_refresh = False
car_sub_cnt = 0
car_fig_refresh = False
ur_sub_record = []
car_sub_record = []

#1.加不加柔顺？   
#  机械臂单单用导纳控制器
#2.当机械臂收到外力偏移时，如何回正？     
#  可以曲线救国，比如收到外力扰动时，往左拉，那么期望位置与当前位置有相差了一个距离之后，则判断需要回正，需要产生协调作用了
#  根据机械臂的当前位置找到机械臂数据库中最近的点对应的那个车子的位置（这一步是否可以通过再训练一个GP实现？），然后车子回正
#  到那个位置。回到了之后，此时通过了机械臂的GP预测出来的位置就是受到了扰动之后的位置，而后车子继续朝新的一个位置移动，然后
#  又受到了阻碍，继续回正。
#  假如外力此时消失了之后，由于车子最终会回到轨道的位置，那么最终臂也会回到期望的位置，可以正常开门
#3.当车子自己有扰动的时候？
#  显然以车子为核心的，车子受扰动之后臂自动会响应车子的扰动
#4.若只有柔顺，没有这一项回正，会怎样？
#  假如车的运行不考虑臂偏不偏离的话，那显然会撞到門，换到测到的误差来说的话，那会越来越大
#5.具体的步骤？
#  先给一条车子的轨迹，训练出一个车子的场后。再复现车子的轨迹，控制臂的轨迹。
#  训练三个高斯过程，一个车子位置->速度，一个车子位置->机械臂位置，一个机械臂位置->车子位置
#  每一步读取当前车子的X Y值，输出车子的运动速度Xdot Ydot（这一部分考虑回正项，就是要做好方差那一步，这边假如为了区别对待的
#  话，可以考虑增加一项长期的回正来避免局部最小点），保证车子最终一定是能收敛于轨迹的。然后通过X Y值输出机械臂位置，作为机械臂
#  导纳控制器的平衡点。然后每一步需要判断机械臂期望值与当前值偏差（这里面涉及到如何求姿态的误差问题，感觉不太能直接求二范数），
#  假如偏差大于了一定的值，那么则开始协调的过程。通过高斯过程反算出扰动后的点对应的车子点并回正。这一项怎么改成连续和光滑的，
#  阈值这种不稳定而且式子不好看。可以在主循环中加一步，每次，车走，臂走之后，下一步就是协调，再走。
#6.如何实现泛化？ 不同把手 不同門 这是创新点
#  涉及三个GP的变换。 不同門，这个只是一个记录过程，硬要说可以泛化也可以，比如我一家只有这样一种門，那确实可以只考虑这一种轨迹。
#  在实际的应用中，可能只用学这个协调关系的正反两个GP，然后底层车子的运动或者整个3D避障就用其他方法弄。
#  不同的門把手，那么这里就得考虑到Z轴的路径，那位置的话整个XYZ都要学习和泛化，这边给四维，XYZ加上一一个轴的转动。通过加中间点应该就好？
#7.如何交互式学习/主动学习
#   交互式即是删点加点，主动再说把

#机械臂状态接收回调函数
def ur_sub_callback(state):
    global ur_sub_cnt
    global ur_fig_refresh
    ur_sub_cnt += 1
    if ur_sub_cnt > 0:
        ur_sub_cnt = 0
        ur_sub_record.append(list(state.actual.positions))

#车子状态接收回调函数
def car_sub_callback(odom):
    global car_sub_cnt
    global car_fig_refresh
    car_sub_cnt +=1
    if car_sub_cnt > 0:
        car_sub_cnt = 0
        x = odom.pose.pose.orientation.x
        y = odom.pose.pose.orientation.y
        z = odom.pose.pose.orientation.z
        w = odom.pose.pose.orientation.w
        theta = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
        car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])


#输入n*3的一个位置数组，输出一个n*3的差分向量，用于速度
def get_velocity(position):
    rows,cols = position.shape
    velocity = np.zeros((rows,cols))
    for i in range(rows - 1):
        velocity[i] = position[i+1] - position[i]
    velocity[rows-1] = velocity[rows-2]

    return velocity


def generate_curve():
    path = np.zeros([400,2])
    for i in range(100):
        path[i,:] = np.array([0,i*0.01])
    for i in range(100):
        path[100+i,:] = np.array([0.01*i,1+np.sqrt(1-(0.01*i-1)**2)])
    for i in range(100):
        path[200+i,:] = np.array([1+0.01*i,3-np.sqrt(1-(0.01*i)**2)])
    for i in range(100):
        path[300+i,:] = np.array([2,3+i*0.01])

    return path

def readTxt():
    data = np.loadtxt('car_xytheta_dot_record1.txt')
    xy_path = data[:,0:2]
    xy_dot_path = np.concatenate((data[:,3].reshape(-1,1)*np.cos(data[:,2].reshape(-1,1)),data[:,3].reshape(-1,1)*np.sin(data[:,2]).reshape(-1,1)),axis=1)
    xy_theta_path = data[:,0:3]
    xy_theta_dot_path = np.concatenate((xy_dot_path,data[:,5].reshape(-1,1)),axis=1)

    return xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path

def mean_filter(data):
    rows, cols = data.shape
    filtedData = np.zeros((rows,cols))
    windows_size = 4
    # b, a = signal.butter(8, 0.2, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
    # for i in range(cols):
    #     filtedData[:,i] = signal.filtfilt(b, a, data[:,i])  #data为要过滤的信号
    windows = np.zeros((windows_size,cols))
    for i in range(windows_size):
        windows[i,:] = data[0,:]
    
    for i in range(rows):
        windows[i%windows_size,:] = data[i,:]
        filtedData[i,:] = np.sum(windows,axis=0)/windows_size
    return filtedData

    
def down_sample(data,times):
    rows, cols = data.shape
    down_data = np.zeros((int(rows//times),cols))
    for i in range(int(rows//times)):
        down_data[i,:] = data[i*times,:]

    return down_data




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
        print(p1x,p1y,p2x,p2y, bezier_total_length)
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

def stop_here():

    while True:
        plt.pause(1)
        time.sleep(1)

if __name__ == "__main__":
    #初始化ros节点
    rospy.init_node("UR5Control")
    ur_pub = rospy.Publisher ("/arm_controller/command", JointTrajectory,queue_size=0)  
    base_pub = rospy.Publisher ("/mobile_base_controller/cmd_vel", Twist,queue_size=0) 
    ur_sub = rospy.Subscriber("/arm_controller/state",JointTrajectoryControllerState,ur_sub_callback)
    base_sub = rospy.Subscriber("/mobile_base_controller/odom",Odometry,car_sub_callback)
    
    time.sleep(0.5)
    msg = JointTrajectory()
    vel = Twist()

    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()

    bezier1 = BEZIER()
    bezier2 = BEZIER()
    p0x,p0y,p3x,p3y = 0,0,1,1
    p6x,p6y = 6,6
    w_record = bezier1.get_result(p0x,p0y,0,p3x,p3y,0)
    bezier1.plot_bezier()

    step = 1
    vel.linear.x = 0
    base_pub.publish(vel)

    time.sleep(0.5)

    ur5_end_vel = np.array((0,0,0,0,0,0.05)).reshape(-1,1)
    

    joint_init=[2.1394758224487305, -2.140566965142721, 1.9454715887652796, -2.9405275783934535, 4.144657135009766, -3.131549660359518]

    #point.positions = [0,-1.63,1.32,0,0,0]
    point.positions = joint_init
    #point.velocities = [0,0,0,0,0,0]  #joint_vel.ravel()
    point.time_from_start = rospy.Duration(1,0)
    msg.points = [point]
    msg.header.seq = step
    ur_pub.publish(msg)
    time.sleep(2)

    ur5_current_joint_position = ur_sub_record[-1]
    car_sub_record.clear()
    ur_sub_record.clear()
    while True:

        tt0 = time.time()
        

        if step >= int((len(w_record))):
            break

        vel.linear.x = 0.1
        vel.angular.z = w_record[step]*10.5
    
        base_pub.publish(vel)
        joint_init=[2.1394758224487305, -2.140566965142721, 1.9454715887652796, -2.9405275783934535, 4.144657135009766, -3.131549660359518]
        J = ur5e_robot_Jacob_tool_length0145(ur5_current_joint_position[0],ur5_current_joint_position[1],ur5_current_joint_position[2],ur5_current_joint_position[3],ur5_current_joint_position[4],ur5_current_joint_position[5])
        manipulability=sqrt(np.linalg.det(np.matmul(J,np.transpose(J))))
        
        # if(manipulability < 0.01):
        #     print(manipulability)
        #     break
        
        
        
        
        joint_vel = np.matmul(np.linalg.pinv(J),ur5_end_vel)
        #print('J',J)
        #print('joint VEL',joint_vel)
        #print('joint',ur5_current_joint_position)
        point.positions = [ur5_current_joint_position[0]+joint_vel[0]*0.1,ur5_current_joint_position[1]+joint_vel[1]*0.1,ur5_current_joint_position[2]+joint_vel[2]*0.1,ur5_current_joint_position[3]+joint_vel[3]*0.1,ur5_current_joint_position[4]+joint_vel[4]*0.1,ur5_current_joint_position[5]+joint_vel[5]*0.1]
        #point.velocities = joint_vel  #joint_vel.ravel()
        point.time_from_start = rospy.Duration(0.1,0)
        msg.points = [point]
        msg.header.seq = step
        #ur_pub.publish(msg)


        step +=1
        time.sleep(0.1+tt0-time.time())
    



        ur5_current_joint_position = ur_sub_record[-1]

    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)




    plt.plot([x[0] for x in car_sub_record],[x[1] for x in car_sub_record])


    np.savetxt('car_xytheta_dot_record3.txt',np.array(car_sub_record)[20:,:])

    plt.show()
    plt.pause(0)


