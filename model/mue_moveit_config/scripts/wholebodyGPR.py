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
from gazebo_msgs.msg import LinkStates
from control_msgs.msg import JointTrajectoryControllerState
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
ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()
ur_sub_record = []
ur_end_record = []
car_sub_record = []
#记录车子和臂在初始位置的六维位姿，而后通过GP算出来的值都相对于这个值进行偏移，初始位置这个值是固定的。
#####可以实现一个反复扫的任务，这个任务感觉是有价值的。类似AGV的效果，但是不需要传感器
#避障也可以是扰动
#机械臂状态接收回调函数
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
    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])

#读取车子数据
def readTxt():
    data = np.loadtxt('car_xytheta_dot_record1.txt')
    xy_path = data[:,0:2]
    xy_dot_path = np.concatenate((data[:,3].reshape(-1,1)*np.cos(data[:,2].reshape(-1,1)),data[:,3].reshape(-1,1)*np.sin(data[:,2]).reshape(-1,1)),axis=1)
    xy_theta_path = data[:,0:3]
    xy_theta_dot_path = np.concatenate((xy_dot_path,data[:,5].reshape(-1,1)),axis=1)

    return xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path
#窗口均值滤波
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

#下采样一个数组，times为下采样的倍数
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
        
        #kernel1 = 100 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+0.000001*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
        kernel1 =  10*RBF(length_scale=1e-3, length_scale_bounds=(1e-15, 1e2))+0.000001*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))

        self.gpr1 = GaussianProcessRegressor(kernel=kernel1,random_state=0,n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize)
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
    def get_next_step(self,predict_point):
        #给定一点，求考虑了方差引力场之后的结果方向
        result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)
        distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        if distance > 2:
            distance = 5.4
        else:
            distance *= 2.7
        return np.array((result[0,0] - self.diff_std[remap_i,remap_j,0]*distance ,  result[0,1] - self.diff_std[remap_i,remap_j,1]*distance)).reshape(1,-1)
 

    def plot_field(self,xmin,xmax,ymin,ymax,demonstrate_point,step=400,field_density = 20):
        diff_std = self.get_diff_varian_matrix(xmin,xmax,ymin,ymax,demonstrate_point,step=400)
        plt.figure()
        plt.subplot(221)
        sns.heatmap(abs(diff_std[:-1,:-1,0]))
        plt.subplot(222)
        sns.heatmap(abs(diff_std[:-1,:-1,1]))


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
            #p = get_minvariance_path(gpr1,predict_point[i,:],demonstrate_point)
            #length = np.sqrt((p[0] - predict_point[i,0])**2+(p[1]- predict_point[i,1])**2)
            #plt.arrow(predict_point[i,0],predict_point[i,1],result[i,0]+(p[0] -predict_point[i,0])/length/15 ,result[i,1]+(p[1]- predict_point[i,1])/length/15,head_width=0.02)
            
            remap_i = int(i//field_density) * int(step//field_density)
            remap_j = int(i%field_density) * int(step//field_density)
            distance = np.min(np.sqrt((demonstrate_point[:,0]-predict_point[i,0])**2+(demonstrate_point[:,1]-predict_point[i,1])**2))
            if distance > 2:
                distance = 5.4
            else:
                distance *= 2.7
            res = self.get_next_step(predict_point[i,:].reshape(1,-1))
            #plt.arrow(predict_point[i,0],predict_point[i,1], res[0], res[1],head_width=0.01)
            plt.arrow(predict_point[i,0],predict_point[i,1], result[i,0]- diff_std[remap_i,remap_j,0]*distance, result[i,1]- diff_std[remap_i,remap_j,1]*distance,head_width=0.01)
        
        plt.plot(demonstrate_point[:,0],demonstrate_point[:,1])

        #plt.show()
    def find_nearest_point(self,point):
        distance = np.linalg.norm((self.data_out-point),axis=1)
        print('arm',np.argmin(distance))
        return self.data_in[np.argmin(distance)]
    def plot_line(self,predict_point):
        #给定一堆点，画出末端的轨迹，默认支持前2维
        result,std = self.gpr1.predict(predict_point,return_std=True)
        #plt.figure()
        plt.plot(result[:,0],result[:,2])



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
    time.sleep(0.5)
    #准备发送数据
    msg = JointTrajectory()
    vel = Twist()
    np.set_printoptions(threshold=np.inf)

    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint','wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()

    #先给车发送一个指令，不然第一次指令无法响应
    vel.linear.x = 0
    base_pub.publish(vel)
    joint_init=[2.1394758224487305, -2.140566965142721, 1.9454715887652796, -2.9405275783934535, 4.144657135009766-2*pi, -3.131549660359518]

    point.positions = joint_init
    point.time_from_start = rospy.Duration(1,0)
    msg.points = [point]
    msg.header.seq = 0
    ur_pub.publish(msg)
    time.sleep(1.5)


    
    #读取数据，数据滤波降采样并输入GPR中
    down_times = 8
    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path = readTxt()
    xy_dot_path_filtered = mean_filter(xy_dot_path)
    xy_dot_path_filtered_down = down_sample(xy_dot_path_filtered,down_times)
    xy_path_down = down_sample(xy_path,down_times)
    #机械臂轨迹生成
    arm_xyztheta_path_noise = [] 
    arm_xyztheta_path = [] 
    for ii in range(1,len(xy_path)+1):
        i = ii/down_times
        arm_xyztheta_path_noise.append([0.0015*i,-0.0015*i,-0.10*np.cos(2*3.14159/200*i),0.001*i] + 0.00001*np.random.normal(0,2,4))
        arm_xyztheta_path.append([0.0015*i,-0.0015*i,-0.10*np.cos(2*3.14159/200*i),0.001*i])
    arm_xyztheta_path_down = down_sample(np.array(arm_xyztheta_path),down_times)
    arm_xyztheta_path_noise_down = down_sample(np.array(arm_xyztheta_path_noise),down_times)


    #GPR对象训练
    car_gpr = MyGPR(xy_path_down,xy_dot_path_filtered_down)
    car2arm_gpr = MyGPR(xy_path_down,arm_xyztheta_path_down,normalize=True)
    car2arm_gpr.data_in = xy_path
    car2arm_gpr.data_out = arm_xyztheta_path
    # arm2car_gpr = MyGPR(arm_xyztheta_path,xy_path_down)         #自变量不能加噪声

    #画图
    #car_gpr.get_diff_varian_matrix(-2,5,-2,5,xy_path,step=400)    #一定要调用这个
    car_gpr.plot_field(-2,5,-2,5,xy_path)
    plt.show()
    # plt.figure()
    # result1 = car2arm_gpr.gpr1.predict(xy_path_down,return_std=False)
    # result2 = car2arm_gpr.gpr1.predict(xy_path_down+np.array((0,0.3)),return_std=False)
    
    # plt.plot(result1[:,0],result1[:,2])
    # plt.plot(result2[:,0],result2[:,2])
    # plt.show()

    end_init_state = ur_end_record[-1]
    end_state = ur_end_record[-1]
    arm_state = ur_sub_record[-1]
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    robot_target = car2arm_gpr.gpr1.predict(np.array((current_x,current_y)).reshape(1,-1))
    print('current',(current_x,current_y),'target',robot_target)
    ur5_end_vel = np.array((robot_target[0,0]+end_init_state[0]-end_state[0],robot_target[0,1]+end_init_state[1]-end_state[1],robot_target[0,2]+end_init_state[2]-end_state[2],0,0,robot_target[0,3]+end_init_state[5]-end_state[5])).reshape(-1,1)
    J = ur5e_robot_Jacob_tool_length(arm_state)
    manipulability=sqrt(np.linalg.det(np.matmul(J,np.transpose(J))))
    if(manipulability < 0.01):
        print('manipulability',manipulability)
    joint_vel = np.matmul(np.linalg.pinv(J),ur5_end_vel)
    point.positions = arm_state + joint_vel.ravel()
    point.time_from_start = rospy.Duration(1,0)
    msg.points = [point]
    ur_pub.publish(msg)
    time.sleep(2)


    #设置当前车子状态
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    arm_state = ur_sub_record[-1]
    end_state = ur_end_record[-1]
    #end_init_state = ur_end_record[-1]
    car_sub_record.clear()
    ur_sub_record.clear()
    ur_end_record.clear()
    last_target = np.array((0,0,0,0)).reshape(1,-1)
    step = 0
    bz_flag = False
    bz_index = 0

    w_record = []
####################################################################################################################################################
    while True:
        tt0 = time.time()
        
        #假如说目前的位置已经离终点很近了，那么就退出
        if np.sqrt((current_x-2.8)**2+(current_y-2.8)**2) < 0.1:
            break
        t0 = time.time()

        #得到下一步
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
        
        #print('current',(current_x,current_y),'target',robot_target)
        #robot_target = np.array((0,0,0,0)).reshape(1,-1)
        ur5_end_vel = np.array((robot_target[0,0]+end_init_state[0]-end_state[0],robot_target[0,1]+end_init_state[1]-end_state[1],robot_target[0,2]+end_init_state[2]-end_state[2],0,0,robot_target[0,3]+end_init_state[5]-end_state[5])).reshape(-1,1)
        ur5_end_vel[1] *= 0.1
        ur5_end_vel[0] *= 0.1
        ur5_end_vel[2] *= 0.1
        ur5_end_vel[3] *= 0.1
        J = ur5e_robot_Jacob_tool_length(arm_state)
        manipulability=sqrt(np.linalg.det(np.matmul(J,np.transpose(J))))
        if(manipulability < 0.01):
            print('manipulability',manipulability)
        joint_vel = np.matmul(np.linalg.pinv(J),ur5_end_vel)
        point.positions = arm_state + joint_vel.ravel()
        point.time_from_start = rospy.Duration(0.01,0)
        msg.points = [point]
        msg.header.seq = step
        ur_pub.publish(msg)
        if time.time()- t0 < 0.1:
            time.sleep(0.1-time.time()+t0)

        step +=1
        last_target = robot_target
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


