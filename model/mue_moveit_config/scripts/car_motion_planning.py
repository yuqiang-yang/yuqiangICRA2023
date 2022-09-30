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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import seaborn as sns

pi = 3.1415926

#图像显示设置
ur_sub_cnt = 0
ur_fig_refresh = False
car_sub_cnt = 0
car_fig_refresh = False
ur_sub_record = []
car_sub_record = []


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
class MyGPR():
    def __init__(self,data_in,data_out):
        self.data_in = data_in
        self.data_out = data_out
        
        kernel1 = 1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+1*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
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

        plt.show()




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
    
    
    #读取数据，数据滤波降采样并输入GPR中
    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path = readTxt()
    xy_dot_path_filtered = mean_filter(xy_dot_path)
    xy_dot_path_filtered_down = down_sample(xy_dot_path_filtered,5)
    xy_path_down = down_sample(xy_path,5)
    gpr = MyGPR(xy_path_down,xy_dot_path_filtered_down)
    #gpr.get_diff_varian_matrix(0,3.2,0,3.2,xy_path,step=400)    #一定要调用这个
    gpr.plot_field(-2,5,-2,5,xy_path)
    msg = JointTrajectory()
    vel = Twist()

    msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    point = JointTrajectoryPoint()


    step = 0
    vel.linear.x = 0
    base_pub.publish(vel)

    time.sleep(0.5)

    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    car_sub_record.clear()
    ur_sub_record.clear()

    plt.figure()
    while True:
        tt0 = time.time()
        
        #假如说目前的位置已经离终点很近了，那么就退出
        if np.sqrt((current_x-2.8)**2+(current_y-2.8)**2) < 0.2:
            break
        t0 = time.time()
        #result,std = gpr.gpr1.predict(np.array((current_x,current_y)).reshape(1,-1),return_std=True)
        result = gpr.get_next_step(np.array((current_x,current_y)).reshape(1,-1))


        target_angle = np.arctan2(result[0,1],result[0,0])
        delta_theta = np.arctan2(result[0,1],result[0,0])-current_theta
        step_size = 0.2
        
        if delta_theta*57.3 > 5 or np.sqrt(result[0,0]**2+result[0,1]**2) > 1:
            bezier = BEZIER()
            w_record = bezier.get_result(current_x,current_y,current_theta,current_x+step_size*np.cos(target_angle),current_y+step_size*np.sin(target_angle),current_theta+delta_theta,step_size=0.01)
            
            bezier.plot_bezier()
            #print(w_record)
            print('bezier')

            for w in w_record:

                vel.linear.x = 0.02
                vel.angular.z = w*2.05
                base_pub.publish(vel)
                time.sleep(0.5)
            vel.linear.x = 0
            vel.angular.z = 0
            base_pub.publish(vel)    
        else:    
            vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*5
            vel.angular.z = (np.arctan2(result[0,1],result[0,0])-current_theta)*10.2
            base_pub.publish(vel)
            time.sleep(0.1)


        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
    




    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)



    #plt.figure()
    plt.plot([x[0] for x in car_sub_record],[x[1] for x in car_sub_record])
    plt.plot(xy_path[:,0],xy_path[:,1])

    np.savetxt('car_xytheta_dot_record3.txt',np.array(car_sub_record)[:,:])

    plt.show()
    plt.pause(0)


