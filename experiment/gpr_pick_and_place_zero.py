# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('.')
import rospy
import time
import numpy as np
import rtde_receive
import rtde_control
import threading
import seaborn as sns

from matplotlib import projections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from trajectory_msgs.msg import *
from control_msgs.msg import *
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from jacobi.ur5e_robot_Jacob_tool_length0145 import *
from jacobi.ur5e_car_kinematics_class_2 import UR5e_car_kinematics
from experiment.robotiq_gripper_control import RobotiqGripper
from scipy import signal
import experiment.robotiq_gripper as robotiq_gripper
ip = "192.168.176.224"
rtde_r = rtde_receive.RTDEReceiveInterface(ip)
for i in range(3):
    try:
        rtde_c = rtde_control.RTDEControlInterface(ip)
        break
    except Exception:
        time.sleep(3)
        print('keep trying to connect RTDE Control')
        if i == 2:
            sys.exit()
def log_info(gripper):
    pass
    # print("Pos:",{str(gripper.get_current_position()): >3}
        #   "Open:" ,{gripper.is_open()}  
        #   "Closed:", {gripper.is_closed()}  )

print("Creating gripper...")
gripper = robotiq_gripper.RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(ip, 63352)
print("Activating gripper...")
gripper.activate()

print("Testing gripper...")
gripper.move_and_wait_for_pos(0, 255, 255)
#gripper.move_and_wait_for_pos(0, 255, 255)
Jiting = False  #emergency stop flag


# gripper = RobotiqGripper(rtde_c)

# # Activate the gripper and initialize force and speed
# gripper.activate()  # returns to previous position after activation
# gripper.set_force(50)  # from 0 to 100 %
# gripper.set_speed(100)  # from 0 to 100 %
#gripper.open()
ur_sub_record = []
ur_end_record = []
car_sub_record = []
car_record = []
#7的整體效果跑得還行
JiTing = False
def press_enter_to_JiTing():#不是完全的急停
    global JiTing
    key=input()
    JiTing=True
    key=input()
    JiTing=True
    sys.exit()  #exit this input thread
listener=threading.Thread(target=press_enter_to_JiTing)
listener.start()


def axixAngle_to_rotationMatrix(axis, angle):
    c=cos(angle)
    s=sin(angle)
    kx=axis[0]
    ky=axis[1]
    kz=axis[2]
    #print(sqrt(kx**2+ky**2+kz**2))
    c_1=1.0-c
    r11=kx**2*c_1+c
    r12=ky*kx*c_1-kz*s
    r13=kz*kx*c_1+ky*s
    r21=kx*ky*c_1+kz*s
    r22=ky**2*c_1+c
    r23=kz*ky*c_1-kx*s
    r31=kx*kz*c_1-ky*s
    r32=ky*kz*c_1+kx*s
    r33=kz**2*c_1+c
    return np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])


def seperate_angvec(angvec):
    phi=sqrt(angvec[0]**2+angvec[1]**2+angvec[2]**2)
    if phi==0:
        k=[0.0,0.0,0.0]
    else:
        k=[angvec[0]/phi,angvec[1]/phi,angvec[2]/phi]
    return k, phi
def axixAngel2R(angvec):
    axi,ang = seperate_angvec(angvec)
    return axixAngle_to_rotationMatrix(axi,ang)
#given tow axieangle, calculate the orientation error
def get_delta_angle(Ac,Ad):
    Rc = axixAngel2R(Ac)
    Rd = axixAngel2R(Ad)

    delta_R=np.matmul(Rd,np.transpose(Rc))
    d_phi=acos((delta_R[0,0]+delta_R[1,1]+delta_R[2,2]-1)/2)
    s_d_phi=sin(d_phi)
    if abs(s_d_phi)<0.00001:
        return 0.0,0.0,0.0
        pass
    else:
        k=np.array([delta_R[2,1]-delta_R[1,2],delta_R[0,2]-delta_R[2,0],delta_R[1,0]-delta_R[0,1]])*1/(2*s_d_phi)
        delta_phi=k*d_phi
        return delta_phi
#给定三个方向的力误差，得到三个方向的速度 ef 和 xdot都要是np数组才行
def admittance_control(ef,m,d,x_dot,dt,speed_limit=0.2):
    k1=(ef-d*x_dot)/m
    k2=(ef-d*(x_dot+dt*k1))/m
    y_t0=x_dot
    y_t1=y_t0+0.5*(k1+k2)*dt

    for i in range(3):
        if abs(y_t1[i]) > speed_limit:
            y_t1[i]=abs(y_t1[i])/y_t1[i]*speed_limit
    return y_t1

#car state subsciber callback function
def car_sub_callback(odom):
    x = odom.pose.pose.orientation.x
    y = odom.pose.pose.orientation.y
    z = odom.pose.pose.orientation.z
    w = odom.pose.pose.orientation.w
    theta = atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    car_sub_record.append([odom.pose.pose.position.x, odom.pose.pose.position.y, theta ,odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])

#read UR and car demonstration data
def readTxt():#16勉強還行
    #data = np.loadtxt('car_cycle_zero54.txt')
    data = np.loadtxt('data/12carpick_real.txt')
    #51還行  54是視頻版本
    xy_path = data[:,0:2]
    xy_dot_path = np.concatenate((data[:,3].reshape(-1,1)*np.cos(data[:,2].reshape(-1,1)),data[:,3].reshape(-1,1)*np.sin(data[:,2]).reshape(-1,1)),axis=1)
    xy_theta_path = data[:,0:3]
    xy_theta_dot_path = np.concatenate((xy_dot_path,data[:,5].reshape(-1,1)),axis=1)

    #ur_end_record = np.loadtxt('ur_circle_zero54.txt')
    ur_end_record = np.loadtxt('data/12pickend_real.txt')

    ur_end_record -= ur_end_record[0]
    return xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path,ur_end_record
#mean filter and lowpass but filter
def mean_filter(data,windows_size=4,is_but=False):
    rows, cols = data.shape
    filtedData = np.zeros((rows,cols))
    windows_size = 4
    if is_but:
        b, a = signal.butter(8, 0.1, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
        for i in range(cols):
            data[:,i] = signal.filtfilt(b, a, data[:,i])  #data为要过滤的信号
    windows = np.zeros((windows_size,cols))
    for i in range(windows_size):
        windows[i,:] = data[0,:]
    for i in range(rows):
        windows[i%windows_size,:] = data[i,:]
        filtedData[i,:] = np.sum(windows,axis=0)/windows_size
    return filtedData

#downsample a numpy array
def down_sample(data,times):
    rows, cols = data.shape
    down_data = np.zeros((int(rows//times),cols))
    for i in range(int(rows//times)):
        down_data[i,:] = data[i*times,:]
    return down_data

class MyGPR():
    def __init__(self,data_in,data_out,normalize = False,n_restarts_optimizer=5,is_noise = True):
        self.data_in = data_in
        self.data_out = data_out
        print(data_in.shape)
        kernel1 = 0.1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+0.1*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
        #kernel1 =  10*RBF(length_scale=1e-3, length_scale_bounds=(1e-15, 1e2))+0.000001*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
        if not is_noise:
            kernel1 = 0.1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-14, 1e3))#+0.1*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))

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
        result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)
        distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        if distance > 2:
            distance = 5.4
        else:
            distance *= 2.7
        self.dis_cof = 0.1/np.max(self.diff_std)
        distance *= self.dis_cof
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
        #print('arm',np.argmin(distance))
        return self.data_in[np.argmin(distance)]

def time_optimal_for_1D_tracking(x1_tf, x2_tf, x1_t0, x2_t0, abs_acc):
    #判断t0的状态在tf的状态的什么位置
    string=''
    if abs(x1_tf-x1_t0)+abs(x2_tf-x2_t0)<0.001:
        abs_acc=0.001
    if x2_t0>x2_tf:#,走abs_acc<0
        string=string+'x2_t0>x2_tf  |'
        t_minus=(x2_t0-x2_tf)/(-abs_acc)
        temp=0.5*(-abs_acc)*t_minus**2+x2_t0*t_minus+x1_tf
        if x1_t0<temp:
            acc=abs_acc
            string=string+'x1_t0<temp  |加速'+str(acc)
        else:#x1_t0>=temp
            acc=-abs_acc
            string=string+'x1_t0>=temp  |减速'+str(acc)
    elif x2_t0<x2_tf:#,走abs_acc>0
        string=string+'x2_t0<x2_tf  |'
        t_plus=(x2_t0-x2_tf)/abs_acc
        temp=0.5*abs_acc*t_plus**2+x2_t0*t_plus+x1_tf
        if x1_t0<=temp:
            acc=abs_acc
            string=string+'x1_t0<=temp  |加速'+str(acc)
            
        else:
            acc=-abs_acc
            string=string+'x1_t0>temp  |减速'+str(acc)
    else:#x2_t0==x2_tf
        string=string+'x2_t0==x2_tf  |'
        if x1_t0<x1_tf:
            acc=abs_acc
            string=string+'x1_t0<x1_tf  |加速'+str(acc)
            
        elif x1_t0>x1_tf:
            acc=-abs_acc
            string=string+'x1_t0>x1_tf  |减速'+str(acc)
            
        else:#x1_t0==x1_tf
            acc=0.0
            string=string+'x1_t0==x1_tf  |不变'+str(acc)
    return acc,string


if __name__ == "__main__":
    print('start the main function')

    #init ros node
    rospy.init_node("UR5Control")
    base_pub = rospy.Publisher ("/mobile_base/cmd_vel", Twist,queue_size=0) 
    base_sub = rospy.Subscriber("/mobile_base/odom",Odometry,car_sub_callback)
    ur5e_car = UR5e_car_kinematics()

    time.sleep(0.1)
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
    last_nearest_index = 0
    w0 = 0     #current angluar velocity


    #stop the car and move the UR to the initial pose
    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)
    joint_init=np.array([-90, -90, -90, -90, 90, -90])/57.3


    rtde_c.moveJ(joint_init,0.5,0.1)


    #read the FT sensor data
    windows_size = 3
    F_raw = np.zeros((windows_size,6))
    rtde_c.zeroFtSensor()
    for i in range(windows_size):
        F_raw[i] = rtde_r.getActualTCPForce()
        time.sleep(0.05)

    #read demonstration data, filter and downsample
    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path,arm_xyztheta_path = readTxt()

    down_times = int(xy_path.shape[0]//40 + 1)
    print('down_times',down_times)
    xy_dot_path_filtered = mean_filter(xy_dot_path,is_but=True)
    xy_dot_path_filtered_down = down_sample(xy_dot_path_filtered,down_times)
    xy_path_down = down_sample(xy_path,down_times)
 
    down_times2 = int(xy_path.shape[0]//40 + 1)
    xy_path_down_2 = down_sample(xy_path,down_times2)
    #arm_xyztheta_path = mean_filter(arm_xyztheta_path,windows_size=6,is_but=True)
    arm_xyztheta_path_down = down_sample(arm_xyztheta_path,down_times2)
    # arm_xyztheta_path_down[:,0] += xy_path_down[:,1]
    # arm_xyztheta_path_down[:,1] -= xy_path_down[:,0]

    #GPR fit
    car_gpr = MyGPR(xy_path_down,xy_dot_path_filtered_down,normalize=False)
    car2arm_gpr = MyGPR(xy_path_down_2,arm_xyztheta_path_down,normalize=False)
    # car2arm_gpr.data_in = xy_path
    # car2arm_gpr.data_out = arm_xyztheta_path
    # arm_xyztheta_path_down[:,0] -= xy_path_down[:,1]
    # arm_xyztheta_path_down[:,1] += xy_path_down[:,0]
    #plot the gpr field 
    car_gpr.plot_field(-1,3,-1,3,xy_path)
    # plt.show()
    #plt.figure()
    result1 = car2arm_gpr.gpr1.predict((xy_path),return_std=False)
    result2 = car2arm_gpr.gpr1.predict(xy_path_down_2+np.array((0,0.1)),return_std=False)
    result3 = car_gpr.gpr1.predict((xy_path),return_std=False)
    # print(result3[:1000])
    # for i in range(1000):
        
    #     print(atan2(result3[i,1],result3[i,0])*57.3)
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.plot3D(arm_xyztheta_path[:,0]+xy_path[:,1],arm_xyztheta_path[:,1]-xy_path[:,0],arm_xyztheta_path[:,2])
    ax1.scatter3D(arm_xyztheta_path_down[:,0]+xy_path_down_2[:,1],arm_xyztheta_path_down[:,1]-xy_path_down_2[:,0],arm_xyztheta_path_down[:,2])
    ax1.plot3D(result1[:,0]+xy_path[:,1],result1[:,1]-xy_path[:,0],result1[:,2])
    # ax1.plot3D(result1[:,0],result1[:,1],result1[:,2])
    # ax1.plot3D(result2[:,0],result2[:,1],result2[:,2])

    plt.show()


    #get wholebody state
    end_init_state = rtde_r.getActualTCPPose()
    end_state = rtde_r.getActualTCPPose()
    arm_state = rtde_r.getActualQ()
    grip_index = np.argmin(arm_xyztheta_path[:3000,2])
    ungrip_index = np.argmin(arm_xyztheta_path[4001:,2])+4001
    grip_flag = False
    print('grip_index',grip_index)
    print('ungrip_index',ungrip_index)
    current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]


    car_sub_record.clear()
    ur_sub_record.clear()
    ur_end_record.clear()
    step = 0
    bz_flag = False #if true, the car will turn to the disired angle without linear vel

    time.sleep(0.5)
    #set the max velocity and acc limits 
    wholebody_linear_acc = 0.05
    wholebody_angular_acc = 0.1
    car_linear_acc = 0.2
    car_angular_acc = 0.3
    car_linear_vel = 0.4
    car_angular_vel = 0.48

    #set the force sensor limits
    force_min = 2
    force_max = 30
    torge_min = 0.1
    torge_max = 3
    
    print('moving to first point slowly')
    last_end_error = np.zeros(6)
    robot_target = car2arm_gpr.gpr1.predict(np.array((current_x,current_y)).reshape(1,-1))
    #first_point = np.zeros(6)
    #first_point[:3] = robot_target[0,:3]+np.array(end_init_state[:3])
    #first_point[2] = arm_xyztheta_path[0][2]+np.array(end_init_state[2])
    #first_point[3:] = end_init_state[3:]#+robot_target[0,3:]
    #rtde_c.moveL(first_point,0.5,0.2)
    time.sleep(0.5)
    end_state = rtde_r.getActualTCPPose()
    arm_state = rtde_r.getActualQ()
    print('start the main loop')
    first_flag = True
    while JiTing == False:
        if rtde_c.isProgramRunning() == False:
            print('RTDE C error')
            break
        if step > 20000:
            break
        t0 = time.time()

        ###########################car control###################################################
        result = car_gpr.get_next_step(np.array((current_x,current_y)).reshape(1,-1))
        target_angle = np.arctan2(result[0,1],result[0,0])
        if last_nearest_index < grip_index+500 and first_flag == True:
            target_angle=0
        if last_nearest_index > 5000:
            first_flag = False
        delta_theta = target_angle-current_theta
        if step % 10 == 0:
            print('position',np.array((current_x,current_y)),'result',np.arctan2(result[0,1],result[0,0]),'delta',delta_theta)
        #-pi~pi 
        if target_angle > 2.5 and current_theta < -2.5:
            delta_theta -= 2*pi
        elif target_angle < -2.5 and current_theta > 2.5:
            delta_theta += 2*pi 

        w0 = car_sub_record[-1][5]

        #if delta angle is small, run directly
        if np.abs(delta_theta) > car_angular_vel*dt*10:        
            bz_flag = True

        if bz_flag ==True:
            if last_nearest_index > 1800 and last_nearest_index < arm_xyztheta_path.shape[0]-1000:
                vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0.9
                acc,_ = time_optimal_for_1D_tracking(delta_theta, delta_theta*0 , 0, w0, car_angular_acc)
                if abs(delta_theta) < 0.002 and car_sub_record:
                    bz_flag = False
            else:



                vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0.4
                acc,_ = time_optimal_for_1D_tracking(delta_theta, delta_theta*0 , 0, w0, car_angular_acc)
                if abs(delta_theta) < 0.005:
                    bz_flag = False

            #print('Bezier','acc:',acc,'delta:',delta_theta,'w0',w0)
        else:    
            acc,_ = time_optimal_for_1D_tracking(delta_theta, delta_theta*0, 0, w0, car_angular_acc)
            vel.linear.x = np.sqrt(result[0,0]**2+result[0,1]**2)*0.9

        vel.angular.z = w0+acc*dt
        

        #linear acc limits. The angular acc limits is considered in time optimal 1d tracking
        if abs(car_pre_linear_vel-vel.linear.x) > car_linear_acc*dt:
            vel.linear.x = car_pre_linear_vel + np.sign(vel.linear.x-car_pre_linear_vel)*car_linear_acc*dt
        car_pre_linear_vel = vel.linear.x
        
        if abs(vel.linear.x) > car_linear_vel:
            vel.linear.x = car_linear_vel*np.sign(vel.linear.x)
            print('the linear vel is too large')
        if abs(vel.angular.z) > car_angular_vel:
            vel.angular.z = car_angular_vel*np.sign(vel.angular.z)
            print('the anglular vel is too large')
            
        base_pub.publish(vel)

        ###########################UR control###################################################

        # GPR Term
        distance = np.linalg.norm(xy_path-np.array((current_x,current_y)).reshape(1,-1),axis=1)
        if step %10 == 0:
            print('arm',np.argmin(distance))
        robot_target = car2arm_gpr.gpr1.predict(xy_path[np.argmin(distance)].reshape(1,-1))
        #robot_target = car2arm_gpr.gpr1.predict(np.array((current_x,current_y)).reshape(1,-1))

        if abs(last_nearest_index - np.argmin(distance)) > 500:
            print('interval too large')
            # vel.linear.x = 0
            # vel.angular.z = 0
            # base_pub.publish(vel)
            # car_pre_linear_vel = 0
            last_nearest_index = np.argmin(distance)
            rtde_c.servoStop(0.5)

            robot_target[0,3:] = np.zeros(3)
            move_point = robot_target.ravel()+np.array(end_init_state)
            current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
            W_B_R = np.array([[cos(current_theta),-sin(current_theta),0],
            [sin(current_theta),cos(current_theta),0],
            [0,0,1]])
            w_v = np.array((-current_x + xy_path[last_nearest_index][0],xy_path[last_nearest_index][1] - current_y,0))
            r_v = np.matmul(np.transpose(B_0_R),np.matmul(np.transpose(W_B_R),w_v))
            r_v = r_v.ravel()

            move_point[0] += r_v[0]
            move_point[1] += r_v[1]
            move_point[2] = arm_xyztheta_path[last_nearest_index][2]+np.array(end_init_state[2])
            rtde_c.moveL(move_point,1,0.4)
            arm_state = rtde_r.getActualQ()
            end_state = rtde_r.getActualTCPPose()
            last_nearest_index = np.argmin(distance)
            car_pre_linear_vel = car_sub_record[-1][3]
            continue

        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]

        ur5_end_vel = np.zeros(6)
        ur5_end_vel[:3] = robot_target[0,:3]+np.array(end_init_state[:3])-np.array(end_state[:3])
        #print('vel:',robot_target[0,:3]+np.array(end_init_state[:3])-np.array(end_state[:3]))
        #ur5_end_vel[3:] = get_delta_angle(np.array(end_state[3:]),robot_target[0,3:]+np.array(end_init_state[3:]))
        B_0_R = np.array([[0,-1,0],
                [1,0,0],
                [0,0,1]])
        W_B_R = np.array([[cos(current_theta),-sin(current_theta),0],
        [sin(current_theta),cos(current_theta),0],
        [0,0,1]])
        last_nearest_index = np.argmin(distance)

        w_v = np.array((-current_x + xy_path[last_nearest_index][0],xy_path[last_nearest_index][1] - current_y,0))
        r_v = np.matmul(np.transpose(B_0_R),np.matmul(np.transpose(W_B_R),w_v))
        r_v = r_v.ravel()

        ur5_end_vel[0] += r_v[0]
        ur5_end_vel[1] += r_v[1]

        ur5_end_vel[0] = arm_xyztheta_path[last_nearest_index][0]+np.array(end_init_state[0])-np.array(end_state[0])
        ur5_end_vel[1] = arm_xyztheta_path[last_nearest_index][1]+np.array(end_init_state[1])-np.array(end_state[1])    
        ur5_end_vel[2] = arm_xyztheta_path[last_nearest_index][2]+np.array(end_init_state[2])-np.array(end_state[2])
        #ur5_end_vel[3:] = get_delta_angle(np.array(end_state[3:]),robot_target[0,3:]+np.array(end_init_state[3:]))

        if grip_flag == False and abs(last_nearest_index - grip_index) < 200 and ur5_end_vel[2] > 0:
            gripper.move_and_wait_for_pos(255, 255, 255)
            grip_flag = True
        if grip_flag == True and abs(last_nearest_index - ungrip_index) < 200 and ur5_end_vel[2] > 0:
            gripper.move_and_wait_for_pos(0, 255, 255)
            grip_flag = False

        
        J = ur5e_robot_Jacob_tool_length(arm_state)
        manipulability=sqrt(np.linalg.det(np.matmul(J,np.transpose(J))))
        if(manipulability < 0.01):
            print('manipulability:',manipulability)
            break
        joint_vel = np.matmul(np.linalg.pinv(J),ur5_end_vel)



        #FT corrdinate transformation
        F_raw[step % windows_size] = rtde_r.getActualTCPForce()
        F_filtered = np.mean(F_raw,axis=0)


        W_0_R = np.matmul(W_B_R,B_0_R)
        W_Force = np.matmul(W_0_R,F_filtered[:3])
        W_Torge = np.matmul(W_0_R,F_filtered[3:])

        for i in range(3):
            if abs(W_Force[i]) < force_min:
                W_Force[i] = 0
            elif abs(W_Force[i]) > force_max:
                W_Force[i] = (abs(W_Force[i]) - force_max)*np.sign(W_Force[i])
        
        for i in range(3):
            if abs(W_Torge[i]) < torge_min:
                W_Torge[i] = 0
            if abs(W_Torge[i]) > torge_max:
                W_Torge[i] = (abs(W_Torge[i]) - torge_max)*np.sign(W_Torge[i])

        #Admittace Controller
        V_pre = admittance_control(W_Force,4,80,V_pre,0.01,speed_limit=0.1)
        W_pre = admittance_control(W_Torge,0.01,0.5,W_pre,0.01,speed_limit=0.1)
        V_pre[2] = 0 
        W_pre[0] = 0    #disable the roll and pitch
        W_pre[1] = 0
        wholebody_vel = np.hstack((V_pre,W_pre))
        
        #wholebody speed limits
        for i in range(3):
            if abs(wholebody_vel[i] - wholebody_pre_vel[i]) >  wholebody_linear_acc*dt:
                wholebody_vel[i] = wholebody_pre_vel[i] + np.sign(wholebody_vel[i] - wholebody_pre_vel[i])*wholebody_linear_acc*dt
        for i in range(3):
            if abs(wholebody_vel[i+3] - wholebody_pre_vel[i+3]) > wholebody_linear_acc*dt:
                wholebody_vel[i+3] = wholebody_pre_vel[i+3] + np.sign(wholebody_vel[i+3] - wholebody_pre_vel[i+3])*wholebody_linear_acc*dt
  
        wholebody_J = ur5e_car.get_jacobian_lx([current_x,current_y,current_theta] + arm_state)
        Weight=np.array([[999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
        z=[0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        J_transpose=np.transpose(wholebody_J)
        Weignt_inverse=np.linalg.pinv(Weight)

        J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, np.linalg.pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
        joint_v=np.matmul(J_sharp,wholebody_vel)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)
        joint_vel = joint_vel.ravel()
        # for i in range(6):
        #     if abs(joint_vel[i])>10*dt:
        #         joint_vel[i] = 10*dt*np.sign(joint_vel[i])
        rtde_c.servoJ(arm_state+joint_vel+ joint_v[2:]*0.0,0,0,0.01,0.15,200)
        step +=1
        wholebody_pre_vel = wholebody_vel
        arm_state = rtde_r.getActualQ()
        end_state = rtde_r.getActualTCPPose()
        ur_end_record.append(end_state)
        car_record.append(car_sub_record[-1])
        current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]
#########################################################################################################
        #在仿真中，机械臂的控制是位置控制，和延时多久没关系，但是车子是速度控制，所以要保证命令的间隔要在0.1s左右
        #协调控制，调整车子位姿.要更新好最新的位姿再开始控制  
        #要减去初始值才是输入GPR的输入吧
        # coordinate_result = car2arm_gpr.find_nearest_point(np.array((end_state[0]-end_init_state[0],end_state[1]-end_init_state[1],end_state[2]-end_init_state[2],end_state[5]-end_init_state[5])).reshape(1,-1))
        # print('car',np.argmin(np.linalg.norm((xy_path-np.array((current_x,current_y)).reshape(1,-1)),axis=1)))
        # nearest_path = xy_path[np.argmin(np.linalg.norm((xy_path-np.array((current_x,current_y)).reshape(1,-1)),axis=1))]
        # coordinate_result = coordinate_result.reshape(1,-1)
        # nearest_path = nearest_path.reshape(1,-1)
        # distance = np.sqrt((current_x-coordinate_result[0,0])**2+(current_y-coordinate_result[0,1])**2)
        # print('current:',current_x,' ',current_y,'corrdinate:',coordinate_result,'distance:',distance)
        # current_x,current_y,current_theta = car_sub_record[-1][0],car_sub_record[-1][1],car_sub_record[-1][2]

        # if(distance > 0.2 and bz_flag == False):
        #     #启动协调过程
        #     result = car_gpr.gpr1.predict(np.array((coordinate_result[0,0],coordinate_result[0,1])).reshape(1,-1))
        #     target_angle = atan2(result[0,1],result[0,0])
        #     delta_theta = target_angle - current_theta
        #     if(abs(atan2(coordinate_result[0,1] - current_y,coordinate_result[0,0] - current_x) - current_theta) < 0.1):#正着走
        #         print('corr1')
        #         vel.linear.x = distance*3
        #         vel.angular.z = (target_angle-current_theta)*3
        #         base_pub.publish(vel)
        #         time.sleep(0.2)

        #     elif(abs(atan2(-coordinate_result[0,1] + current_y,-coordinate_result[0,0] + current_x) - current_theta) < 0.1):#反着走   
        #         print('corr2')
        #         vel.linear.x = -distance*3
        #         vel.angular.z = (target_angle-current_theta)*3
        #         base_pub.publish(vel)
        #         time.sleep(0.2)

        time.sleep(dt)

    rtde_c.servoStop(2)
    vel.linear.x = 0
    vel.angular.z = 0
    base_pub.publish(vel)
    # plt.plot(np.array(ur_end_record)-end_init_state)
    # plt.legend(('1','2','3','4','5','6','7','8','9','10','11','12'))
    plt.figure()
    plt.plot([x[0] for x in car_sub_record],[x[1] for x in car_sub_record])
    plt.plot(xy_path[:,0],xy_path[:,1])

    ur_end_record = np.array(ur_end_record)
    car_record = np.array(car_record)
    ur_end_record -= end_init_state
    fig = plt.figure()
    ax1 = fig.add_subplot(121,projection='3d')
    ax1.plot3D(ur_end_record[:,0]+car_record[:,1],ur_end_record[:,1]-car_record[:,0],ur_end_record[:,2])
    ax1.plot3D(arm_xyztheta_path_down[:,0]+xy_path_down[:,1],arm_xyztheta_path_down[:,1]-xy_path_down[:,0],arm_xyztheta_path_down[:,2])

    ax2 = fig.add_subplot(122,projection='3d')
    ax2.plot3D(ur_end_record[:,3]-ur_end_record[0,3],ur_end_record[:,4]-ur_end_record[0,4],ur_end_record[:,5]-ur_end_record[0,5])
    ax2.plot3D(arm_xyztheta_path_down[:,3],arm_xyztheta_path_down[:,4],arm_xyztheta_path_down[:,5])
    rtde_c.disconnect()

    np.savetxt('data/1pickend_real.txt',ur_end_record-ur_end_record[0])
    np.savetxt('data/1pickend_ref.txt',arm_xyztheta_path_down)
    np.savetxt('data/1carpick_real.txt',car_record)
    np.savetxt('data/1carpick_ref.txt',xy_path)
    plt.show()
    plt.pause(0)


