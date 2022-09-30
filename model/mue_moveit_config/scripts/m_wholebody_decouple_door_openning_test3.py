#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import rtde_receive
import rtde_control
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from numpy.linalg import norm, inv, det, cond, pinv
import time
from ur5e_car_kinematics_class_2 import UR5e_car_kinematics
import matplotlib.pyplot as plt
from matplotlib import pyplot
#from WholeBody_kinematics_dr1 import WB_KINEMATICS
#from scipy.spatial.transform.rotation import Rotation as rotate

IP = "192.168.100.2"
rtde_r = rtde_receive.RTDEReceiveInterface(IP)
rtde_c = rtde_control.RTDEControlInterface(IP)
l_k = 0.145
rtde_c.setTcp([0, 0, l_k, 0, 0, 0])
rtde_c.zeroFtSensor()

def list_multiply(l,scale):
    l_scaled=[]
    for ii in range(0,len(l)):
        l_scaled.append(l[ii]*scale)
    return l_scaled
def list_element_add(a,b):
    c=[]
    for i in range(0,len(a)):
        c.append(a[i]+b[i])
    return c
def list_element_minus(a,b):
    c=[]
    for i in range(0,len(a)):
        c.append(a[i]-b[i])
    return c
def list_norm(l):
    num=len(l)
    l_out=[]
    length_square_sum=0.0
    for i in range(0,num):
        length_square_sum=l[i]**2+length_square_sum
    length=sqrt(length_square_sum)
    for i2 in range(0,num):
        l_out.append(l[i2]/length)
    return l_out
def list_dot_multiply(l1,l2):
    temp1=0.0
    for ii in range(0,len(l1)):
        temp1=temp1+l1[ii]*l2[ii]
    return temp1
def list_multiply_elementBYelement(l1,l2):
    len_l1=len(l1)
    if len_l1==2:
        temp=[0.0,0.0]
    elif len_l1==3:
        temp=[0.0,0.0,0.0]
    for ii in range(0,len_l1):
        temp[ii]=l1[ii]*l2[ii]
    return temp
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
def rotationMatrix_to_axixAngle(R):
    tr_R=R[0,0]+R[1,1]+R[2,2]

    if tr_R!=3.0 and tr_R!=-1.0:
        angle=acos((tr_R-1)/2)
        s_angle_2=2*sin(angle)
        axis=[(R[2,1]-R[1,2])/s_angle_2, \
            (R[0,2]-R[2,0])/s_angle_2, \
            (R[1,0]-R[0,1])/s_angle_2]
    elif tr_R==3.0:
        angle=0.0
        axis=[0.0,0.0,1.0]#any is ok
    else:#tr_R==-1
        angle=pi
        axis=[sqrt((1.0+R[0,0])/2.0),sqrt((1.0+R[1,1])/2.0),sqrt((1.0+R[2,2])/2.0)]
    return axis, angle
def axixAngle_to_rotationMatrix_anxisAngle(T1):
    T0=[0.0,0.0,1.0]
    T0_length=sqrt(T0[0]**2+T0[1]**2+T0[2]**2)
    T1_length=sqrt(T1[0]**2+T1[1]**2+T1[2]**2)
    T1=[T1[0]/T1_length,T1[1]/T1_length,T1[2]/T1_length]
    T1_length=sqrt(T1[0]**2+T1[1]**2+T1[2]**2)
    axis0=T0
    if T1[0]**2+T1[1]**2!=0:
        angle0=pi-atan2(T1[0],T1[1])
    else:
        print('T1[0]**2+T1[1]**2=0')
        angle0=0
    R0=axixAngle_to_rotationMatrix(axis0,angle0)
    axis1=np.cross(T0,T1)
    T0_length=sqrt(T0[0]**2+T0[1]**2+T0[2]**2)
    T1_length=sqrt(T1[0]**2+T1[1]**2+T1[2]**2)
    angle1=acos(np.dot(T0,T1)/(T0_length*T1_length))
    R1=axixAngle_to_rotationMatrix(axis1,angle1)
    R_total=np.dot(R1, R0)
    axis, angle = rotationMatrix_to_axixAngle(R_total)
    return axis, angle, R_total
def seperate_angvec(angvec):
    phi=sqrt(angvec[0]**2+angvec[1]**2+angvec[2]**2)
    if phi==0:
        k=[0.0,0.0,0.0]
    else:
        k=[angvec[0]/phi,angvec[1]/phi,angvec[2]/phi]
    return k, phi
def pi_forceerror2desireddisplacement2(ef, m, d, x_dot, dt):
    k1=(ef-d*x_dot)/m
    #l1=(ef-b*dot_increment)/m
    k2=(ef-d*(x_dot+dt*k1))/m
    y_t0=x_dot
    y_t1=y_t0+0.5*(k1+k2)*dt
    # speed limitation
    if isContact:
        speed_limit=0.05
    else:
        speed_limit=0.01
    #speed_limit=0.2
    if abs(y_t1)>speed_limit:
        #print('abs(delta)>0.02')
        #print('y_t1',y_t1)
        y_t1=abs(y_t1)/y_t1*speed_limit
    return y_t1
def pi_forceerror2desireddisplacement_forTorque(ef, m, d, x_dot, dt):
    
    k1=(ef-d*x_dot)/m
    #l1=(ef-b*dot_increment)/m
    k2=(ef-d*(x_dot+dt*k1))/m
    y_t0=x_dot
    y_t1=y_t0+0.5*(k1+k2)*dt
    # speed limitation
    
    speed_limit=1.0
    if abs(y_t1)>speed_limit:
        #print('abs(delta)>0.02')
        #print('y_t1',y_t1)
        y_t1=abs(y_t1)/y_t1*speed_limit
    return y_t1
def admittance_for_obtacleAvoidance(M, m, d, k, theta_dot_previous, theta_c_inFrameW, theta_d_inFrameW, dt):
    theta_ddot=(M-d*theta_dot_previous-k*(theta_c_inFrameW-theta_d_inFrameW))/m
    theta_dot_next=theta_dot_previous+theta_ddot*dt
    theta_next=theta_c_inFrameW+theta_dot_next*dt
    return theta_dot_next

def list_length(l):
    temp=0.0
    for ii in range(0,len(l)):
        temp=temp+l[ii]**2
    return sqrt(temp)
def baseObstacleAvoidance_angularVelocity(distanceThreshold, ObstacleAvoidance_angularVelocity_previous, dt, obstacle_position, carCenter, carTheta, carTheta_d):
    print('obstacle_position',obstacle_position)
    d2=list_length(list_element_minus(carCenter,obstacle_position))
    print('d2',d2)
    if d2>=distanceThreshold:
        print('carTheta_c',carTheta*57.3)
        angularVelocity_to_track_carTheta_d=admittance_for_obtacleAvoidance(0.0, 1.0, 5.0, 10.0, ObstacleAvoidance_angularVelocity_previous, carTheta, carTheta_d, dt)
        angularVelocity_to_track_carTheta_d=angularVelocity_to_track_carTheta_d/abs(angularVelocity_to_track_carTheta_d)*min(0.05,abs(angularVelocity_to_track_carTheta_d))
        return angularVelocity_to_track_carTheta_d
    else:
        half_carLength=0.35
        print('carCenter',carCenter)
        point1=[carCenter[0]+half_carLength*cos(carTheta),carCenter[1]+half_carLength*sin(carTheta)]
        print('point1',point1)
        point3=[carCenter[0]-half_carLength*cos(carTheta),carCenter[1]-half_carLength*sin(carTheta)]
        print('point3',point3)

        v1=list_element_minus(point1,obstacle_position)
        v2=list_element_minus(carCenter,obstacle_position)
        v3=list_element_minus(point3,obstacle_position)

        len_v1=list_length(v1)
        len_v2=list_length(v2)
        len_v3=list_length(v3)

        
        theta1=acos(list_dot_multiply(v1,v2)/(len_v1*len_v2))
        print('theta1',theta1)
        theta3=acos(list_dot_multiply(v1,v3)/(len_v1*len_v3))
        print('theta3',theta3)

        d1=len_v1
        d3=len_v3
        k1=10.0
        F1=1/d1*k1
        k3=10.0
        F3=1/d3*k3
        M1=F1*len_v2*cos(theta1)
        M3=F3*len_v2*cos(theta3)
        M=M1-M3
        vector_point3_to_point1=list_element_minus(point1,point3)
        print('vector_point3_to_point1',vector_point3_to_point1)
        vector_point3_to_obstacle=list_element_minus(obstacle_position,point3)
        print('vector_point3_to_obstacle',vector_point3_to_obstacle)
        print('np.cross(vector_point3_to_point1,vector_point3_to_obstacle)',np.cross(vector_point3_to_point1,vector_point3_to_obstacle))
        if np.cross(vector_point3_to_point1,vector_point3_to_obstacle)>0:
            M=-M
        print('M',M)
        ObstacleAvoidance_angularVelocity_next=admittance_for_obtacleAvoidance(M, 0.5, 5.0, 5.0, ObstacleAvoidance_angularVelocity_previous, carTheta, carTheta_d, dt)
        print('ObstacleAvoidance_angularVelocity_next',ObstacleAvoidance_angularVelocity_next)
        #angularAcc=M/rotationalInertia
        #ObstacleAvoidance_angularVelocity_next=ObstacleAvoidance_angularVelocity_previous+angularAcc*dt
        ObstacleAvoidance_angularVelocity_next=ObstacleAvoidance_angularVelocity_next/abs(ObstacleAvoidance_angularVelocity_next)*min(0.05,abs(ObstacleAvoidance_angularVelocity_next))
        return ObstacleAvoidance_angularVelocity_next
def PI_inctrement_to_TCP_displacement_wholbody(PI_increment, currentP, init_contact_P, hingeCenter, Radius, dt):
    # theta
    #print('init_contact_P',init_contact_P)
    #print('PI_increment=',PI_increment*0.002)
    door_angle_increment=(PI_increment*1.0)/Radius

    door_init_vector=[init_contact_P[0]-hingeCenter[0],init_contact_P[1]-hingeCenter[1]]
    door_current_vector=[currentP[0]-hingeCenter[0],currentP[1]-hingeCenter[1]]
    # eta
    len_door_init_vector=sqrt(door_init_vector[0]**2+door_init_vector[1]**2)
    len_door_current_vector=sqrt(door_current_vector[0]**2+door_current_vector[1]**2)
    door_angle_already_opened=acos((door_init_vector[0]*door_current_vector[0]+door_init_vector[1]*door_current_vector[1])/(len_door_init_vector*len_door_current_vector))
    #print('len_door_init_vector*len_door_current_vector',len_door_init_vector*len_door_current_vector)
    # alpha
    cord_angle=(pi-door_angle_increment)/2

    # beta
    #print('door_angle_increment=',door_angle_increment,'cord_angle=',cord_angle,'door_angle_already_opened=',door_angle_already_opened)
    beta=cord_angle-door_angle_already_opened
    cord_length=(PI_increment*1.0)
    #cord_length=2*Radius*sin(door_angle_increment/2.0)
    if beta<pi/2:
        x_displacement=cord_length*sin(beta)
        y_displacement=-cord_length*cos(beta)
    else:
        # not yet test
        x_displacement, y_displacement = 0.0, 0.0
        print('beta>pi/2,beta=',beta,'. door is opening larger then 90 degrees. coding is not yet finished')
    #print('x_displacement, y_displacement=',x_displacement, y_displacement)
    return x_displacement, y_displacement
t_delta_threthold=0.0006
def Compute_direction_for_the_next_step(t0, t1):
    t_delta_out=[0.0,0.0,0.0]
    t_delta=list_element_minus(t1,t0)
    for ii in range(0,3):
        if t_delta[ii]!=0.0:
            if abs(t_delta[ii])>t_delta_threthold:
                t_delta_out[ii]=t_delta_threthold*abs(t_delta[ii])/t_delta[ii]
            else:
                t_delta_out[ii]=t_delta[ii]
    #print(t_delta_out)
    return list_norm(list_element_add(t0, t_delta_out))
def R2orientationVelocity(R0,R1):
    r1=np.array(R1)
    r0=np.array(R0)
    dt=0.01
    r_dot=(r1-r0)/dt
    omiga_cross=np.dot(r_dot,r1)
    omiga=[omiga_cross[2,1],omiga_cross[0,2],omiga_cross[1,0]]
    return omiga
class getWholeBodyState():
    def __init__(self):
        self.odom_data=Odometry()
        #rospy.init_node('odom_listener',anonymous=True)
        rate=rospy.Rate(100)
        rospy.Subscriber('/mobile_base/odom',Odometry,self.Odometry_callback)
        #time.sleep(1)
        
    def get_rz(self,qz,qw):
        if qz != 0:
            rz=acos(qw)*2.0*abs(qz)/qz
        else:
            rz=0.0
        return rz

    def Odometry_callback(self,data):
        self.odom_data=data

    def get_mobile_monipulator_positions(self):
        base_orientation_rz=self.get_rz(self.odom_data.pose.pose.orientation.z, self.odom_data.pose.pose.orientation.w)#yu shi jie zuo biao x zhou de jiajiao,nishizhen wei zheng. -pi ~ pi
        ur5e_J_position=rtde_r.getActualQ()
        return [self.odom_data.pose.pose.position.x, self.odom_data.pose.pose.position.y, base_orientation_rz, 
                ur5e_J_position[0], ur5e_J_position[1], ur5e_J_position[2], 
                ur5e_J_position[3], ur5e_J_position[4], ur5e_J_position[5]]

    def get_mobile_monipulator_velocities(self):
        ur5e_J_velocities=rtde_r.getActualQd()
        return [self.odom_data.twist.twist.linear.x, self.odom_data.twist.twist.linear.y, self.odom_data.twist.twist.angular.z,
                ur5e_J_velocities[0], ur5e_J_velocities[1], ur5e_J_velocities[2], 
                ur5e_J_velocities[3], ur5e_J_velocities[4], ur5e_J_velocities[5]]

    def main_body_go(self):
        while not rospy.is_shutdown():
            print(self.get_mobile_monipulator_positions())
            print(self.get_mobile_monipulator_positions()[2]*57.3)
            print(self.get_mobile_monipulator_velocities())
            time.sleep(0.2)

class cmdWholeBodyVelocity():
    def __init__(self):
        #rospy.init_node('base_twist_publish',anonymous=True)
        self.PUB=rospy.Publisher('/mobile_base/cmd_vel',Twist,queue_size=1)
        self.pub_twist=Twist()
        self.pub_twist.linear.x=0.0
        self.pub_twist.angular.z=0.0

    def cmd_whole_body_velocity(self, base_x, base_y, base_rz, j1, j2, j3, j4, j5, j6):
        self.pub_twist.linear.x=base_x
        self.pub_twist.angular.z=base_rz
        self.PUB.publish(self.pub_twist)
        rtde_c.speedJ([j1, j2, j3, j4, j5, j6],1.0,0.002)
def get_Angle_between_two_vectors(v1,v2):
    #print('v1',v1)
    #print('v2',v2)
    #print('======',(v1[0]*v2[0]+v1[1]*v2[1])/(sqrt(v1[0]**2+v1[1]**2)*sqrt(v2[0]**2+v2[1]**2)))
    absA=acos((v1[0]*v2[0]+v1[1]*v2[1])/(sqrt(v1[0]**2+v1[1]**2)*sqrt(v2[0]**2+v2[1]**2)))
    pm=np.sign(np.cross(v1,v2))
    angle=pm*absA
    return angle
def get_Angle_between_two_vectors_3D(v1,v2):
    print('v1',v1)
    print('v2',v2)
    #print('======',(v1[0]*v2[0]+v1[1]*v2[1])/(sqrt(v1[0]**2+v1[1]**2)*sqrt(v2[0]**2+v2[1]**2)))
    absA=acos((v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])/(sqrt(v1[0]**2+v1[1]**2+v1[2]**2)*sqrt(v2[0]**2+v2[1]**2+v2[2]**2)))
    pm=np.sign(np.cross(v1,v2))
    angle=pm*absA
    return angle
def time_optimal_for_1D_tracking(x1_tf, x2_tf, x1_t0, x2_t0, abs_acc):
    #判断t0的状态在tf的状态的什么位置
    string=''
    '''mini1=0.5/abs_acc_mini*x2_t0**2+x1_tf-d
    mini2=0.5/(-abs_acc_mini)*x2_t0**2+x1_tf+d
    isIN=False
    if x1_t0>mini1 and x1_t0<mini2:
        abs_acc=abs_acc_mini
        string=string+'abs_acc_mini  |'
        isIN=True'''
    if abs(x1_tf-x1_t0)+abs(x2_tf-x2_t0)<0.001:
        abs_acc=0.001
    if x2_t0>x2_tf:#,走abs_acc<0
        string=string+'x2_t0>x2_tf  |'
        t_minus=(x2_t0-x2_tf)/(-abs_acc)
        temp=0.5*(-abs_acc)*t_minus**2+x2_t0*t_minus
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
class BEZIER():
    def __init__(self):
        #rospy.init_node('base_twist_publish',anonymous=True)
        self.nSection=10000000#根据经验，1米的路程分大于10000000份（7个零）
        self.t_0_previous=0.0
        self.h_t_0_previous=0.0
        self.f_t_x=[]
        self.f_t_y=[]
    def derive_door_opening_init_position(self, carL, carW, door_left, door_right, flag, gap_to_the_left, distance_to_the_front):
        print('calculating the door-opening init position.')
        if flag=='hinge_on_the_right':
            a=carW/2.0+gap_to_the_left
            b=carL/2.0+distance_to_the_front
            doorW=sqrt((door_left[0]-door_right[0])**2+(door_left[1]-door_right[1])**2)
            doorAngle=atan2(door_left[1]-door_right[1],door_left[0]-door_right[0])
            c=sqrt((doorW-a)**2+b**2)
            cAngle=acos((doorW-a)/c)+doorAngle
            carCx=door_right[0]+c*cos(cAngle)
            carCy=door_right[1]+c*sin(cAngle)
            carAngle=-(pi/2-doorAngle)
            frameW_frameDoor_T=np.array([[cos(carAngle),-sin(carAngle),carCx],
                                        [sin(carAngle),cos(carAngle),carCy],
                                        [0.0,0.0,1.0]])
        else:
            print('coding is not finished')
            pass
        return carCx, carCy, carAngle, frameW_frameDoor_T

    def bezier_tracjectory_sparce_generator(self, p0x,p0y,initA,p3x,p3y,endA):
        print('generating a bezier trajectory.')
        L=sqrt((p0x-p3x)**2+(p0y-p3y)**2)
        section=L/2.5
        p1x=p0x+section*cos(initA)
        p1y=p0y+section*sin(initA)
        p2x=p3x-section*cos(endA)
        p2y=p3y-section*sin(endA)
        n=self.nSection#粗略估算一下，如果相距2米，2米分成2000份，每份就是1豪米。也就是说，相距不能太大。要么把n增大。
        self.f_t_x=[]
        self.f_t_y=[]
        for ii in range(0,n+1):
            t=ii/n
            bezier=[p3x*t**3 - p0x*(t - 1)**3 + 3*p1x*t*(t - 1)**2 - p2x*t**2*(3*t - 3),
                    p3y*t**3 - p0y*(t - 1)**3 + 3*p1y*t*(t - 1)**2 - p2y*t**2*(3*t - 3)]
            self.f_t_x.append(bezier[0])
            self.f_t_y.append(bezier[1])
        print('calculating the bezier length.')
        bezier_total_length=0.0
        for ii in range(0,n):
            bezier_total_length=bezier_total_length+sqrt((self.f_t_x[ii+1]-self.f_t_x[ii])**2+(self.f_t_y[ii+1]-self.f_t_y[ii])**2)
        return p1x,p1y,p2x,p2y, bezier_total_length

    def bezier_bisection_calculate_from_s_to_t(self, s,S,count):
        t=s/S
        upper=min(self.t_0_previous+0.01,1.0)
        lower=self.t_0_previous
        step_previous=round(self.t_0_previous*self.nSection)
        for i_iter in range(0,100):
            t=(lower+upper)/2.0
            step=round(t*self.nSection)
            #print('step',step,'t',t)
            h_t=self.h_t_0_previous
            for ii in range(step_previous,step):
                h_t=h_t+sqrt((self.f_t_x[ii+1]-self.f_t_x[ii])**2+(self.f_t_y[ii+1]-self.f_t_y[ii])**2)
            F_t=h_t-s
            if abs(F_t)<0.00000005:#精度跟轨迹长度、分段数、加速度等都有关。
                self.t_0_previous=t
                self.h_t_0_previous=h_t
                print(count,i_iter,step_previous,self.t_0_previous)
                return t
            if F_t>0.0:# t_optimal<t<upper
                upper=t
            else:#F_t<0--upper<t<t_optimal
                lower=t
        print(count,i_iter,step_previous,self.t_0_previous)
        self.t_0_previous=t
        self.h_t_0_previous=h_t
        return t
    def bezier_tangent_at_t(self, t,p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y):
        tangent_x=3*p1x*(t - 1)**2 - 3*p0x*(t - 1)**2 - 3*p2x*t**2 + 3*p3x*t**2 + 3*p1x*t*(2*t - 2) - 2*p2x*t*(3*t - 3)
        tangent_y=3*p1y*(t - 1)**2 - 3*p0y*(t - 1)**2 - 3*p2y*t**2 + 3*p3y*t**2 + 3*p1y*t*(2*t - 2) - 2*p2y*t*(3*t - 3)
        length=sqrt(tangent_x**2+tangent_y**2)
        norm_direction=[tangent_x/length, tangent_y/length]
        angle=atan2(norm_direction[1],norm_direction[0])
        return norm_direction, angle



def forceV_to_eeR_frame0(frame0_T1):
    frame0_tcp_z=list_norm([frame0_T1[0],frame0_T1[1],frame0_T1[2]])
    frame0_tcp_y=list_norm([0.0,0.0,1.0])
    frame0_tcp_x=list_norm(np.cross(frame0_tcp_y, frame0_tcp_z))
    frame0_R=np.array([[frame0_tcp_x[0], frame0_tcp_y[0], frame0_tcp_z[0]],
                        [frame0_tcp_x[1], frame0_tcp_y[1], frame0_tcp_z[1]],
                        [frame0_tcp_x[2], frame0_tcp_y[2], frame0_tcp_z[2]]])
    return frame0_R
def get_door_opened_angle(ii):
    '''max_angle=120.0
    desiredA=min(max_angle*pi/180.0, ii*0.02*pi/180.0)
    if desiredA>=max_angle*pi/180.0:
        desiredA_disturbance=desiredA-60.0*pi/180.0
    else:
        desiredA_disturbance=desiredA+0.0'''
    door_angle_for_traj2_end=2.1
    desiredA_disturbance=min(door_angle_for_traj2_end,ii*0.001)
    if desiredA_disturbance>door_angle_for_traj2_end-0.02:
        desiredA_disturbance=desiredA_disturbance-abs((door_angle_for_traj2_end-0.02)*sin((ii-(door_angle_for_traj2_end-0.02)*1000)*0.0005))


    return desiredA_disturbance



def get_base_target_position(door_angle,carW,carL,hingeCx,hingeCy,distance):
    eta=pi/2.0-door_angle
    k=tan(eta)
    frameDoor_Ay=-carW/2.0
    frameDoor_Ax=(frameDoor_Ay-hingeCy+k*hingeCx-distance*sqrt(1+k**2))/k
    frameDoor_xb=frameDoor_Ax-carL/2.0
    frameDoor_yb=frameDoor_Ay+carW/2.0
    #print('door_angle',door_angle,'hingeCx',hingeCx,'hingeCy',hingeCy,'eta',eta,'k',k,'frameDoor_Ay',frameDoor_Ay,'frameDoor_Ax',frameDoor_Ax,'frameDoor_xb',frameDoor_xb,'frameDoor_yb',frameDoor_yb)
    return [frameDoor_xb, frameDoor_yb,1.0], frameDoor_Ax, frameDoor_Ay
def get_stage2_desire_position(door_opened_angle, frameD_traj2_center_x,frameD_traj2_center_y, stage2_traj2_radius_to_A, frameD_door_hinge_x,frameD_door_hinge_y,distance,carW,carL):
    #print('stage2_traj2_radius_to_A',stage2_traj2_radius_to_A)
    eta=pi/2.0-door_opened_angle
    k=tan(eta)
    #print('k',k)
    a=1.0+k**2
    temp=frameD_door_hinge_y-k*frameD_door_hinge_x+distance*sqrt(k**2+1.0)
    b=-2.0*frameD_traj2_center_x+2.0*k*temp-2.0*k*frameD_traj2_center_y
    c=frameD_traj2_center_x**2+temp**2-2.0*temp*frameD_traj2_center_y+frameD_traj2_center_y**2-stage2_traj2_radius_to_A**2
    #print('a',a)

    if b**2-4.0*a*c<0:
        print('b**2-4.0*a*c<0')
        print('eta',eta)
        print('k',k)
    Ax=(-b+sqrt(b**2-4.0*a*c))/(2.0*a)
    Ay=k*Ax+temp
    
    Ax0=frameD_traj2_center_x+carL/2.0
    Ay0=0.0-carW/2.0
    O2Ax0=[Ax0-frameD_traj2_center_x, Ay0-frameD_traj2_center_y]
    O2Ax=[Ax-frameD_traj2_center_x, Ay-frameD_traj2_center_y]
    #print('O2Ax0',O2Ax0)
    #print('O2Ax',O2Ax)
    phi=get_Angle_between_two_vectors(O2Ax,O2Ax0)
    #phi=max(0.0,phi)#实际是不对了，但近似处理了，不知道为什么刚开始转时为什么会有-3.9度的误差。这样处理会导致在第一、二阶段转换处，门在3.9度的范围内运动时，底盘不会动。
    #acos(list_dot_multiply(O2Ax0,O2Ax)/(list_length(O2Ax0)*list_length(O2Ax)))

    frameD_xb_0=frameD_traj2_center_x
    frameD_yb_0=frameD_traj2_center_y+stage2_traj2_radius#=0.0
    #print('phi',phi*57.3)
    frameD_xb=frameD_xb_0+stage2_traj2_radius*sin(phi)
    frameD_yb=frameD_yb_0-(stage2_traj2_radius-stage2_traj2_radius*cos(phi))
    return frameD_xb, frameD_yb, phi, Ax, Ay

def get_door_opened_angle_threshold1(pass_door_front, carL, carW, frameD_traj2_center_x, frameD_traj2_center_y, frameD_doorL_x,frameD_doorL_y, distance):
    #print(pass_door_front, carL, carW, frameD_hinge_x, frameD_hinge_y, distance)
    Ax=frameD_traj2_center_x+carL/2.0
    Ay=-carW/2.0
    Cx=frameD_traj2_center_x-carL/2.0-pass_door_front
    Cy=frameD_traj2_center_y-carL/2.0-pass_door_front
    print('Ax',Ax,'Ay',Ay,'Cx',Cx,'Cy',Cy)
    CA=[Ax-Cx,Ay-Cy]
    CD=[frameD_doorL_x-Cx,frameD_doorL_y-Cy]
    angle_ACD=get_Angle_between_two_vectors(CA,CD)
    print('angle_ACD',angle_ACD*57.3)
    length_AC=list_length(CA)
    angle_ACB=asin(distance/length_AC)
    print('angle_ACB',angle_ACB*57.3)
    angle_BCD=angle_ACD+angle_ACB
    return angle_BCD



if __name__ == "__main__":
    print('start!')

    pose_init=[0.1973983069853642, -0.30385074609865037, 0.6959411818669748, 1.5708023950890553, -2.350036156691199e-05, -1.862457754068252e-05]
    #rtde_c.moveL(pose_init,0.1,0.1)
    joint_init=[2.1394758224487305, -2.140566965142721, 1.9454715887652796, -2.9405275783934535, 4.144657135009766, -3.131549660359518]
    rtde_r.getJointTorques()
    rtde_c.moveJ(joint_init,0.2,0.1)

    
    print('finished initializing ur5e.')
    time.sleep(1)
    '''ur5e_init_Q=rtde_r.getActualQ()
    print('ur5e_init_Q',ur5e_init_Q)
    pose_init=rtde_r.getActualTCPPose()
    print('pose_init',pose_init)
    time.sleep(1)
    for testii in range(0,1000):
        rtde_c.moveL([pose_init[0]+testii/20000.0, pose_init[1]+testii/20000.0, pose_init[2]+testii/20000.0, pose_init[3], pose_init[4], pose_init[5]],0.1,0.1)
        print('testing', testii)
        time.sleep(0.001)
    print('finished testing')
    time.sleep(100)'''

    ######################down#####################################
    frame0_T1=[-0.0,-1.0,0.0]#在ur5e基坐标系下的方向向量
    frame0_T0=frame0_T1
    
    axis, angle, frame0_R1  = axixAngle_to_rotationMatrix_anxisAngle(frame0_T1)
    print('frame0_R1',frame0_R1)
    length=sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
    unit_axis=[axis[0]/length,axis[1]/length,axis[2]/length]
    axis_angle=[angle*unit_axis[0],angle*unit_axis[1],angle*unit_axis[2]]
    axis_angle_0=axis_angle
    rtde_c.moveL([pose_init[0],pose_init[1]+0.08,pose_init[2], axis_angle[0], axis_angle[1], axis_angle[2]],0.1,0.1)
    #print('finished 1')
    #rtde_c.moveL([pose_init[0],pose_init[1]+0.3,pose_init[2], axis_angle[0], axis_angle[1], axis_angle[2]],0.01,0.01)
    frame0_R1=forceV_to_eeR_frame0(frame0_T1)
    print('frame0_R1',frame0_R1)
    frame0_R0=frame0_R1#在ur5e基坐标系下的旋转矩阵
    #######################up###############################
    print('finished init')
    time.sleep(1)

    ##test down##
    '''rospy.init_node('whole_body_open_a_door',anonymous=True)
    wholebodystate=getWholeBodyState()
    cmdVelocities=cmdWholeBodyVelocity()
    ur5e_car = UR5e_car_kinematics()
    time.sleep(1)
    q_joint_init=wholebodystate.get_mobile_monipulator_positions()
    print('q_joint_init',q_joint_init)
    ee = ur5e_car.get_end_effector_posture(q_joint_init)
    print('ee',ee)
    for ii in range(0,20):
        q_joint_2=q_joint_init
        q_joint_2[0]=q_joint_init[0]+ii/100.0-q_joint_init[0]
        q_joint_2[1]=0.0
        q_joint_2[2]=0.0
        ee1 = ur5e_car.get_end_effector_posture(q_joint_2)
        print('ee1',ee1)
        time.sleep(0.01)
    
    sys.exit()'''
    ###test up###





    print('start!')
    rospy.init_node('whole_body_open_a_door',anonymous=True)
    wholebodystate=getWholeBodyState()
    cmdVelocities=cmdWholeBodyVelocity()
    ur5e_car = UR5e_car_kinematics()
    #wholebody_kinematics=WB_KINEMATICS()
    print('initializing ros node.')
    bezier=BEZIER()
    time.sleep(1)

    q_joint=wholebodystate.get_mobile_monipulator_positions()
    q_velocity=wholebodystate.get_mobile_monipulator_velocities()
    print('q_joint',q_joint)

    second=10
    steps=second*100
    init_flag=True
    dt=0.01

    inverse=1.0

    door_left=[1.45,0.225]#门的参数可以通过视觉识别。目前仅用手工测量。
    door_right=[1.45,-0.775]
    '''door_left=[0.7+0.375,0.26]#门的参数可以通过视觉识别。目前仅用手工测量。
    door_right=[0.7+0.375,-1.0-(-0.26)]'''
    '''door_left=[1.0,0.36]#门的参数可以通过视觉识别。目前仅用手工测量。
    door_right=[1.0,-0.64]'''
    door_length=sqrt((door_left[0]-door_right[0])**2+(door_left[1]-door_right[1])**2)
    carL=0.75
    carW=0.52
    gap_to_the_left=0.1#这个数不能太大，要根据门的宽度来定
    distance_to_the_front=0.1#这个数不能太小，要根据门的宽度来定
    carC_init_x, carC_init_y, car_init_Angle, frameW_frameDoor_T=bezier.derive_door_opening_init_position(carL, carW, door_left, door_right, 'hinge_on_the_right', gap_to_the_left, distance_to_the_front)
    p0x=q_joint[0]
    p0y=q_joint[1]
    initA=q_joint[2]
    p3x=carC_init_x
    p3y=carC_init_y
    endA=car_init_Angle
    p1x,p1y,p2x,p2y,bezier_total_length=bezier.bezier_tracjectory_sparce_generator(p0x,p0y,initA,p3x,p3y,endA)
    
    x2_t0=0.0
    x1_t0=0.0
    abs_acc=0.2
    x1_tf=bezier_total_length
    S=bezier_total_length
    base_rz_dot_previous=0.0
    s_record=[]
    t_record=[]
    angle0_record=[]
    base_x_dot_record=[]
    base_rz_dot_cmd_record=[]
    ii_record=[]
    print('start to compute velocities.')
    for ii in range(0,steps):
        #q_joint=wholebodystate.get_mobile_monipulator_positions()
        #q_velocity=wholebodystate.get_mobile_monipulator_velocities()
        
        s=x1_t0
        t_0=bezier.bezier_bisection_calculate_from_s_to_t(s,S,ii)
        norm_direction0, angle0=bezier.bezier_tangent_at_t(t_0,p0x,p0y,p1x,p1y,p2x,p2y,p3x,p3y)
        s_record.append(s)
        t_record.append(s)
        angle0_record.append(angle0)

        acc,string=time_optimal_for_1D_tracking(x1_tf, 0.0, x1_t0, x2_t0, abs_acc)
        x2_t0=x2_t0+acc*dt
        x1_t0=x1_t0+x2_t0*dt
        base_x_dot=x2_t0

        base_x_dot_record.append(base_x_dot)
        
        if abs(S-s)<0.000001:
            print('ii=',ii,'and break.')
            break
    
    for ii2 in range(0,ii-1):
        base_rz_dot=(angle0_record[ii2+1]-angle0_record[ii2])/dt
        alpha=1.0
        base_rz_dot_cmd=alpha*base_rz_dot+(1-alpha)*base_rz_dot_previous
        base_rz_dot_previous=base_rz_dot
        base_rz_dot_cmd_record.append(base_rz_dot_cmd)
        ii_record.append(ii2)
    print('finished all velocities calculation.')
    palette = pyplot.get_cmap('Set1')
    plt.clf()
    plt.subplot(121)
    plt.plot(ii_record, base_rz_dot_cmd_record,color=palette(1),linewidth=1.0,label="base_rz_dot_cmd_record")
    
    plt.subplot(122)
    plt.plot(bezier.f_t_x, bezier.f_t_y,color=palette(1),linewidth=4.0,label="f_t")
    plt.plot([p0x,p1x,p2x,p3x],[p0y,p1y,p2y,p3y],color='r')
    plt.plot([door_left[0],door_right[0]],[door_left[1],door_right[1]],color='b')
    plt.axis('equal')
    plt.show()


    cmd_num=len(base_rz_dot_cmd_record)
    for ii2 in range(0, cmd_num):
        t0=time.time()
        base_x_dot=base_x_dot_record[ii2]
        base_rz_dot_cmd=base_rz_dot_cmd_record[ii2]
        cmdVelocities.cmd_whole_body_velocity(base_x_dot, 0.0, base_rz_dot_cmd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        if ii>=steps-1:
            for iii in range(0,50):
                if abs(base_x_dot)>0.01:
                    base_x_to_stop=base_x_dot*(50.0-iii)/50.0
                    base_rx_to_stop=base_rz_dot_cmd*(50.0-iii)/50.0
                    cmdVelocities.cmd_whole_body_velocity(base_x_to_stop, 0.0, base_rx_to_stop, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    print(base_x_to_stop)
                    time.sleep(0.01)
                else:
                    cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            rtde_c.speedStop()
            print('stop!')
            
        t1=time.time()
        if t1-t0<0.01:
            #print('ii = '+str(iiprint('stop!')
            time.sleep(0.01-(t1-t0))
        else:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx  |',t1-t0)
    
    print('already at the front of the door.')

    #重新确认一下车的位置，当前位置设置为{d}的原点;在图上显示两条轨迹会断开，距离相差5mm左右（与bezier的长度有关）
    num=10
    xb=0.0
    yb=0.0
    rz=0.0
    for ii3 in range(0,num):
        q_joint_at_the_front=wholebodystate.get_mobile_monipulator_positions()
        xb=xb+q_joint_at_the_front[0]
        yb=yb+q_joint_at_the_front[1]
        rz=rz+q_joint_at_the_front[2]
        time.sleep(0.1)
    print('previous frameW_frameDoor_T ', frameW_frameDoor_T)
    xb=xb/num
    yb=yb/num
    rz=rz/num
    frameW_frameDoor_T=np.array([[cos(rz),-sin(rz),xb],
                                    [sin(rz),cos(rz),yb],
                                    [0.0,0.0,1.0]])
    print('re-located frameW_frameDoor_T ', frameW_frameDoor_T)
    frameW_frameD_R=np.mat([[frameW_frameDoor_T[0,0],frameW_frameDoor_T[0,1]],
                            [frameW_frameDoor_T[1,0],frameW_frameDoor_T[1,1]]])
    frameW_frameD_traslate=np.mat([[frameW_frameDoor_T[0,2]],[frameW_frameDoor_T[1,2]]])
    frameD_frameW_R=np.transpose(frameW_frameD_R)
    frameD_frameW_translate=-np.matmul(frameD_frameW_R,frameW_frameD_traslate)
    frameD_frameW_T=np.mat([[frameD_frameW_R[0,0],frameD_frameW_R[0,1],frameD_frameW_translate[0,0]],
                            [frameD_frameW_R[1,0],frameD_frameW_R[1,1],frameD_frameW_translate[1,0]],
                            [0.0,0.0,1.0]])
    frameD_frameW_T_test=pinv(frameW_frameDoor_T)
    print('frameW_frameDoor_T\n',frameW_frameDoor_T)
    print('frameD_frameW_T\n',frameD_frameW_T)
    print('frameD_frameW_T_test\n',frameD_frameW_T_test)
    print('[door_right[0],door_right[1],1.0]',[door_right[0],door_right[1],1.0])
    frameD_door_hinge=np.matmul(frameD_frameW_T,np.mat([[door_right[0]],[door_right[1]],[1.0]]))
    frameD_door_left=np.matmul(frameD_frameW_T,np.mat([[door_left[0]],[door_left[1]],[1.0]]))
    print('frameD_door_hinge',frameD_door_hinge)


    distance=distance_to_the_front#这个距离不能大于等于车的右边缘与门的右边的距离，小一些比较好，但太小的话小车的反应距离也不够。
    frameD_xb_record=[]
    frameD_yb_record=[]
    frameW_xb_record=[]
    frameW_yb_record=[]
    x1_t0=0.0#两个stage,位置要连续
    x2_t0=0.0#速度也要保持连续
    #while not rospy.is_shutdown():
    iii1_num=100*200

    isPassedTheEndOfStage1=False
    isVerticalToDoor=True
    pass_door_front=-0.13
    frameD_traj2_center_x=frameD_door_hinge[0,0]+carL/2.0+pass_door_front
    frameD_traj2_center_y=-(door_length-gap_to_the_left-carW/2.0-pass_door_front-carL/2.0)#frameD_door_hinge[1,0]+carL/2.0+pass_door_front
    frameW_traj2_center=np.matmul(frameW_frameDoor_T,[frameD_traj2_center_x, frameD_traj2_center_y, 1.0])
    door_width=sqrt((door_left[0]-door_right[0])**2+(door_left[1]-door_right[1])**2)
    stage2_traj2_radius=door_width-gap_to_the_left-carW/2.0-(carL/2.0+pass_door_front)
    Ax0=frameD_traj2_center_x+carL/2.0
    Ay0=0.0-carW/2.0
    O2Ax0=[Ax0-frameD_traj2_center_x, Ay0-frameD_traj2_center_y]
    stage2_traj2_radius_to_A=list_length(O2Ax0)
    #print('stage2_traj2_radius',stage2_traj2_radius)
    stage2_traj2_length=pi*stage2_traj2_radius/2.0
    isTurning=False
    #print('frameD_door_hinge[0,0], frameD_door_hinge[1,0]',frameD_door_hinge[0,0], frameD_door_hinge[1,0])
    door_opened_angle_threshold1=get_door_opened_angle_threshold1(pass_door_front, carL, carW, frameD_traj2_center_x, 
                                                                    frameD_traj2_center_y, frameD_door_left[0,0], frameD_door_left[1,0], distance)
    print('door_opened_angle_threshold1',door_opened_angle_threshold1,door_opened_angle_threshold1*57.3)
    frameD_Ax_record=[]
    frameD_Ay_record=[]




    #########################down########################################
    
    
    init_flag=True
    
    F_raw_history_x=[]
    F_raw_history_y=[]
    F_raw_history_z=[]
    T_raw_history_x=[]
    T_raw_history_y=[]
    T_raw_history_z=[]
    sliding_width=5
    rtde_c.zeroFtSensor()#刚开始不要让末端有接触力
    for ii1 in range(0,sliding_width+1):
        F_raw=rtde_r.getActualTCPForce()
        F_raw_history_x.append(F_raw[0])
        F_raw_history_y.append(F_raw[1])
        F_raw_history_z.append(F_raw[2])
        T_raw_history_x.append(F_raw[3])
        T_raw_history_y.append(F_raw[4])
        T_raw_history_z.append(F_raw[5])
    linearVelocity_x_previous=0.0
    linearVelocity_y_previous=0.0
    linearVelocity_z_previous=0.0

    angularVelocity_x_previous=0.0
    angularVelocity_y_previous=0.0
    angularVelocity_z_previous=0.0

    F_raw_history_x2=[]
    F_raw_history_y2=[]
    F_raw_history_z2=[]
    sliding_width2=150

    for ii1 in range(0,sliding_width2+1):
        F_raw=rtde_r.getActualTCPForce()
        F_raw_history_x2.append(F_raw[0])
        F_raw_history_y2.append(F_raw[1])
        F_raw_history_z2.append(F_raw[2])

    isInDoor=False
    isContact=False
    first_cantact_flag=True
    hingeCenter=[frameD_door_hinge[0,0], frameD_door_hinge[1,0]]#changed
    init_contact_P=[hingeCenter[0]+100.0,hingeCenter[0]] #[x, y]
    force_threshold=8.0

    count_record=[]
    ef_x_record=[]
    delta_x_record=[]
    F_filter_in_frame_w_record=[]
    linearVelocity_previous=0.0




    ##############################up################################################


    
    base_rz_dot_previous=0.0
    for iii1 in range(0,iii1_num):
        t0=time.time()
        
        #print('door_opened_angle',door_opened_angle)
        q_joint=wholebodystate.get_mobile_monipulator_positions()
        #frameD_base_position=np.matmul(frameD_frameW_T,np.mat([[q_joint[0]],[q_joint[1]],[1.0]]))
        #if frameD_base_position[0]>=frameD_traj2_center_x:
        
        ############################down##########################################################
        frameD_q_joint=q_joint
        temp=np.matmul(frameD_frameW_T,np.mat([[q_joint[0]],[q_joint[1]],[1.0]]))
        frameD_q_joint[0]=temp[0,0]#frameD_base_position[0,0]
        frameD_q_joint[1]=temp[1,0]#frameD_base_position[1,0]
        frameD_q_joint[2]=q_joint[2]-rz
        frameD_ee = ur5e_car.get_end_effector_posture(frameD_q_joint)
        #frameD_ee_xy=np.matmul(frameD_frameW_T,np.mat([[float(ee[0,3])],[float(ee[1,3])],[1.0]]))
        '''ee = ur5e_car.get_end_effector_posture(q_joint)
        frameD_ee_xy=np.matmul(frameD_frameW_T,np.mat([[float(ee[0,3])],[float(ee[1,3])],[1.0]]))'''
        


        F_raw=rtde_r.getActualTCPForce()
        ii2=0
        for ii2 in range(0,sliding_width):
            F_raw_history_x[ii2]=F_raw_history_x[ii2+1]
            F_raw_history_y[ii2]=F_raw_history_y[ii2+1]
            F_raw_history_z[ii2]=F_raw_history_z[ii2+1]
            T_raw_history_x[ii2]=T_raw_history_x[ii2+1]
            T_raw_history_y[ii2]=T_raw_history_y[ii2+1]
            T_raw_history_z[ii2]=T_raw_history_z[ii2+1]
        F_raw_history_x[sliding_width]=F_raw[0]
        F_raw_history_y[sliding_width]=F_raw[1]
        F_raw_history_z[sliding_width]=F_raw[2]
        T_raw_history_x[sliding_width]=F_raw[3]
        T_raw_history_y[sliding_width]=F_raw[4]
        T_raw_history_z[sliding_width]=F_raw[5]
        F_filter=[sum(F_raw_history_x)/(sliding_width+1),sum(F_raw_history_y)/(sliding_width+1),sum(F_raw_history_z)/(sliding_width+1),
                    sum(T_raw_history_x)/(sliding_width+1),sum(T_raw_history_y)/(sliding_width+1),sum(T_raw_history_z)/(sliding_width+1)]
        ii3=0
        for ii3 in range(0,sliding_width2):
            F_raw_history_x2[ii3]=F_raw_history_x2[ii3+1]
            F_raw_history_y2[ii3]=F_raw_history_y2[ii3+1]
            F_raw_history_z2[ii3]=F_raw_history_z2[ii3+1]
        F_raw_history_x2[sliding_width2]=F_raw[0]
        F_raw_history_y2[sliding_width2]=F_raw[1]
        F_raw_history_z2[sliding_width2]=F_raw[1]
        F_filter2=[sum(F_raw_history_x2)/(sliding_width2+1),sum(F_raw_history_y2)/(sliding_width2+1),sum(F_raw_history_z2)/(sliding_width2+1)]
        

        frameB_frame0_T=np.array([[0.0, -1.0, 0.0, 0.22],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.599],
                                [0.0, 0.0, 0.0, 1.0]])
        frameD_frameB_x=frameD_q_joint[0] #changed
        frameD_frameB_y=frameD_q_joint[1]#changed
        theta=frameD_q_joint[2]#changed
        frameD_frameB_T=np.array([[cos(theta), -sin(theta), 0.0, frameD_frameB_x],#changed
                                [sin(theta), cos(theta), 0.0, frameD_frameB_y],#changed
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
        frameD_frame0_T=np.matmul(frameD_frameB_T,frameB_frame0_T)#changed
        frameD_frame0_R=[[frameD_frame0_T[0,0],frameD_frame0_T[0,1],frameD_frame0_T[0,2]],#changed
                        [frameD_frame0_T[1,0],frameD_frame0_T[1,1],frameD_frame0_T[1,2]],#changed
                        [frameD_frame0_T[2,0],frameD_frame0_T[2,1],frameD_frame0_T[2,2]]]#changed
        F_filter_in_frame_D=np.matmul(frameD_frame0_R,[F_filter[0],F_filter[1],F_filter[2]])#changed
        #print('F_filter_in_frame_w',F_filter_in_frame_D)#changed
        F_filter2_in_frame_D=np.matmul(frameD_frame0_R,[F_filter2[0],F_filter2[1],F_filter2[2]])#changed

        F_filter_len2=sqrt(F_filter2_in_frame_D[0]**2+F_filter2_in_frame_D[1]**2)
        if F_filter_len2>force_threshold:
            frame0_T1=list_norm([-F_filter2[0],-F_filter2[1],0.0])#还是在ur5e基坐标系下的方向向量
        
        ###############test down##########################
        '''frame0_T1=list_norm([-1.0,-1.0,0.0])
        print('frame0_T1',frame0_T1)'''
        ################test up#########################

        #print('frame0_T1',frame0_T1)
        F_filter_len=sqrt(F_filter_in_frame_D[0]**2+F_filter_in_frame_D[1]**2)
        if F_filter_len>force_threshold and first_cantact_flag==True:
            init_contact_P=[float(frameD_ee[0,3]),float(frameD_ee[1,3])]
            radius=sqrt((hingeCenter[0]-init_contact_P[0])**2+(hingeCenter[1]-init_contact_P[1])**2)
            first_cantact_flag=False
            print('init_contact_P',init_contact_P, 'hingeCenter',hingeCenter)
        
        pose_test=rtde_r.getActualTCPPose()
        k_test,phi_test=seperate_angvec([pose_test[3],pose_test[4],pose_test[5]])
        R_test=axixAngle_to_rotationMatrix(k_test, phi_test)
        frame0_TCP_z=list_norm([R_test[0,2],R_test[1,2],0.0])
        #print('frame0_TCP_z',frame0_TCP_z)

        T_next=Compute_direction_for_the_next_step(list_norm(frame0_TCP_z), list_norm(frame0_T1))
        #print('T_next',T_next)
        #frame0_R1=forceV_to_eeR_frame0(T_next)
        #frame0_orientationVelocity=R2orientationVelocity(frame0_R0,frame0_R1)
        
        frame0_orientationVelocity_rz=get_Angle_between_two_vectors([frame0_TCP_z[0],frame0_TCP_z[1]],[T_next[0],T_next[1]])/0.01
        #print('frame0_orientationVelocity_rz',frame0_orientationVelocity_rz)
        frame0_orientationVelocity=[0.0,0.0,frame0_orientationVelocity_rz]
        
        ##################test down##############################
        '''velocity_X=[0.0, 0.0, 0.0, 0.0, 0.0, frame0_orientationVelocity[2]]
        z=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        wholebody_J = ur5e_car.get_jacobian_lx(frameD_q_joint)
        J_transpose=np.transpose(wholebody_J)
        Weight=np.array([[99999999999999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 99999999999999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        Weignt_inverse=pinv(Weight)

        J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
        joint_v=np.matmul(J_sharp,velocity_X)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)
        j1=joint_v[2]
        j2=joint_v[3]
        j3=joint_v[4]
        j4=joint_v[5]
        j5=joint_v[6]
        j6=joint_v[7]
        cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, j1, j2, j3, j4, j5, j6)
        if iii1>=iii1_num-1:
            cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            rtde_c.speedStop()
            print('stop!')

        ## 用于下一轮
        frame0_T0=T_next
        frame0_R0=frame0_R1

        t1=time.time()
        if t1-t0<0.01:
            #print('ii = '+str(iiprint('stop!')
            time.sleep(0.01-(t1-t0))
        else:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx  |',t1-t0)'''
        ###################test up#######################################



        #orientationVelocity=np.matmul(frameD_frame0_R,frame0_orientationVelocity)#changed, important!
        orientationVelocity=frame0_orientationVelocity
        #delta_rx,delta_ry,delta_rz=orientationDelta[0],orientationDelta[1],orientationDelta[2] #角度计算有问题

        fd_normal2door=-40#min(10.0,ii*0.01)
        f_in_xoy=[F_filter_in_frame_D[0],F_filter_in_frame_D[1]]
        #print('f_in_xoy',f_in_xoy)
        pm=np.cross([-1.0,0.0],f_in_xoy)
        door_normal_vector_angle=acos(list_dot_multiply(f_in_xoy,[-1.0,0.0])/list_length(f_in_xoy))*pm/abs(pm)
        #print('door_normal_vector_angle',door_normal_vector_angle*57.3)
        f_normal2door=f_in_xoy[0]/cos(door_normal_vector_angle)#+f_in_xoy[1]*sin(door_normal_vector_angle)
        #print('f_normal2door',f_normal2door)
        ef_normal2door=(f_normal2door-fd_normal2door)#-10.0##-min(10.0,ii*0.1)#
        linearVelocity=pi_forceerror2desireddisplacement2(ef_normal2door, 2.0, 100.0, linearVelocity_previous, 0.01)
        #print('linearVelocity',linearVelocity)
        #delta_normal2door=linearVelocity*0.01
        
        if float(frameD_ee[0,3])>init_contact_P[0]:
            isInDoor=True
            #print('in door')
        else:
            isInDoor=False
            #print('not in door')
        if F_filter_len>force_threshold:
            isContact=True
            #print('contact')
        else:
            isContact=False
            #print('not contact')
        
        wholebody_J = ur5e_car.get_jacobian_lx(frameD_q_joint)#changed,

        
        #############################up###############################################################
        currentP=[float(frameD_ee[0,3]),float(frameD_ee[1,3])]
        #print('currentP',currentP)
        if not isInDoor:
            door_opened_angle=0.0
            #print('door_opened_angle',door_opened_angle)
        else:
            OD=[init_contact_P[0]-frameD_door_hinge[0,0],init_contact_P[1]-frameD_door_hinge[1,0]]
            OD_current=[currentP[0]-frameD_door_hinge[0,0],currentP[1]-frameD_door_hinge[1,0]]
            door_opened_angle=get_Angle_between_two_vectors(OD_current, OD)
            print('door_opened_angle',door_opened_angle)
            #print('init_contact_P',init_contact_P,'currentP',currentP,'frameD_door_hinge',frameD_door_hinge,'door_opened_angle',door_opened_angle*57.3)


        if door_opened_angle>door_opened_angle_threshold1:
            frameD_stage2_xb, frameD_stage2_yb, stage2_phi, frameDoor_Ax, frameDoor_Ay = get_stage2_desire_position(door_opened_angle,frameD_traj2_center_x,frameD_traj2_center_y, stage2_traj2_radius_to_A, frameD_door_hinge[0,0],frameD_door_hinge[1,0],distance,carW,carL)
            frameD_xb_record.append(frameD_stage2_xb)
            frameD_yb_record.append(frameD_stage2_yb)
            frameD_Ax_record.append(frameDoor_Ax)
            frameD_Ay_record.append(frameDoor_Ay)
            frameD_base_target_position=[frameD_stage2_xb, frameD_stage2_yb, 1.0]
            frameW_base_target_position=np.matmul(frameW_frameDoor_T,frameD_base_target_position)
            frameW_xb_record.append(frameW_base_target_position[0])
            frameW_yb_record.append(frameW_base_target_position[1])
            #print('stage2_phi',stage2_phi)
            
            x1_tf=frameD_traj2_center_x+stage2_traj2_radius*stage2_phi
            if stage2_phi>=90.0/180.0*pi:
                iii=0
                for iii in range(0,50):
                    if abs(base_x_dot)>0.01:
                        base_x_to_stop=base_x_dot*(50.0-iii)/50.0
                        base_rz_dot_to_stop=base_rz_dot*(50.0-iii)/50.0
                        cmdVelocities.cmd_whole_body_velocity(base_x_to_stop, 0.0, base_rz_dot_to_stop, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        print(base_x_to_stop)
                        time.sleep(0.01)
                    else:
                        cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                rtde_c.speedStop()
                print('break,','stage2_phi>=90.0')
                print('door_opened_angle',door_opened_angle*57.3)
                print('the second stage finished.')
                break
        else:
            frameD_base_target_position, frameDoor_Ax, frameDoor_Ay=get_base_target_position(door_opened_angle,carW,carL,frameD_door_hinge[0,0],frameD_door_hinge[1,0],distance)
            frameD_xb_record.append(frameD_base_target_position[0])
            frameD_yb_record.append(frameD_base_target_position[1])
            frameD_Ax_record.append(frameDoor_Ax)
            frameD_Ay_record.append(frameDoor_Ay)
            #print(iii1,'frameD_base_target_position',frameD_base_target_position)
            frameW_base_target_position=np.matmul(frameW_frameDoor_T,frameD_base_target_position)
            frameW_xb_record.append(frameW_base_target_position[0])
            frameW_yb_record.append(frameW_base_target_position[1])
            #print(iii1,'frameD_base_target_position',frameW_base_target_position)
            x1_tf=frameD_base_target_position[0]


        if x1_t0>=frameD_traj2_center_x:
            isTurning=True
            #print('frameD_door_hinge',frameD_door_hinge)
            #print('stage2_traj2_radius',stage2_traj2_radius)
            #print('already to the end of stage 1.', 'door_opened_angle=',door_opened_angle)
            isPassedTheEndOfStage1=True
            current_angle=-(x1_t0-frameD_traj2_center_x)/stage2_traj2_radius
            if current_angle>0:
                current_angle=0.0
            acc,string=time_optimal_for_1D_tracking(x1_tf, 0.0, x1_t0, x2_t0, abs_acc)
            x2_t0=x2_t0+acc*dt
            x1_t0=x1_t0+x2_t0*dt
            base_x_dot=x2_t0
            next_angle=-(x1_t0-frameD_traj2_center_x)/stage2_traj2_radius
            if next_angle>0:
                next_angle=0.0
            alpha=0.5
            base_rz_dot=alpha*(next_angle-current_angle)/dt+(1.0-alpha)*base_rz_dot_previous
            base_rz_dot_previous=base_rz_dot
        else:
            isTurning=False
            #print('sim2real error=',x1_t0-frameD_base_position[0])
            isPassedTheEndOfStage1=False
            
            acc,string=time_optimal_for_1D_tracking(x1_tf, 0.0, x1_t0, x2_t0, abs_acc)
            x2_t0=x2_t0+acc*dt
            x1_t0=x1_t0+x2_t0*dt
            base_x_dot=x2_t0
            base_rz_dot=0.0


        Weight=np.array([[99999999999999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 99999999999999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        jointV=rtde_r.getActualQd()
        jointP=rtde_r.getActualQ()
        damping=-5.0
        stiffness=-10.0
        z_v_joint5=damping*jointV[4]+stiffness*(jointP[4]-joint_init[4])
        z_v_joint5=0.0
        if isInDoor:
            if isContact:# in door, contact
                print('indoor, contact')
                vx,vy=PI_inctrement_to_TCP_displacement_wholbody(linearVelocity, currentP, init_contact_P, hingeCenter, radius, 0.01)
                velocity_X=[vx, vy, 0.0, orientationVelocity[0],orientationVelocity[1],orientationVelocity[2]]
                z=[base_x_dot, base_rz_dot, 0.0, 0.0, 0.0, 0.0, z_v_joint5, 0.0]
                
                J_transpose=np.transpose(wholebody_J)
                Weignt_inverse=pinv(Weight)

                J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
                joint_v=np.matmul(J_sharp,velocity_X)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)
            else:# in door, not contact
                print('indoor, not contact')
                vx,vy=PI_inctrement_to_TCP_displacement_wholbody(linearVelocity, currentP, init_contact_P, hingeCenter, radius, 0.01)
                velocity_X=[vx, vy, 0.0, orientationVelocity[0],orientationVelocity[1],orientationVelocity[2]]
                z=[base_x_dot, base_rz_dot, 0.0, 0.0, 0.0, 0.0, z_v_joint5, 0.0]
                
                J_transpose=np.transpose(wholebody_J)
                Weignt_inverse=pinv(Weight)

                J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
                joint_v=np.matmul(J_sharp,velocity_X)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)
        else:
            print('not indoor')
            velocity_X=[linearVelocity, 0.0, 0.0, 0.0, 0.0, 0.0]
            z=[base_x_dot, base_rz_dot, 0.0, 0.0, 0.0, 0.0, z_v_joint5, 0.0]
            
            J_transpose=np.transpose(wholebody_J)
            Weignt_inverse=pinv(Weight)

            J_sharp=np.matmul(Weignt_inverse,np.matmul(J_transpose, pinv(np.matmul(wholebody_J,np.matmul(Weignt_inverse,J_transpose)))))
            joint_v=np.matmul(J_sharp,velocity_X)+np.matmul(np.eye(8,8)-np.matmul(J_sharp,wholebody_J),z)
        
        base_x=joint_v[0]
        base_y=0.0
        base_rz=joint_v[1]
        j1=joint_v[2]
        j2=joint_v[3]
        j3=joint_v[4]
        j4=joint_v[5]
        j5=joint_v[6]
        j6=joint_v[7]

        cmdVelocities.cmd_whole_body_velocity(base_x, base_y, base_rz, j1, j2, j3, j4, j5, j6)


        ## 用于下一轮
        frame0_T0=T_next
        frame0_R0=frame0_R1
        linearVelocity_previous=linearVelocity

        


        if iii1>=iii1_num-1:
            for iii in range(0,50):
                if abs(base_x_dot)>0.01:
                    base_x_to_stop=base_x_dot*(50.0-iii)/50.0
                    base_rz_dot_to_stop=base_rz_dot*(50.0-iii)/50.0
                    cmdVelocities.cmd_whole_body_velocity(base_x_to_stop, 0.0, base_rz_dot_to_stop, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    print(base_x_to_stop)
                    time.sleep(0.01)
                else:
                    cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            rtde_c.speedStop()
            print('stop!')


        t1=time.time()
        if t1-t0<0.01:
            #print('ii = '+str(iiprint('stop!')
            time.sleep(0.01-(t1-t0))
        else:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx  |',t1-t0)
    
    if stage2_phi>=90.0/180.0*pi:
        print('continue to the third stage-leaving the door.')
        time.sleep(0.5)
        stage3_base_motion_radius=0.7
        stage3_base_motion_center=[frameD_stage2_xb+stage3_base_motion_radius, frameD_stage2_yb]
        stage3_base_motion_length=pi*stage3_base_motion_radius/2.0


        desired_joint_position=joint_init
        current_joint_position=rtde_r.getActualQ()
        manipulator_joint_delta=list_element_minus(desired_joint_position, current_joint_position)
        movement_time_span=2*sqrt(stage3_base_motion_length/abs_acc)
        print('movement_time_span',movement_time_span)
        manipulator_joint_velocity=list_multiply(manipulator_joint_delta, 1.0/movement_time_span)

        x1_t0=0.0#这里的x1是弧长度
        x2_t0=0.0#圆周线速度
        x1_tf=-stage3_base_motion_length
        iii3_mum=ceil(movement_time_span*100)
        for iii3 in range(0,iii3_mum):
            t0=time.time()
            acc,string=time_optimal_for_1D_tracking(x1_tf, 0.0, x1_t0, x2_t0, abs_acc)
            current_angle=-(x1_t0-0.0)/stage3_base_motion_radius
            acc,string=time_optimal_for_1D_tracking(x1_tf, 0.0, x1_t0, x2_t0, abs_acc)
            x2_t0=x2_t0+acc*dt
            x1_t0=x1_t0+x2_t0*dt
            next_angle=-(x1_t0-0.0)/stage3_base_motion_radius
            base_x=x2_t0
            base_y=0.0
            #print('base_x',base_x)
            #print('current_angle',current_angle)
            #print('next_angle',next_angle)
            #print('(current_angle-next_angle)/dt',(current_angle-next_angle)/dt)
            base_rz=(current_angle-next_angle)/dt
            j1=manipulator_joint_velocity[0]
            j2=manipulator_joint_velocity[1]
            j3=manipulator_joint_velocity[2]
            j4=manipulator_joint_velocity[3]
            j5=manipulator_joint_velocity[4]
            j6=manipulator_joint_velocity[5]
            cmdVelocities.cmd_whole_body_velocity(base_x, base_y, base_rz, j1, j2, j3, j4, j5, j6)

            if iii3>=iii3_mum-1:
                for iii in range(0,50):
                    if abs(base_x_dot)>0.01:
                        base_x_to_stop=base_x_dot*(50.0-iii)/50.0
                        base_rz_dot_to_stop=base_rz_dot*(50.0-iii)/50.0
                        cmdVelocities.cmd_whole_body_velocity(base_x_to_stop, 0.0, base_rz_dot_to_stop, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        print(base_x_to_stop)
                        time.sleep(0.01)
                    else:
                        cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                cmdVelocities.cmd_whole_body_velocity(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                rtde_c.speedStop()
                print('stop!')
                break

            t1=time.time()
            if t1-t0<0.01:
                #print('ii = '+str(iiprint('stop!')
                time.sleep(0.01-(t1-t0))
            else:
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx  |',t1-t0)

        print('stage3 done.')




    


