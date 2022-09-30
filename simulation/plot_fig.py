#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from trajectory_msgs.msg import *
from control_msgs.msg import *
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from geometry_msgs.msg import Twist
from control_msgs.msg import JointTrajectoryControllerState
from nav_msgs.msg import Odometry
import numpy as np
import rospy
import time
from math import cos, acos, sin, asin, sqrt, exp, atan, atan2, pi, tan, ceil
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    p=plt.figure(figsize=(12,3))
    plt.subplot(142)
    for i in range(13):
        if i == 11:
            continue
        data=np.loadtxt('data/2multistart'+str(i+1)+'.txt')
        plt.scatter(data[0,0],data[0,1],marker='*',linewidths=3,color='#17becf')
        plt.plot(data[:,0],data[:,1],'--',linewidth=2,color='#1f77b4')
        #plt.scatter(data[-1,0],data[-1,1],marker='o',linewidths=2,color='red')
    ref = np.loadtxt('data/car_xytheta_dot_record1.txt')
    plt.plot(ref[:-20,0],ref[:-20,1],linewidth=3,color='red',alpha=0.7)
    fontdict={'family':'Times New Roman','size':14}
    plt.xlabel('x/m',fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks([],fontproperties='Times New Roman',size=14)  
    plt.title('(b)',position=(0.07,0.8))
    plt.subplot(144)
    for i in range(13):
        if i == 11:
            continue
        data=np.loadtxt('data/3multistart_mul'+str(i+1)+'.txt')
        plt.scatter(data[0,0],data[0,1],marker='*',linewidths=3,color='#17becf')
            
        if i > 4 and i!=10:
            plt.plot(data[:-100,0],data[:-100,1],'--',linewidth=2,color='#1f77b4')

        else:
            plt.plot(data[:,0],data[:,1],'--',linewidth=2,color='#1f77b4')
        #plt.scatter(data[-1,0],data[-1,1],marker='o',linewidths=2,color='red')
    ref = np.loadtxt('data/car_xytheta_dot_record4.txt')
    plt.plot(ref[:-20,0],ref[:-20,1],linewidth=3,color='red',alpha=0.7)
    ref = np.loadtxt('data/car_xytheta_dot_record5.txt')
    plt.plot(ref[:-5,0],ref[:-5,1],linewidth=3,color='red',alpha=0.7)
    fontdict={'family':'Times New Roman','size':14}
    plt.xlabel('x/m',fontdict=fontdict)
    plt.title('(d)',position=(0.07,0.8))
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks([],fontproperties='Times New Roman',size=14)  
    plt.subplot(141)
    ref = np.loadtxt('data/car_xytheta_dot_record1.txt')
    plt.plot(ref[:-20,0],ref[:-20,1],linewidth=3,color='red',alpha=0.7,label='demon.')
    for i in range(12):
        if i == 12 or i==3 or i ==7:
            data=np.loadtxt('data/5multistart_jj'+str(i+1)+'.txt')
            if i == 999 :
                plt.scatter(data[0,0],data[0,1],marker='*',linewidths=3,color='green',label='not conv.')
            else:
                plt.scatter(data[0,0],data[0,1],marker='*',linewidths=3,color='green')
            continue
        if i==11:
            continue
        data=np.loadtxt('data/5multistart_jj'+str(i+1)+'.txt')
        if i == 1:
            plt.scatter(data[0,0],data[0,1],marker='*',linewidths=3,color='#17becf',label='starting')
            plt.plot(data[:,0],data[:,1],'--',linewidth=2,color='#1f77b4',label='stable')
        else:
            plt.scatter(data[0,0],data[0,1],marker='*',linewidths=3,color='#17becf')
            plt.plot(data[:,0],data[:,1],'--',linewidth=2,color='#1f77b4')            
        #plt.scatter(data[-1,0],data[-1,1],marker='o',linewidths=2,color='red')
    plt.scatter(0,3,marker='*',linewidths=3,color='green')
    plt.legend(loc='center left',fontsize=8)
    #plt.legend(('demonstration','start point','stabilization'),loc=2,fontsize=8)
    plt. ('(a)',position=(0.07,0.8))

    fontdict={'family':'Times New Roman','size':14}
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)  
    plt.xlabel('x/m',fontdict=fontdict)
    plt.ylabel('y/m',fontdict=fontdict)

    plt.subplot(143)
    ref = np.loadtxt('data/car_xytheta_dot_record4.txt')
    plt.plot(ref[:-20,0],ref[:-20,1],linewidth=3,color='red',alpha=0.7)
    for i in range(12):
        if i == 12 or i==3 or i==7:
            data=np.loadtxt('data/5multistart_mul_jj'+str(i+1)+'.txt')
            plt.scatter(data[0,0],data[0,1],marker='*',linewidths=3,color='green')
            continue
        if i == 11:
            continue
        data=np.loadtxt('data/5multistart_mul_jj'+str(i+1)+'.txt')
        plt.scatter(data[0,0],data[0,1],marker='*',linewidths=3,color='#17becf')

        plt.plot(data[:,0],data[:,1],'--',linewidth=2,color='#1f77b4')
        #plt.scatter(data[-1,0],data[-1,1],marker='o',linewidths=2,color='red')
    plt.title('(c)',position=(0.07,0.8))
    plt.scatter(0,3,marker='*',linewidths=3,color='green')
    ref = np.loadtxt('data/car_xytheta_dot_record5.txt')
    plt.plot(ref[:-5,0],ref[:-5,1],linewidth=3,color='red',alpha=0.7)
    fontdict={'family':'Times New Roman','size':14}
    #plt.ylabel('y/m',fontdict=fontdict)
    plt.xlabel('x/m',fontdict=fontdict)

    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks([],fontproperties='Times New Roman',size=14)  
    p.subplots_adjust(left=0.14,bottom=0.25,wspace=0.02)
    #plt.legend(('demonstration','start point','stabilization'),loc=2,fontsize=8)
    plt.savefig('IMAGE/tt8.png',bbox_inches='tight',dpi=400)
    plt.show()