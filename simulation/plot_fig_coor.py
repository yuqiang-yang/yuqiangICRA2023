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
from scipy import signal

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

if __name__ == "__main__":
    p=plt.figure()

    drawsin_car_coor1 = np.loadtxt('data/2adrawsin_car_coor.txt')
    drawsin_ur_coor1 = np.loadtxt('data/2adrawsin_ur_coor.txt')
    drawsin_wb1 = np.loadtxt('data/2adrawsin_wb.txt')
    drawsin_car1 = np.loadtxt('data/2adrawsin_car.txt')
    drawsin_car_ref1 = np.loadtxt('data/2adrawsin_car_ref.txt')
    drawsin_wb_ref1 = np.loadtxt('data/2adrawsin_wb_ref.txt')

    drawsin_car_coor2 = np.loadtxt('data/1adrawsin_car_coor.txt')
    drawsin_ur_coor2 = np.loadtxt('data/1adrawsin_ur_coor.txt')
    drawsin_wb2 = np.loadtxt('data/1adrawsin_wb.txt')
    drawsin_car2 = np.loadtxt('data/1adrawsin_car.txt')
    drawsin_car_ref2 = np.loadtxt('data/1adrawsin_car_ref.txt')
    drawsin_wb_ref2 = np.loadtxt('data/1adrawsin_wb_ref.txt')
    drawsin_wb3 = np.loadtxt('data/20drawsin_wb.txt')
    drawsin_car3 = np.loadtxt('data/20drawsin_car.txt')
    drawsin_wb1 = mean_filter(drawsin_wb1,8,is_but=True)
    drawsin_wb2 = mean_filter(drawsin_wb2,8,is_but=True)
    drawsin_wb3 = mean_filter(drawsin_wb3,8,is_but=True)



    plt.subplot(221)
    plt.plot(drawsin_car_ref1[:,0],drawsin_car_ref1[:,1],linewidth=2)
    plt.plot(drawsin_car1[:,0],drawsin_car1[:,1],linewidth=2)
    plt.plot(drawsin_car2[:,0],drawsin_car2[:,1],linewidth=2)
    plt.plot(drawsin_car3[:,0],drawsin_car3[:,1],linewidth=2)

    fontdict={'family':'Times New Roman','size':20}
    #plt.ylabel('y/m',fontdict=fontdict)
    plt.xlabel('x/m',fontdict=fontdict)
    plt.ylabel('y/m',fontdict=fontdict)

    plt.xticks(fontproperties='Times New Roman',size=20)
    plt.yticks(fontproperties='Times New Roman',size=20)  
    p.subplots_adjust(hspace=0.22,wspace=0.2)
    plt.legend(('reference','without coordination','with coordination','without disturbance'),fontsize=14,loc=2)


    plt.subplot(222)
    plt.plot(drawsin_wb_ref2[:,1],drawsin_wb_ref2[:,0],linewidth=2)
    plt.plot(drawsin_wb1[:,1],drawsin_wb1[:,0],linewidth=2)
    plt.plot(drawsin_wb2[:,1],drawsin_wb2[:,0],linewidth=2)
    plt.plot(drawsin_wb3[:,1],drawsin_wb3[:,0],linewidth=2)

    plt.legend(('reference','without coordination','with coordination','without disturbance'),fontsize=14,loc=1)
    plt.xlabel('x/m',fontdict=fontdict)
    plt.ylabel('y/m',fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman',size=20)
    plt.yticks(fontproperties='Times New Roman',size=20)  
    plt.subplot(414)
    ymax = np.max(drawsin_car_coor1)+10
    ymin = np.min(drawsin_car_coor1)

    tmax2 = drawsin_car_coor2.shape[0]
    t = np.arange(0.0,tmax2/10,0.1)
    plt.plot(t,drawsin_car_coor2,linewidth=3)
    plt.plot(t,drawsin_ur_coor2,linewidth=3) 
    xmax= tmax2/10+1
    plt.xlim((0,tmax2/10+1))
    plt.ylim((ymin,ymax))
    x = [0,4.91,9.19,11.43,14.6,18.16,20.47,24.99,29.46,xmax]
    color = ['pink','cyan','pink','g','pink','b','pink','violet','pink']
    for i in range(len(x)-1):
        plt.fill([x[i],x[i],x[i+1],x[i+1]],[0,ymax,ymax,0],'-.',color=color[i],alpha=0.2,linewidth=0)
    plt.legend(('mobile base','manipulator'),fontsize=14,loc=2)
    plt.ylabel('index',fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman',size=20)
    plt.yticks(fontproperties='Times New Roman',size=20)  
    plt.xlabel('time/s',fontdict=fontdict)

    plt.subplot(413)

    ymax = np.max(drawsin_car_coor1)+10
    ymin = np.min(drawsin_car_coor1)
    tmax1 = drawsin_car_coor1.shape[0]
    t = np.arange(0.0,tmax1/10,0.1)


    for i in range(len(x)-1):
        plt.fill([x[i],x[i],x[i+1],x[i+1]],[0,ymax,ymax,0],'-.',color=color[i],alpha=0.2,linewidth=0)
    plt.plot(t,drawsin_car_coor1,linewidth=3)
    plt.plot(t,drawsin_ur_coor1,linewidth=3)
    plt.xlim((0,tmax2/10+1))
    plt.ylim((ymin,ymax))
    plt.legend(('mobile base','manipulator'),fontsize=14,loc=2)
    plt.ylabel('index',fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman',size=20)
    plt.yticks(fontproperties='Times New Roman',size=20)  
    plt.show()