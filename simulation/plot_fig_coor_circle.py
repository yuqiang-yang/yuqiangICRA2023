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
    fontdict={'family':'Times New Roman','size':14}

    drawcircle_car_coor1 = np.loadtxt('data/1drawcircle_car_coor.txt')
    drawcircle_ur_coor1 = np.loadtxt('data/1drawcircle_ur_coor.txt')
    drawcircle_wb1 = np.loadtxt('data/1drawcircle_wb.txt')
    drawcircle_car1 = np.loadtxt('data/1drawcircle_car.txt')
    drawcircle_car_ref1 = np.loadtxt('data/1drawcircle_car_ref.txt')
    drawcircle_wb_ref1 = np.loadtxt('data/1drawcircle_wb_ref.txt')

    drawcircle_car_coor2 = np.loadtxt('data/2drawcircle_car_coor.txt')
    drawcircle_ur_coor2 = np.loadtxt('data/2drawcircle_ur_coor.txt')
    drawcircle_wb2 = np.loadtxt('data/2drawcircle_wb.txt')
    drawcircle_car2 = np.loadtxt('data/2drawcircle_car.txt')
    drawcircle_car_ref2 = np.loadtxt('data/2drawcircle_car_ref.txt')
    drawcircle_wb_ref2 = np.loadtxt('data/2drawcircle_wb_ref.txt')

    drawcircle_wb3 = np.loadtxt('data/21drawsin_wb.txt')
    drawcircle_car3 = np.loadtxt('data/21drawsin_car.txt')

    drawcircle_wb1 = mean_filter(drawcircle_wb1,32,is_but=False)
    drawcircle_wb2 = mean_filter(drawcircle_wb2,32,is_but=False)
    drawcircle_wb3 = mean_filter(drawcircle_wb3,32,is_but=False)


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
    drawsin_wb1 = mean_filter(drawsin_wb1,32,is_but=True)
    drawsin_wb2 = mean_filter(drawsin_wb2,32,is_but=True)
    drawsin_wb3 = mean_filter(drawsin_wb3,32,is_but=True)


    plt.subplot(221)
    plt.plot(drawsin_wb_ref1[:,1],drawsin_wb_ref1[:,0],linewidth=2.5,alpha=0.9)
    plt.plot(drawsin_wb1[:,1],drawsin_wb1[:,0],linewidth=2.5,alpha=0.9)
    plt.plot(drawsin_wb2[:,1],drawsin_wb2[:,0],'--',linewidth=2.5,alpha=0.9)
    plt.plot(drawsin_wb3[:,1],drawsin_wb3[:,0],'--',linewidth=2.5,alpha=0.9)
    plt.legend(('reference','dis. not .','dis. coor.','without dis.'),fontsize=8,loc=3)

    #plt.legend(('reference','without coordination','with coordination'),fontsize=14,loc=1)
    #plt.xlabel('x/m',fontdict=fontdict)
    plt.ylabel('y/m',fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)  
    plt.title('(a)',position=(0.07,0.82))

    plt.subplot(222)
    plt.plot(drawcircle_wb_ref2[:,1],drawcircle_wb_ref2[:,0],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_wb1[:,1],drawcircle_wb1[:,0],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_wb2[:,1],drawcircle_wb2[:,0],'--',linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_wb3[:,1],drawcircle_wb3[:,0],'--',linewidth=2.5,alpha=0.9)
    plt.title('(b)',position=(0.07,0.82))
    #plt.legend(('reference','without coordination','with coordination'),fontsize=14,loc=1)
    #plt.xlabel('x/m',fontdict=fontdict)
    #plt.ylabel('y/m',fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks([],fontproperties='Times New Roman',size=14)  
    plt.subplot(413)
    ymax = np.max(drawcircle_car_coor1)+10
    ymin = np.min(drawcircle_car_coor1)

    tmax1 = drawcircle_car_coor1.shape[0]
    t = np.arange(0.0,tmax1/10,0.1)
    plt.plot(t,drawcircle_car_coor1,linewidth=2.5,alpha=0.9)
    plt.plot(t,drawcircle_ur_coor1,'--',linewidth=2.5,alpha=0.9) 

    tmax2 = drawcircle_car_coor2.shape[0]
    t = np.arange(0.0,tmax2/10,0.1)
    plt.plot(t,drawcircle_car_coor2,linewidth=2.5,alpha=0.9)
    plt.plot(t,drawcircle_ur_coor2,'--',linewidth=2.5,alpha=0.9) 
    
    xmax= tmax2/10+1
    plt.xlim((0,tmax2/10+1))
    plt.ylim((ymin,ymax))
    x = [0,5.58,9.06,13.76,15.88,20.53,24.09,27.72,29.64,xmax]
    color = ['pink','g','pink','b','pink','violet','pink','cyan','pink']
    for i in range(len(x)-1):
        plt.fill([x[i],x[i],x[i+1],x[i+1]],[0,ymax,ymax,0],'-.',color=color[i],alpha=0.2,linewidth=0)
    plt.legend(('mobile base','manipulator'),fontsize=10,loc=2)
    plt.ylabel('index',fontdict=fontdict)
    plt.xticks([],fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)  
    
    plt.legend(('base not.','arm not.','base coor.','arm coor.'),fontsize=8,loc=4)

    #plt.xlabel('time/s',fontdict=fontdict)
    plt.title('(c)',position=(0.03,0.68))
    plt.subplot(414)

    ymax = np.max(drawsin_car_coor1)+10
    ymin = np.min(drawsin_car_coor1)
    tmax1 = drawsin_car_coor1.shape[0]
    t = np.arange(0.0,tmax1/10,0.1)
    plt.plot(t,drawsin_car_coor1,linewidth=2.5,alpha=0.9)
    plt.plot(t,drawsin_ur_coor1,'--',linewidth=2.5,alpha=0.9)


    ymax = np.max(drawsin_car_coor1)+10
    ymin = np.min(drawsin_car_coor1)

    tmax2 = drawsin_car_coor2.shape[0]
    t = np.arange(0.0,tmax2/10,0.1)
    plt.plot(t,drawsin_car_coor2,linewidth=2.5,alpha=0.9)
    plt.plot(t,drawsin_ur_coor2,'--',linewidth=2.5,alpha=0.9) 
    
    xmax= tmax2/10+1
    plt.xlim((0,tmax2/10+1))
    plt.ylim((ymin,ymax))
    x = [0,4.91,9.19,11.43,14.6,18.16,20.47,24.99,29.46,xmax]
    color = ['pink','cyan','pink','g','pink','b','pink','violet','pink']
    for i in range(len(x)-1):
        plt.fill([x[i],x[i],x[i+1],x[i+1]],[0,ymax,ymax,0],'-.',color=color[i],alpha=0.2,linewidth=0)
    plt.ylabel('index',fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)  
    plt.xlabel('time/s',fontdict=fontdict)
    plt.title('(d)',position=(0.03,0.68))

    # for i in range(len(x)-1):
    #     plt.fill([x[i],x[i],x[i+1],x[i+1]],[0,ymax,ymax,0],'-.',color=color[i],alpha=0.2,linewidth=0)
    # plt.plot(t,drawsin_car_coor1,linewidth=2.5)
    # plt.plot(t,drawsin_ur_coor1,linewidth=2.5)
    # plt.xlim((0,tmax2/10+1))
    # plt.ylim((ymin,ymax))
    # plt.legend(('mobile base','manipulator'),fontsize=14,loc=2)
    # plt.ylabel('index',fontdict=fontdict)
    # plt.xticks(fontproperties='Times New Roman',size=20)
    # plt.yticks(fontproperties='Times New Roman',size=20) 
    plt.savefig('tt2.png',dpi=400) 
    plt.show()