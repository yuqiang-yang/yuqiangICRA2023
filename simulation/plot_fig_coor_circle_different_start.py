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

    drawcircle_car_coor1 = np.loadtxt('data/6drawsin_car_coor.txt')
    drawcircle_ur_coor1 = np.loadtxt('data/6drawsin_ur_coor.txt')
    drawcircle_wb1 = np.loadtxt('data/6drawsin_wb.txt')
    drawcircle_car1 = np.loadtxt('data/6drawsin_car.txt')
    drawcircle_car_ref1 = np.loadtxt('data/6drawsin_car_ref.txt')
    drawcircle_wb_ref1 = np.loadtxt('data/6drawsin_wb_ref.txt')

    drawcircle_car_coor2 = np.loadtxt('data/5drawsin_car_coor.txt')
    drawcircle_ur_coor2 = np.loadtxt('data/5drawsin_ur_coor.txt')
    drawcircle_wb2 = np.loadtxt('data/5drawsin_wb.txt')
    drawcircle_car2 = np.loadtxt('data/5drawsin_car.txt')
    drawcircle_car_ref2 = np.loadtxt('data/5drawsin_car_ref.txt')
    drawcircle_wb_ref2 = np.loadtxt('data/5drawsin_wb_ref.txt')

    drawcircle_car_coor3 = np.loadtxt('data/4drawsin_car_coor.txt')
    drawcircle_ur_coor3 = np.loadtxt('data/4drawsin_ur_coor.txt')
    drawcircle_wb3 = np.loadtxt('data/4drawsin_wb.txt')
    drawcircle_car3 = np.loadtxt('data/4drawsin_car.txt')
    drawcircle_car_ref3 = np.loadtxt('data/4drawsin_car_ref.txt')
    drawcircle_wb_ref3 = np.loadtxt('data/4drawsin_wb_ref.txt')



    drawcircle_wb10 = np.loadtxt('data/10drawsin_wb.txt')
    drawcircle_car10 = np.loadtxt('data/10drawsin_car.txt')
    drawcircle_wb_ref10 = np.loadtxt('data/10drawsin_wb_ref.txt')
    drawcircle_car_ref10 = np.loadtxt('data/10drawsin_car_ref.txt')

   
    drawcircle_wb11 = np.loadtxt('data/11drawsin_wb.txt')
    drawcircle_car11 = np.loadtxt('data/11drawsin_car.txt')
    drawcircle_wb_ref11 = np.loadtxt('data/11drawsin_wb_ref.txt')

    drawcircle_wb12 = np.loadtxt('data/12drawsin_wb.txt')
    drawcircle_car12 = np.loadtxt('data/12drawsin_car.txt')
    drawcircle_wb_ref12 = np.loadtxt('data/12drawsin_wb_ref.txt')
    # drawcircle_car_coor4 = np.loadtxt('data/3drawsin_car_coor.txt')
    # drawcircle_ur_coor4 = np.loadtxt('data/3drawsin_ur_coor.txt')
    # drawcircle_wb4 = np.loadtxt('data/3drawsin_wb.txt')
    # drawcircle_car4 = np.loadtxt('data/3drawsin_car.txt')
    # drawcircle_car_ref4 = np.loadtxt('data/3drawsin_car_ref.txt')
    # drawcircle_wb_ref4 = np.loadtxt('data/3drawsin_wb_ref.txt')
    # drawcircle_car_coor2 = np.loadtxt('data/2drawcircle_car_coor.txt')
    # drawcircle_ur_coor2 = np.loadtxt('data/2drawcircle_ur_coor.txt')
    # drawcircle_wb2 = np.loadtxt('data/2drawcircle_wb.txt')
    # drawcircle_car2 = np.loadtxt('data/2drawcircle_car.txt')
    # drawcircle_car_ref2 = np.loadtxt('data/2drawcircle_car_ref.txt')
    # drawcircle_wb_ref2 = np.loadtxt('data/2drawcircle_wb_ref.txt')
    print(drawcircle_wb1[-1])

    drawcircle_wb1 = mean_filter(drawcircle_wb1,32,is_but=False)
    drawcircle_wb2 = mean_filter(drawcircle_wb2,32,is_but=False)
    drawcircle_wb3 = mean_filter(drawcircle_wb3,32,is_but=False)

    drawcircle_wb10 = mean_filter(drawcircle_wb10,32,is_but=False)
    drawcircle_wb11 = mean_filter(drawcircle_wb11,32,is_but=False)
    drawcircle_wb12 = mean_filter(drawcircle_wb12,32,is_but=False)
    print(drawcircle_wb1[-1])


    plt.subplot(221)
    plt.plot(drawcircle_car_ref1[:,0],drawcircle_car_ref1[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_car1[:,0],drawcircle_car1[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_car3[:,0],drawcircle_car3[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_car2[:,0],drawcircle_car2[:,1],linewidth=2.5,alpha=0.9)
    
    # plt.plot(drawcircle_car2[:,0],drawcircle_car2[:,1],linewidth=2.5,alpha=0.9)
    fontdict={'family':'Times New Roman','size':14}
    #plt.ylabel('y/m',fontdict=fontdict)
    plt.ylabel('y/m',fontdict=fontdict)

    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)  
    p.subplots_adjust(hspace=0.22,wspace=0.2)
    plt.legend(('reference','start1','start2','start3',),fontsize=8,loc='center left')
    plt.title('(a)',position=(0.07,0.82))

    plt.subplot(222)
    plt.plot(drawcircle_wb_ref1[:,0],drawcircle_wb_ref1[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_wb1[:,0],drawcircle_wb1[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_wb3[:,0],drawcircle_wb3[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_wb2[:,0],drawcircle_wb2[:,1],linewidth=2.5,alpha=0.9)



    # plt.plot(drawcircle_wb2[:,1],drawcircle_wb2[:,0],linewidth=2.5,alpha=0.9)
    #plt.legend(('reference','without coordination','with coordination'),fontsize=14,loc=1)
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)  
    plt.title('(b)',position=(0.07,0.82))


    plt.subplot(223)
    plt.plot(drawcircle_car_ref10[:,0],drawcircle_car_ref10[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_car10[:,0],drawcircle_car10[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_car11[:,0],drawcircle_car11[:,1],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_car12[:,0],drawcircle_car12[:,1],linewidth=2.5,alpha=0.9)
    # plt.plot(drawcircle_car4[:,0],drawcircle_car4[:,1],linewidth=2.5,alpha=0.9)

    # plt.plot(drawcircle_car2[:,0],drawcircle_car2[:,1],linewidth=2.5,alpha=0.9)
    fontdict={'family':'Times New Roman','size':14}
    #plt.ylabel('y/m',fontdict=fontdict)
    plt.xlabel('x/m',fontdict=fontdict)
    plt.ylabel('y/m',fontdict=fontdict)

    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)  
    p.subplots_adjust(hspace=0.22,wspace=0.2)
    #plt.legend(('reference','start1','start2','start3'),fontsize=10,loc=2)

    plt.title('(c)',position=(0.07,0.82))

    plt.subplot(224)
    plt.plot(drawcircle_wb_ref10[:,1],drawcircle_wb_ref10[:,0],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_wb10[:,1],drawcircle_wb10[:,0],linewidth=2.5,alpha=0.9)
    plt.plot(drawcircle_wb11[:,1],drawcircle_wb11[:,0],linewidth=2.5,alpha=0.9)

    plt.plot(drawcircle_wb12[:,1],drawcircle_wb12[:,0],linewidth=2.5,alpha=0.9)

    # plt.plot(drawcircle_wb2[:,1],drawcircle_wb2[:,0],linewidth=2.5,alpha=0.9)
    #plt.legend(('reference','without coordination','with coordination'),fontsize=14,loc=1)
    plt.xlabel('x/m',fontdict=fontdict)
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)  
    plt.title('(d)',position=(0.07,0.82))

    # plt.subplot(414)
    # ymax = np.max(drawcircle_car_coor1)+10
    # ymin = np.min(drawcircle_car_coor1)

    # tmax2 = drawcircle_car_coor2.shape[0]
    # t = np.arange(0.0,tmax2/10,0.1)
    # plt.plot(t,drawcircle_car_coor2,linewidth=2.5,alpha=0.9)
    # plt.plot(t,drawcircle_ur_coor2,linewidth=2.5,alpha=0.9) 
    # xmax= tmax2/10+1
    # plt.xlim((0,tmax2/10+1))
    # plt.ylim((ymin,ymax))
    # x = [0,5.58,9.06,13.76,15.88,14.53,24.09,27.72,29.64,xmax]
    # color = ['pink','g','pink','b','pink','violet','pink','cyan','pink']
    # for i in range(len(x)-1):
    #     plt.fill([x[i],x[i],x[i+1],x[i+1]],[0,ymax,ymax,0],'-.',color=color[i],alpha=0.2,linewidth=0)
    # plt.legend(('mobile base','manipulator'),fontsize=14,loc=2)
    # plt.ylabel('index',fontdict=fontdict)
    # plt.xticks(fontproperties='Times New Roman',size=14)
    # plt.yticks(fontproperties='Times New Roman',size=14)  
    # plt.xlabel('time/s',fontdict=fontdict)

    # plt.subplot(413)

    # ymax = np.max(drawcircle_car_coor1)+10
    # ymin = np.min(drawcircle_car_coor1)
    # tmax1 = drawcircle_car_coor1.shape[0]
    # t = np.arange(0.0,tmax1/10,0.1)


    # for i in range(len(x)-1):
    #     plt.fill([x[i],x[i],x[i+1],x[i+1]],[0,ymax,ymax,0],'-.',color=color[i],alpha=0.2,linewidth=0)
    # plt.plot(t,drawcircle_car_coor1,linewidth=2.5,alpha=0.9)
    # plt.plot(t,drawcircle_ur_coor1,linewidth=2.5,alpha=0.9)
    # plt.xlim((0,tmax2/10+1))
    # plt.ylim((ymin,ymax))
    # plt.legend(('mobile base','manipulator'),fontsize=14,loc=2)
    # plt.ylabel('index',fontdict=fontdict)
    # plt.xticks(fontproperties='Times New Roman',size=14)
    # plt.yticks(fontproperties='Times New Roman',size=14) 
    
    plt.savefig('tt3.png',dpi=600) 
    plt.show()