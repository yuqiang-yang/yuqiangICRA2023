# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('.')
import rospy
import time
import numpy as np
import  threading                                                            
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
from scipy import signal
import matplotlib
#read UR and car demonstration data
def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)
light_jet = cmap_map(lambda x: x/2 +0.5, matplotlib.cm.coolwarm)
def readTxt():#16勉強還行
    data = np.loadtxt('data/car_cycle_zero32.txt')
    xy_path = data[:,0:2]
    xy_dot_path = np.concatenate((data[:,3].reshape(-1,1)*np.cos(data[:,2].reshape(-1,1)),data[:,3].reshape(-1,1)*np.sin(data[:,2]).reshape(-1,1)),axis=1)
    xy_theta_path = data[:,0:3]
    xy_theta_dot_path = np.concatenate((xy_dot_path,data[:,5].reshape(-1,1)),axis=1)

    ur_end_record = np.loadtxt('data/ur_circle_zero32.txt')
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


#Gaussian Process Regression implementation
class MyGPR():
    def __init__(self,data_in,data_out):
        self.data_in = data_in
        self.data_out = data_out
        #kernel1 = 1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+1*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
        kernel1 = 1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+0.1*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
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
        self.std_matrix = std_matrix
        for i in range(step-1):
            for j in range(step-1):
                self.diff_std[i,j,0] = (std_matrix[i,j+1] - std_matrix[i,j])
                self.diff_std[i,j,1] = (std_matrix[i+1,j] - std_matrix[i,j])
        self.diff_std[:,:,0] /= x_step
        self.diff_std[:,:,1] /= y_step
        self.long_dis_stable = (np.max(abs(self.diff_std)))/4

        return self.diff_std
    def get_next_step_jens(self,predict_point):
        result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)

        return np.array((result[0,0] - (self.diff_std[remap_i,remap_j,0])*2 ,  result[0,1] - (self.diff_std[remap_i,remap_j,1])*2)).reshape(1,-1)
 
    def get_next_step(self,predict_point):
        result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)
        distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        min_index = np.argmin(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        long_dis_stable_direction = -(self.demonstrate_point[min_index] - predict_point)/np.linalg.norm(self.demonstrate_point[min_index] - predict_point)

        if distance > 2:
            distance = 5.4
        else:
            distance *= 2.7
        self.dis_cof = 0.07/np.max(self.diff_std)
        distance *= self.dis_cof
        return np.array((result[0,0] - (self.diff_std[remap_i,remap_j,0]+long_dis_stable_direction[0,0]*self.long_dis_stable)*distance ,  result[0,1] - (self.diff_std[remap_i,remap_j,1]+long_dis_stable_direction[0,1]*self.long_dis_stable)*distance)).reshape(1,-1)
    def get_varian(self,predict_point):
        #result,_ = self.gpr1.predict(predict_point,return_std=True)
        remap_j = int((predict_point[0,0]-self.xmin)/(self.xmax-self.xmin)*self.step)
        remap_i = int((predict_point[0,1]-self.ymin)/(self.ymax-self.ymin)*self.step)
        distance = np.min(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        min_index = np.argmin(np.sqrt((self.demonstrate_point[:,0]-predict_point[0,0])**2+(self.demonstrate_point[:,1]-predict_point[0,1])**2))
        long_dis_stable_direction = -(self.demonstrate_point[min_index] - predict_point)/np.linalg.norm(self.demonstrate_point[min_index] - predict_point)

        if distance > 2:
            distance = 5.4
        else:
            distance *= 2.7
        self.dis_cof = 0.07/np.max(self.diff_std)
        distance *= self.dis_cof
        return np.array((- (self.diff_std[remap_i,remap_j,0]+long_dis_stable_direction[0,0]*self.long_dis_stable)*distance ,   - (self.diff_std[remap_i,remap_j,1]+long_dis_stable_direction[0,1]*self.long_dis_stable)*distance)).reshape(1,-1)


    def plot_field(self,xmin,xmax,ymin,ymax,demonstrate_point,step=75,field_density = 7,carpick_real=None):
        diff_std = self.get_diff_varian_matrix(xmin,xmax,ymin,ymax,demonstrate_point,step=step)
        
        x_step = (xmax - xmin)/step
        y_step = (ymax - ymin)/step
        xx = np.arange(xmin,xmax,x_step)
        yy = np.arange(ymin,ymax,y_step)
        xxx1,yyy1 = np.meshgrid(xx,yy)
        predict_point1 = np.concatenate((xxx1.reshape(-1,1),yyy1.reshape(-1,1)),1)        

        varian_res = np.zeros((step*step))
        for i in range(predict_point1.shape[0]):
            varian_res[i]=np.linalg.norm((self.get_varian(predict_point1[i].reshape(1,-1))).ravel())
            print(i)
        varian_res = varian_res.reshape(step,step)
        vmax = np.max(varian_res)/1.1
        vmin = np.min(varian_res)/10

        #p=plt.figure()
        matplotlib.colors.Normalize()


        # plt.xlim((xmin,xmax))
        # plt.ylim((ymin,ymax))

        #sns.heatmap(df,cmap="Blues",cbar=False,xticklabels=False,yticklabels=False,linewidths=1).invert_yaxis()

        x_step = (xmax - xmin)/field_density
        y_step = (ymax - ymin)/field_density
        xx = np.arange(xmin+x_step/2,xmax+x_step/2,x_step)
        yy = np.arange(ymin+y_step/2,ymax+y_step/2,y_step)
        xxx,yyy = np.meshgrid(xx,yy)
        predict_point = np.concatenate((xxx.reshape(-1,1),yyy.reshape(-1,1)),1)
        result,std = self.gpr1.predict(predict_point,return_std=True)
        
        p1=plt.figure(figsize=(6,4))

        pc = plt.pcolormesh(xxx1,yyy1,varian_res,cmap=light_jet,alpha=1,shading='gouraud')
        plt.plot(demonstrate_point[:,0],demonstrate_point[:,1],linewidth=4,color='yellow',alpha=0.5)
        plt.plot(carpick_real[:,0],carpick_real[:,1],linewidth=2,color='b',alpha=0.5)
        plt.legend(('demonstration','reproduction'),loc=1,fontsize=12)

        for i in range(field_density**2): 
            remap_i = int(i//field_density) * int(step//field_density)
            remap_j = int(i%field_density) * int(step//field_density)
            res = self.get_next_step(predict_point[i,:].reshape(1,-1))
            res*=2
            plt.arrow(predict_point[i,0],predict_point[i,1], res[0,0], res[0,1],head_width=0.05,color='purple')        


        fontdict={'family':'Times New Roman','size':14}
        #plt.xlabel('x/m',fontdict=fontdict,labelpad=0)
        plt.ylabel('y/m',fontdict=fontdict,labelpad=0)
        plt.xlabel('x/m',fontdict=fontdict,labelpad=0)

        xx = [-1,0,1,2,3]
        yy = [-1,0,1,2,3]

        plt.xticks(xx,fontproperties='Times New Roman',size=14)
        plt.yticks(yy,fontproperties='Times New Roman',size=14)        
        pc=plt.pcolormesh(xxx1,yyy1,self.std_matrix,cmap=light_jet,alpha=1,shading='gouraud')
        p1.subplots_adjust(left=0.15,bottom=0.25,wspace=0.02)
        cax=p1.add_axes([0.18,0.1,0.7,0.03])


        c=p1.colorbar(pc,cax=cax,orientation='horizontal')
        c.ax.tick_params(labelsize=14)
        c.ax.set_yticklabels(['0','0.1'],family='Time New Roman')
    def find_nearest_point(self,point):
        distance = np.linalg.norm((self.data_out-point),axis=1)
        print('arm',np.argmin(distance))
        return self.data_in[np.argmin(distance)]

if __name__ == "__main__":
    print('start the main function')
    #read demonstration data, filter and downsample
    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path,arm_xyztheta_path = readTxt()

    down_times = int(xy_path.shape[0]//40 + 1)
    print('down_times',down_times)
    xy_dot_path_filtered = mean_filter(xy_dot_path,is_but=True)
    xy_dot_path_filtered_down = down_sample(xy_dot_path_filtered,down_times)
    xy_path_down = down_sample(xy_path,down_times)
 
    down_times2 = int(xy_path.shape[0]//250 + 1)
    xy_path_down_2 = down_sample(xy_path,down_times2)
    arm_xyztheta_path_down = down_sample(arm_xyztheta_path,down_times2)
    pickend_real = np.loadtxt('data/1pickend_real.txt')
    pickend_ref = np.loadtxt('data/1pickend_ref.txt')
    carpick_real = np.loadtxt('data/1carpick_real.txt')
    carpick_ref = np.loadtxt('data/1carpick_ref.txt')


    pickend_real1 = np.loadtxt('data/6pickend_real.txt')
    pickend_ref1 = np.loadtxt('data/6pickend_ref.txt')
    carpick_real1 = np.loadtxt('data/6carpick_real.txt')
    carpick_ref1 = np.loadtxt('data/6carpick_ref.txt')

    pickend_real2 = np.loadtxt('data/7pickend_real.txt')
    pickend_ref2 = np.loadtxt('data/7pickend_ref.txt')
    carpick_real2 = np.loadtxt('data/7carpick_real.txt')
    carpick_ref2 = np.loadtxt('data/7carpick_ref.txt')
    #GPR fit
    car_gpr = MyGPR(xy_path_down,xy_dot_path_filtered_down)
    car2arm_gpr = MyGPR(xy_path_down_2,arm_xyztheta_path_down)
    car_gpr.plot_field(-1,3,-1,3,xy_path,carpick_real=carpick_real)
    result1 = car2arm_gpr.gpr1.predict((xy_path),return_std=False)
    plt.savefig('IMAGE/tt3.png',bbox_inches='tight',dpi=200)


    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111,projection='3d')
    #ax1.set_aspect(aspect=2.5/4)
    #ax1.set_aspect('equal')
    carpick_real=carpick_real[:-8000]
    pickend_real=pickend_real[:-8000]
    ax1.plot3D(arm_xyztheta_path[:,0]+xy_path[:,1],arm_xyztheta_path[:,1]-xy_path[:,0],arm_xyztheta_path[:,2],linewidth=2,alpha=0.9)
    ax1.plot3D(pickend_real[:,0]+carpick_real[:,1],pickend_real[:,1]-carpick_real[:,0],pickend_real[:,2],linewidth=2,alpha=0.9)
    ax1.plot3D(pickend_real2[:,0]+carpick_real2[:,1],pickend_real2[:,1]-carpick_real2[:,0],pickend_real2[:,2],linewidth=2,alpha=0.9)
    pickend_real1=down_sample(pickend_real1,30)
    carpick_real1=down_sample(carpick_real1,30)
    start_index = int(870/30)
    end_index = int(1800/30)
    ax1.scatter(pickend_real1[start_index:end_index,0]+carpick_real1[start_index:end_index,1]-0.02,pickend_real1[start_index:end_index,1]-carpick_real1[start_index:end_index,0]+0.01,pickend_real1[start_index:end_index,2],linewidth=1,color='b',alpha=0.5)
    ax1.plot3D(arm_xyztheta_path[:,0]+xy_path[:,1],arm_xyztheta_path[:,1]-xy_path[:,0],-0.1,'--',linewidth=2,alpha=0.9,color='#1f77b4')
    ax1.plot3D(pickend_real[:,0]+carpick_real[:,1],pickend_real[:,1]-carpick_real[:,0],-0.1,'--',linewidth=2,alpha=0.9,color='#ff7f0e')
    ax1.plot3D(pickend_real2[:,0]+carpick_real2[:,1],pickend_real2[:,1]-carpick_real2[:,0],-0.1,'--',linewidth=2,alpha=0.9,color='#2ca02c')

    ax1.scatter(pickend_real1[start_index:end_index,0]+carpick_real1[start_index:end_index,1]-0.02,pickend_real1[start_index:end_index,1]-carpick_real1[start_index:end_index,0]+0.01,-0.1,linewidth=1,color='b',alpha=0.5)

    down=400
    for i in range(int(pickend_real.shape[0]/down)):
        ax1.quiver(pickend_real[i*down,0]+carpick_real[i*down,1],pickend_real[i*down,1]-carpick_real[i*down,0],pickend_real[i*down,2],0,0,-pickend_real[i*down,2]-0.1,linestyle='-.',linewidth=1,alpha=0.5,color='#d62728')
    
    # ax1.plot3D(arm_xyztheta_path[:,1]+xy_path[:,0],arm_xyztheta_path[:,0]-xy_path[:,1],arm_xyztheta_path[:,2],linewidth=2)
    # ax1.plot3D(pickend_real[:,1]+carpick_real[:,0],pickend_real[:,0]-carpick_real[:,1],pickend_real[:,2],linewidth=2)
    plt.legend(('demonstration','reproduction','generalization','correction'),loc=1,fontsize=14)
    fontdict={'family':'Times New Roman','size':16}


    ax1.set_xlim((-1,2))
    ax1.set_ylim((-3,1))
    ax1.set_zlim((-0.1,0.2))
    ax1.grid(False)
    ax1.set_xticks([-1,0,1,2])
    ax1.set_yticks([-3,-2,-1,0,1])
    ax1.set_zticks([-0.1,0,0.1,0.2])


    ax1.set_xlabel('y/m',fontdict= fontdict)
    ax1.set_ylabel('x/m',fontdict=fontdict)
    ax1.set_zlabel('z/m',fontdict=fontdict)
    ax1.xaxis._axinfo['label']['space_factor']=4
    ax1.yaxis._axinfo['label']['space_factor']=4
    ax1.zaxis._axinfo['label']['space_factor']=4
    ax1.tick_params('z',labelsize=14)  
    ax1.tick_params('x',labelsize=14)  
    ax1.tick_params('y',labelsize=14)  
    # # make the panes transparent
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.azim=-160
    ax1.elev=50
    # make the grid lines transparent
    # ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    plt.savefig('IMAGE/tt4.png',bbox_inches='tight',dpi=300)
    plt.show()