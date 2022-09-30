from operator import length_hint
from turtle import color
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
import matplotlib.pyplot as plt;            
import scipy.interpolate as si
import seaborn as sns
import time
from scipy import signal
pi = 3.1415926
#输入n*3的一个位置数组，输出一个n*3的差分向量，用于速度
def get_velocity(position):
    rows,cols = position.shape
    velocity = np.zeros((rows,cols))
    for i in range(rows - 1):
        velocity[i] = position[i+1] - position[i]
    velocity[rows-1] = velocity[rows-2]

    return velocity

#先假设是二维的情形
def get_minvariance_path(gpr,query_point,demonstrate_point):
    minVariance = 9999
    minPoint = query_point
    for p in demonstrate_point:
        length = np.sqrt((p[0] - query_point[0])**2 + (p[1] - query_point[1])**2)
        step = int(length//0.2)
        if step == 0:
            _,std = gpr.predict(p.reshape(1,-1),return_std=True)
            if std <= minVariance:
                minVariance = std
                minPoint = p
            continue
        else:
            xstep = (p[0] - query_point[0])/step
            ystep = (p[1] - query_point[1])/step
            var = 0

        x=np.zeros((step+1,2))
        for i in range(step):
            x[i,:] = np.array([query_point[0] + xstep * i , query_point[1] + ystep * i]).reshape(1,-1)

        x[step,:] = np.array(p).reshape(1,-1)
        _,std = gpr.predict(x,return_std=True)
        var = np.sum(std)
        if var <= minVariance:
            minVariance = var
            minPoint = p
            if minVariance < 1e-6:
                print('break')
                break
            #print(minVariance,'  ',p)
        
    return minPoint

    
#画场图
def plot_field(gpr,xmin,xmax,ymin,ymax,demonstrate_point,step=400,field_density = 20):
    x_step = (xmax - xmin)/step
    y_step = (ymax - ymin)/step
    xx = np.arange(xmin,xmax,x_step)
    yy = np.arange(ymin,ymax,y_step)
    xxx,yyy = np.meshgrid(xx,yy)
    predict_point = np.concatenate((xxx.reshape(-1,1),yyy.reshape(-1,1)),1)
    result,std = gpr.predict(predict_point,return_std=True)


    diff_std = np.zeros((step,step,2))
    std_matrix = std[:].reshape(step,step)
    

    for i in range(step-1):
        for j in range(step-1):
            # if i == 0 or i == step - 2 or j == 0 or j == step - 2:
            #     continue
            diff_std[i,j,0] = (std_matrix[i,j+1] - std_matrix[i,j])
            diff_std[i,j,1] = (std_matrix[i+1,j] - std_matrix[i,j])

    diff_std[:,:,0] /= x_step
    diff_std[:,:,1] /= y_step
    sns.heatmap(abs(diff_std[1:-1,1:-1,0]))
    plt.figure()
    sns.heatmap(abs(diff_std[1:-1,1:-1,1]))
    print(diff_std.shape)

    #plt.show()


    x_step = (xmax - xmin)/field_density
    y_step = (ymax - ymin)/field_density
    xx = np.arange(xmin,xmax,x_step)
    yy = np.arange(ymin,ymax,y_step)
    xxx,yyy = np.meshgrid(xx,yy)
    predict_point = np.concatenate((xxx.reshape(-1,1),yyy.reshape(-1,1)),1)
    result,std = gpr.predict(predict_point,return_std=True)
    
    plt.figure()

    plt.scatter(predict_point[:,0],predict_point[:,1])

    for i in range(field_density**2):
        plt.arrow(predict_point[i,0],predict_point[i,1],result[i,0],result[i,1],head_width=0.01)
    
       
    
    plt.figure()
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
        plt.arrow(predict_point[i,0],predict_point[i,1], result[i,0]- diff_std[remap_i,remap_j,0]*distance, result[i,1]- diff_std[remap_i,remap_j,1]*distance,head_width=0.01)

        #print(i)
    #plt.show()
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

if __name__ == '__main__':


    xy_path,xy_dot_path,xy_theta_path,xy_theta_dot_path = readTxt()
    #+0.0001*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-7))
    kernel1 = 1 * RBF(length_scale=1e-3, length_scale_bounds=(1e-8, 1e2))+1*WhiteKernel(noise_level = 1e-7,noise_level_bounds=(1e-15,1e-2))
    gpr1 = GaussianProcessRegressor(kernel=kernel1,random_state=0,n_restarts_optimizer=5,normalize_y=False)
    xy_dot_path_filtered = mean_filter(xy_dot_path)
    xy_dot_path_filtered_down = down_sample(xy_dot_path_filtered,4)
    xy_path_down = down_sample(xy_path,4)
    # print(xy_dot_path_filtered.shape)
    #plt.plot(xy_dot_path_filtered_down)
    #plt.show()

    gpr1.fit(xy_path_down,xy_dot_path_filtered_down)

    #print(get_minvariance_path(gpr1,np.array([.5,0.5]),xy_path))
    plot_field(gpr1,0,3.2,0,3.2,xy_path)
    plt.plot(xy_path[:,0],xy_path[:,1])
    plt.show()

