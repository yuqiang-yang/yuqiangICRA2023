from turtle import color
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
import matplotlib.pyplot as plt;            
import scipy.interpolate as si
def read_data2():
    #给定训练数据
    t = np.array((0.9,2.1,3.2)).reshape(-1,1)
    y = np.array((1.1*100,2.5*100,3.1*100)).reshape(-1,1)
    return (t,y)

def get_via_point_sigma_function(sigma_max,x,sigma):
    #给定最大的方差，以及每个时间点的方差，得到二次差值平滑的方差函数曲线
    rows = x.shape[0]
    for i in range(rows):
            if i==rows-1:
                    break
            x=np.append(x,(x[i]+x[i+1])/2)
            sigma=np.append(sigma,sigma_max)
    quadratic = si.interp1d(x, sigma, kind="quadratic")
    return quadratic    


def plot1(t_predict,y_mean,y_std,i=0):
    #画图
    plt.figure()
    plt.plot(t_predict,y_mean[:],color='darkblue')
    plt.fill_between(t_predict.ravel(),
    y_mean.ravel() - 1.96 * y_std.ravel(),
    y_mean.ravel() + 1.96 * y_std.ravel(),
    alpha=0.7,
    color='cornflowerblue'
    )
    plt.fill_between(t_predict.ravel(),
    y_mean.ravel() - 3 * y_std.ravel(),
    y_mean.ravel() + 3 * y_std.ravel(),
    alpha=0.3,
    color='cornflowerblue'
    )        


t,y = read_data2()
##kernel1为初始根据训练数据的高斯过程
kernel1 = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 2e1))
gpr1 = GaussianProcessRegressor(kernel=kernel1,random_state=0,n_restarts_optimizer=2,normalize_y=True)

gpr1.fit(t,y)
t_predict = np.linspace(0,4,1000).reshape(-1,1)
y_mean1,y_std1 = gpr1.predict(t_predict,return_std=True)    #预测结果的均值和方差
##到这一步完成了初始数据的训练和预测

#给定中间点的时间t，位置y，和方差sigma，注：为了保证初始轨迹不变，所有初始轨迹的起点和终点均包含在via point中，方差设小
t_via=np.array((0,1.33333,2.66666,4)).reshape(-1,1)
#注：由于via point的GP的方差是我们先验给定的，所以不要给白噪声作为kernel
kernel2 = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3))  
#各中间点的方差，这个是人为给定
noise = np.array((0.1,10,10,0.1))      
noise_ = noise.reshape(-1,1)
#这个插值函数目前是根据自带的interp1d实现的，但是其实是有些小问题的，可能会差值出方差小于0的点，所以实操时报错最好检验一下这个
quadratic = get_via_point_sigma_function(50,t_via,noise_)   
#给定经过via point的y，起点设为初始轨迹的起点，终点设为初始轨迹的终点。中间两个点人工给定
y_via = np.array([y_mean1[0],200,240,y_mean1[-1]],dtype=object).reshape(-1,1)   

#via point的GP模型，其中方差的噪声给定
gpr2 = GaussianProcessRegressor(kernel=kernel2,random_state=0,n_restarts_optimizer=5,alpha=noise)
gpr2.fit(t_via,y_via)
y_mean2,y_std2 = gpr2.predict(t_predict,return_std=True)
#加上先验的方差
y_std2 = y_std2 + quadratic(t_predict).ravel()


#画图
plot1(t_predict,y_mean1,y_std1)
plt.plot(t_via[0],y_via[0],'o',label='o')
plt.plot(t_via[1],y_via[1],'o',label='o')
plt.plot(t_via[2],y_via[2],'o',label='o')
plt.plot(t_via[3],y_via[3],'o',label='o')
plt.ylim(0,400)
plot1(t_predict,y_mean2,y_std2)

plt.plot(t_via[0],y_via[0],'o',label='o')
plt.plot(t_via[1],y_via[1],'o',label='o')
plt.plot(t_via[2],y_via[2],'o',label='o')
plt.plot(t_via[3],y_via[3],'o',label='o')
plt.ylim(0,400)

#高斯融合的过程
y_std1 += 1e-8
y_std2 += 1e-8
blend_std = 1/(1/(y_std1)+1/(y_std2))
y_mean1 = y_mean1.ravel()
y_mean2 = y_mean2.ravel()
blend_mean = blend_std*(1/(y_std1)*y_mean1 + 1/(y_std2)*y_mean2)

plot1(t_predict,blend_mean,blend_std)
plt.plot(t_predict,y_mean1,'r--')
plt.plot(t_via[0],y_via[0],'o',label='o')
plt.plot(t_via[1],y_via[1],'o',label='o')
plt.plot(t_via[2],y_via[2],'o',label='o')
plt.plot(t_via[3],y_via[3],'o',label='o')
plt.ylim(0,400)

plt.show()

