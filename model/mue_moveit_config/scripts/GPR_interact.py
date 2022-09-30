from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt;            # doctest: +SKIP
import scipy.interpolate as si
import seaborn as sns
def read_data():
    position1 = np.loadtxt('Robot_Position9.txt')
    position3 = np.loadtxt('Robot_Position14.txt')
    position4 = np.loadtxt('Robot_Position12.txt')
    position5 = np.loadtxt('Robot_Position13.txt')

    position1[:,2] = (position1[:,2] - np.min(position1[:,2]))*2 + np.min(position1[:,2])
    position3[:,2] = (position3[:,2] - np.min(position3[:,2]))*2 + np.min(position3[:,2])
    position4[:,2] = (position4[:,2] - np.min(position4[:,2]))*2 + np.min(position4[:,2])
    position5[:,2] = (position5[:,2] - np.min(position5[:,2]))*2 + np.min(position5[:,2])



    position1[:,1] = (position1[:,1] - np.min(position1[:,1]))/1.2 + np.min(position1[:,1])
    position3[:,1] = (position3[:,1] - np.min(position3[:,1]))/1.2 + np.min(position3[:,1])
    position4[:,1] = (position4[:,1] - np.min(position4[:,1]))/1.2 + np.min(position4[:,1])
    position5[:,1] = (position5[:,1] - np.min(position5[:,1]))/1.2 + np.min(position5[:,1])

    y=np.vstack((position1,position3,position4,position5))
    t=np.hstack((np.linspace(0,0.01*position1.shape[0],position1.shape[0]),
    np.linspace(0,0.01*position3.shape[0],position3.shape[0]),
    np.linspace(0,0.01*position4.shape[0],position4.shape[0]),
    np.linspace(0,0.01*position5.shape[0],position5.shape[0]))).reshape(-1,1)
    return (t,y[:,1])
def plot1(t_predict,y_mean,y_std,i=0):
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

#得到人工干预的轨迹t，y，和方差，这部分的话，又臭又长，大概就是，要把干预线段的GP模型给建出来 
def get_interactive(end_std=0.1,mid_std=5,left_time = 1,right_time = 2):
    t=np.array((0))     #将起点先加进去
    sigma_interact=np.array((end_std))  #起点和终点的方差
    y=np.array((-437.23))   #起点的y设置为轨迹的起点y
    #给定中间的干预线段
    for i in range((right_time-left_time)*100):
        t = np.append(t,1+0.01*i)
        y = np.append(y,-390+0.1*i)
        sigma_interact = np.append(sigma_interact,mid_std)
    #将终点加进去
    t = np.append(t,4)
    y = np.append(y,-188.23)
    sigma_interact = np.append(sigma_interact,0.1)
    t = t.reshape(-1,1)
    y = y.reshape(-1,1)
    sigma_interact = sigma_interact.reshape(-1,1)
    

    #下面这部分是要得到除了干预位置其他位置的方差
    t_predict1 = np.linspace(0,left_time,100).reshape(-1,1)
    t_predict2 = np.linspace(right_time,4,200).reshape(-1,1)
    quadratic1 = get_via_point_sigma_function(30,np.array((0,1)),np.array((mid_std,mid_std)))
    quadratic2 = get_via_point_sigma_function(30,np.array((2,4)),np.array((mid_std,mid_std)))

    sigma_all = quadratic1(t_predict1)
    for i in range((right_time-left_time)*100):
        sigma_all = np.append(sigma_all,mid_std)
    sigma_all = np.append(sigma_all,quadratic2(t_predict2))
    return (t,y,sigma_interact,sigma_all)


#干预部分GPR训练
t_,y_,sigma_int,sigma_all = get_interactive()
kernel2 = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3))
gpr2 = GaussianProcessRegressor(kernel=kernel2,random_state=0,n_restarts_optimizer=5,alpha=sigma_int.ravel())
gpr2.fit(t_,y_)
t_predict = np.linspace(0,4,400).reshape(-1,1)
y_mean2,y_std2 = gpr2.predict(t_predict,return_std=True)
y_std2 +=sigma_all

#初始部分GPR训练
t,y = read_data()
kernel1 = 1.0 * RBF(length_scale=1e0, length_scale_bounds=(1e-8, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 2e1))
gpr1 = GaussianProcessRegressor(kernel=kernel1,random_state=0,n_restarts_optimizer=2,normalize_y=True)

gpr1.fit(t,y)
t_predict = np.linspace(0,4,400).reshape(-1,1)
y_mean1,y_std1 = gpr1.predict(t_predict,return_std=True)

y_std1 += 1e-8
y_std2 += 1e-8
blend_std = 1/(1/(y_std1)+1/(y_std2))
y_mean1 = y_mean1.ravel()
y_mean2 = y_mean2.ravel()

blend_mean = blend_std*(1/(y_std1)*y_mean1 + 1/(y_std2)*y_mean2)

#画图
plot1(t_predict,y_mean1,y_std1)
plot1(t_predict,y_mean2,y_std2)
plot1(t_predict,blend_mean,blend_std)

plt.show()






