import numpy as np
#import sys
#sys.path.append('/home/vision/casadi')
import casadi as cs
import pdb

'''
通用的运动学求解类

将sympy 替换成casadi
'''

func_opts = {"jit": True, "jit_options": {"flags": "-Ofast"}}

# 旋转关节
def rotz(theta):

    Rz = cs.SX.zeros(3,3)

    Rz[0,0] += cs.cos(theta)
    Rz[0,1] += -cs.sin(theta)

    Rz[1,0] += cs.sin(theta)
    Rz[1,1] += cs.cos(theta)

    Rz[2,2] = 1

    return Rz

def translate_revolute(R, P, theta):
    T = cs.SX.zeros(4,4)
    T[:3,:3] = cs.SX(R) @ rotz(theta)
    T[:3, 3] = P
    T[3, 3] = 1
    return T

# 平移关节

def translate_prismatic(R, P, distance):
    # np array 一定要通国 cs.SX() 转化为casadi类型
    T = cs.SX.zeros(4,4)
    
    T[:3,:3] = R 
    T[:3, 3] = P
    T[3, 3] = 1

    Tp = cs.SX.eye(4)
    Tp[2, 3] += distance

    return T @ Tp    

# 固定关节

def translate_fixed(R, P):
    T = cs.SX.zeros(4,4)
    T[:3,:3] = R
    T[:3, 3] = P
    T[3, 3] = 1
    return T    

class Kinematics:

    def __init__(self, n, R_list, P_list, joint_type, gravity_position_vector_list = None):
        
        '''
        n 关节数目
        R_list 当前关节到下一个关节的旋转矩阵 3x3
        P_list 当前关节到下一个关节的平移向量 3
        joint_type 是一个list，长度为n
        gravity_position_vector 重心的位置
        '''

        assert len(R_list) == n and len(P_list) == n and len(joint_type) == n, "dimension not match"

        joint_type_sigma_list = []

        self.T_list = []

        self.nb_fixed_joint = 0

        for i in range(n):

            assert joint_type[i] in ["revolute", "prismatic", "fixed"], "unknow joint type"

            if joint_type[i] == "fixed":
                self.nb_fixed_joint += 1


        self.nb_active_joint = n - self.nb_fixed_joint # 可运动关节的数目

        self.q = cs.SX.sym('q', self.nb_active_joint)
        self.q_dot = cs.SX.sym('q_dot', self.nb_active_joint)
        self.q_ddot = cs.SX.sym('q_ddot', self.nb_active_joint)

        joint_counter = 0

        for i in range(n):
            
            if joint_type[i] == "revolute":
                joint_type_sigma_list.append(0)
                _T = translate_revolute(R_list[i], P_list[i], self.q[joint_counter])
                joint_counter += 1

            elif joint_type[i] == "prismatic":
                joint_type_sigma_list.append(1)
                _T = translate_prismatic(R_list[i], P_list[i], self.q[joint_counter])
                joint_counter += 1

            else:
                # 固定关节
                joint_type_sigma_list.append(-1)
                _T = translate_fixed(R_list[i], P_list[i])
                

            self.T_list.append(_T)


        self.T_world = [] # 计算每个关节轴相对于大地的坐标系，不含fixed关节
        self.T_world_all = [] # 所有坐标变换都记录

        # 计算正运动学
        for i in range(n):

            _T_world = cs.SX.eye(4)
            for j in range(i+1):
                _T_world = _T_world @ self.T_list[j]

            self.T_world_all.append(_T_world)
            
            if joint_type_sigma_list[i] >= 0 or i == n-1:
                # 只记录活动关节和end effector的变换矩阵
                self.T_world.append(_T_world)


        self.Joint_func_list = []
        
        for i in range(len(self.T_world)):
            _joint_func = cs.Function("joint_func" + str(i), [self.q], [self.T_world[i]], func_opts)
            self.Joint_func_list.append(_joint_func)


        self.gravity_world_list = []

        if gravity_position_vector_list != None:
            # 中心只是位置相对转轴改变了，但是朝向并没有改变
            assert len(gravity_position_vector_list) == n

            for i in range(n):

                _T_gravity = cs.SX.eye(4)
                _T_gravity[:3, 3] = gravity_position_vector_list[i]

                _gravity_world = self.T_world_all[i] @ _T_gravity

                self.gravity_world_list.append(_gravity_world)

        self.gravity_func_list = []

        for i in range(len(self.gravity_world_list)):
            _gravity_func = cs.Function("gravity_func" + str(i), [self.q], [self.gravity_world_list[i]], func_opts)
            self.gravity_func_list.append(_gravity_func)

        # 计算雅可比矩阵
        # 获得每个关节轴的向量z
        # 获得每个关节轴到ee的向量r

        self.z = []
        self.r = []

        for i in range(self.nb_active_joint):

            _z = self.T_world[i][:3, 2]
            _r = self.T_world[-1][:3, 3] - self.T_world[i][:3, 3]  
                
            self.r.append(_r)
            self.z.append(_z)

        self.Jacobian = None

        for i in range(self.nb_active_joint):

            if joint_type_sigma_list[i] == 0:
                JP = cs.cross(self.z[i], self.r[i])
                JO = self.z[i]
                
            elif joint_type_sigma_list[i] == 1:
                JP = self.z[i]
                JO = cs.SX([0, 0, 0])
            
            J = cs.vertcat(JP, JO)

            if i == 0:
                self.Jacobian = J
            else:
                self.Jacobian = cs.horzcat(self.Jacobian, J)

        self.Jacobian_func = cs.Function("jacobian_func", [self.q], [self.Jacobian], func_opts)

        # 求解加速度关系，通过雅可比得到关节所在杆件的速度和角速度
        
        self.velocity_list = []
        self.angular_list = []

        for i in range(self.nb_active_joint):

            _state = self.Jacobian[:,: i+1] @ self.q_dot[:i+1]
            
            self.velocity_list.append(_state[:3,:])
            self.angular_list.append(_state[3:,:])


        self.Jacobian_dot = None

        for i in range(self.nb_active_joint):

            joint2end_velocity = self.velocity_list[self.nb_active_joint - 1] - self.velocity_list[i]

            if joint_type_sigma_list[i] == 0:
                # revolute joint
                JP_dot = cs.cross(cs.cross(self.angular_list[i],self.z[i]), self.r[i]) + cs.cross(self.z[i], joint2end_velocity)
                JO_dot = cs.cross(self.angular_list[i], self.z[i])

            elif joint_type_sigma_list[i] == 1:
                # prismatic joint
                JP_dot = cs.cross(self.angular_list[i], self.z[i])
                JO_dot = cs.SX([0, 0, 0])

            J_dot = cs.vertcat(JP_dot, JO_dot)

            if i == 0:
                self.Jacobian_dot = J_dot
            else:
                self.Jacobian_dot = cs.horzcat(self.Jacobian_dot, J_dot)

    def get_end_effector_posture(self, q_joint):

        ee_posture = self.Joint_func_list[-1](q_joint)

        return ee_posture

    def get_joint_axis_posture(self, q_joint):
        
        joint_pose = []
        for _j_func in self.Joint_func_list:
            joint_pose.append( np.array(_j_func(q_joint)) )

        return joint_pose


    def get_gravity_posture(self, q_joint):
        
        gravity_pose = []
        for _g_func in self.gravity_func_list:
            gravity_pose.append( np.array(_g_func(q_joint)) )

        return gravity_pose

    def get_forward_kinematics(self, q_joint):

        pass

    def get_jacobian(self, q_joint):

        jacob = self.Jacobian_func(q_joint)

        return np.array(jacob, dtype=float)
