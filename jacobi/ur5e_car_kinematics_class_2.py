import numpy as np
import jacobi.general_kinematics_casadi as Kine
import pdb

'''
ur5e_car 是底盘和机械臂的组合
'''

class UR5e_car_kinematics(Kine.Kinematics):

    def __init__(self):

        n = 10

        '''
        底盘参数
        '''
        height = 0.599
        offset = 0.22

        # x轴平移
        rotation_list_car = []
        position_list_car = []
        joint_type_list_car = []
        gravity_list_car = []

        Rx = np.array([
            [0,-1,0],
            [0,0,-1],
            [1,0,0]
        ])
        Rx = Rx.T

        Px = np.array([0, 0, height / 2])
        Gx = np.array([0, 0.85, 0])

        rotation_list_car.append(Rx)
        position_list_car.append(Px)
        joint_type_list_car.append("prismatic")
        gravity_list_car.append(Gx)


        # y轴平移
        Ry = np.array([
            [0,0,-1],
            [0,-1,0],
            [-1,0,0]
        ])
        Ry = Ry.T

        Py = np.array([0, 0, 0])
        Gy = np.array([0, 0, 0])

        rotation_list_car.append(Ry)
        position_list_car.append(Py)
        joint_type_list_car.append("prismatic")
        gravity_list_car.append(Gy)

        # z轴旋转
        Rz = np.array([
            [-1,0,0],
            [0,0,1],
            [0,1,0]
        ])
        Rz = Rz.T

        Pz = np.array([0, 0, 0])
        Gz = np.array([0, 0, 0])

        rotation_list_car.append(Rz)
        position_list_car.append(Pz)
        joint_type_list_car.append("revolute")        
        gravity_list_car.append(Gz)

        '''
        ur5e的参数，由于底盘的存在，ur5e的第一个旋转关节的位置需要改变
        '''

        rotation_list = []
        position_list = []
        joint_type_list = []

        # URe机器人的参数表
        d1 = 0.1625
        a2 = 0.425
        a3 = 0.3922
        d4 = 0.1333
        d5 = 0.0997
        tool_length=0.0
        d6 = 0.0996+tool_length

        # 关节0，认为0不转动时与base重合
        R0 = np.array([
            [0,1,0],
            [-1,0,0],
            [0,0,1]
        ])
        R0 = R0.T # 注意，坐标轴是列向量，因此需要转置一下, 旋转矩阵的列向量即为第二个坐标系在第一个坐标系中的坐标

        #P0 = np.array([0, 0, 0])
        P0 = np.array([ offset, 0, height/2])

        rotation_list.append(R0)
        position_list.append(P0)
        joint_type_list.append("revolute")

        G0 = np.array([0, 0, 2 * d1/3])

        # 关节 1
        R1 = np.array([
            [1,0,0],
            [0,0,1],
            [0,-1,0]
        ])
        R1 = R1.T

        P1 = np.array([0, 0, d1])

        rotation_list.append(R1)
        position_list.append(P1)
        joint_type_list.append("revolute")

        G1 = np.array([-a2/2, 0, 0.15])

        # 关节 2
        R2 = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        R2 = R2.T

        P2 = np.array([-a2, 0, 0]) # 仔细观察就会发现，实际上a2是在向着负方向运动的

        rotation_list.append(R2)
        position_list.append(P2)
        joint_type_list.append("revolute")

        G2 = np.array([-a3/2, 0, 0])

        # 关节3
        R3 = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        R3 = R3.T

        P3 = np.array([-a3, 0, 0])

        rotation_list.append(R3)
        position_list.append(P3)
        joint_type_list.append("revolute")

        G3 = np.array([0, 0, d4/2])

        # 关节4
        R4 = np.array([
            [1,0,0],
            [0,0,1],
            [0,-1,0]
        ])
        R4 = R4.T

        P4 = np.array([0, 0, d4])

        rotation_list.append(R4)
        position_list.append(P4)
        joint_type_list.append("revolute")

        G4 = np.array([0, 0, d5/2])

        # 关节5
        R5 = np.array([
            [1,0,0],
            [0,0,-1],
            [0,1,0]
        ])
        R5 = R5.T

        P5 = np.array([0, 0, d5])

        rotation_list.append(R5)
        position_list.append(P5)
        joint_type_list.append("revolute")

        G5 = np.array([0, 0, d6/2])

        # 关节6，固定fixed，用于标出end effector的位置
        R6 = np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
        R6 = R6.T

        P6 = np.array([0, 0, d6])

        rotation_list.append(R6)
        position_list.append(P6)
        joint_type_list.append("fixed")

        G6 = np.array([0, 0, 0])
        #self.ur5e_kine = Kine.Kinematics(n, rotation_list, position_list, joint_type_list)

        gravity_list = [G0, G1, G2, G3, G4, G5, G6]

        super(UR5e_car_kinematics, self).__init__(n, rotation_list_car + rotation_list, position_list_car + position_list, joint_type_list_car + joint_type_list, gravity_list_car + gravity_list)


    def get_jacobian_lx(self, q_joint):

        angle_z = q_joint[2]

        jacob = self.get_jacobian(q_joint)

        lx_matrix = np.zeros((9, 8)) 

        lx_matrix[0, 0] = np.cos(angle_z)
        lx_matrix[1, 0] = np.sin(angle_z)

        lx_matrix[2:, 1:] = np.eye(7)

        return np.matmul(jacob, lx_matrix)


if __name__ == "__main__":

    ur5e_car = UR5e_car_kinematics()

    q_joint = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ee = ur5e_car.get_end_effector_posture(q_joint)

    jacob = ur5e_car.get_jacobian(q_joint)

    pdb.set_trace()


