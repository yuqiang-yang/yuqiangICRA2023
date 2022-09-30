import numpy as np
import ikfastpy

# Initialize kinematics for UR5 robot arm
ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()

joint_angles = [ 0.21819526 ,-2.83339381 , 1.94259298 ,-2.22738194 ,-0.21731699 , 3.12555647] # in radians

# Test forward kinematics: get end effector pose from joint angles
print("\nTesting forward kinematics:\n")
print("Joint angles:")
print(joint_angles)
ee_pose = ur5_kin.forward(joint_angles)
ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
print("\nEnd effector pose:")
print(ee_pose)
print("\n-----------------------------")
ee_pose = np.array(([[ 9.99976312e-01,-6.81907112e-03,9.35938603e-04,0.17611989378929138],
 [ 9.01488332e-04,-5.05337078e-03,-9.99986825e-01,-0.15506547689437866],
 [ 6.82371093e-03,9.99963981e-01,-5.04710377e-03,0.6172924041748047]]))
# Test inverse kinematics: get joint angles from end effector pose
print("\nTesting inverse kinematics:\n")
joint_configs = ur5_kin.inverse(ee_pose.reshape(-1).tolist())
n_solutions = int(len(joint_configs)/n_joints)
print("%d solutions found:"%(n_solutions))
joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
for joint_config in joint_configs:
    print(joint_config)
    print(np.linalg.norm(joint_config-joint_angles))

# Check cycle-consistency of forward and inverse kinematics
assert(np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-4 for joint_config in joint_configs]))
print("\nTest passed!")