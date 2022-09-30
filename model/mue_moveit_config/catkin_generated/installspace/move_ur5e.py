#!/usr/bin/env python2

## To use the Python MoveIt interfaces, we will import the `moveit_commander`_ namespace.
## This namespace provides us with a `MoveGroupCommander`_ class, a `PlanningSceneInterface`_ class,
## and a `RobotCommander`_ class. (More on these below)
##
## We also import `rospy`_ and some messages that we will use:
##

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from tf.transformations import euler_from_quaternion, quaternion_from_euler

sys.path.append('/home/robot/.local/lib/python2.7/site-packages')

def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True

class MoveGroupPythonIntefaceUR5e(object):
  """MoveGroupPythonInteface"""
  def __init__(self):
    super(MoveGroupPythonIntefaceUR5e, self).__init__()

    ## First initialize `moveit_commander`_ and a `rospy`_ node:

    ### need to remap /joint_states to /robot1_ns/joint_states, important !!!! do it like below or pass the argument in command line using sys.argv
    joint_state_topic = ['joint_states:=/robot1_ns/joint_states']
    moveit_commander.roscpp_initialize(joint_state_topic)
    # moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('move_group_python_interface_move_ur5e', anonymous=True)

    ## Instantiate a `RobotCommander`_ object. This object is the outer-level interface to
    ## the robot:
    robot = moveit_commander.RobotCommander()

    ## Instantiate a `PlanningSceneInterface`_ object.  This object is an interface
    ## to the world surrounding the robot:
    scene = moveit_commander.PlanningSceneInterface()

    ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
    ## to one group of joints.  
    group_name = "ur5e_arm"
    group = moveit_commander.MoveGroupCommander(group_name) 

    ## We create a `DisplayTrajectory`_ publisher which is used later to publish
    ## trajectories for RViz to visualize:
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    # We can get the name of the reference frame for this robot:
    planning_frame = group.get_planning_frame()
    print "============ Planning Reference frame: %s" % planning_frame
    # group.set_pose_reference_frame('/camera_link')
    pose_frame = group.get_pose_reference_frame()
    print "============ Pose Reference frame: %s" % pose_frame

    # We can also print the name of the end-effector link for this group:
    eef_link = group.get_end_effector_link()
    print "============ End effector: %s" % eef_link

    # # set number of times the planner computes the plans:
    # group.set_num_planning_attempts(3)

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print "============ Robot Groups:", robot.get_group_names()

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print "============ Printing robot state"
    print robot.get_current_state()
    print ""

    # Misc variables
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.group = group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names

  def go_to_joint_state(self):
    ## Planning to a Joint Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^^
    ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_ so the first
    ## thing we want to do is move it to a slightly better configuration.
    # We can get the joint values from the group and adjust some of the values:
    joint_goal = self.group.get_current_joint_values()
    joint_goal[0] = 0
    joint_goal[1] = -pi/4
    joint_goal[2] = 0
    joint_goal[3] = -pi/2
    joint_goal[4] = 0
    joint_goal[5] = pi/3
    joint_goal[6] = 0

    # The go command can be called with joint values, poses, or without any
    # parameters if you have already set the pose or joint target for the group
    self.group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    self.group.stop()

    # For testing:
    # Note that since this section of code will not be included in the move_ur5es
    # we use the class variable rather than the copied state variable
    current_joints = self.group.get_current_joint_values()
    return all_close(joint_goal, current_joints, 0.01)

  def go_to_pose_goal(self):
    ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## We can plan a motion for this group to a desired pose for the
    ## end-effector:
    pose_goal = geometry_msgs.msg.Pose()
    quat = quaternion_from_euler(0.0, 0.0, -1.57079)
    pose_goal.orientation.x = quat[0]
    pose_goal.orientation.y = quat[1]
    pose_goal.orientation.z = quat[2]
    pose_goal.orientation.w = quat[3]
    pose_goal.position.x = 0.05
    pose_goal.position.y = 0.27
    pose_goal.position.z = 0.5
    # pose_goal.header.frame_id = "camera_link"   
    self.group.set_pose_target(pose_goal)

    ## Now, we call the planner to compute the plan and execute it.
    plan = self.group.go(wait=True)
    # Calling `stop()` ensures that there is no residual movement
    self.group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    self.group.clear_pose_targets()

    # For testing:
    # we use the class variable rather than the copied state variable
    current_pose = self.group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)

  def go_to_pose_goal_with_constraints(self, pose_goal):
        ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    self.group.set_planning_time(10.0)
    self.group.set_num_planning_attempts(5)
    # self.group.set_planner_id('RRTstar')
    self.group.set_constraints_database('127.0.0.1', 33829)

    current_pose = self.group.get_current_pose().pose
    home_pose = current_pose

    # specify the orientation path constraint to keep the eef facing forward
    ocm = moveit_msgs.msg.OrientationConstraint()
    ocm.link_name = self.eef_link
    ocm.header.frame_id = self.planning_frame
    # ocm.orientation.x = quat[0]
    # ocm.orientation.y = quat[1]
    # ocm.orientation.z = quat[2]
    # ocm.orientation.w = quat[3]
    ocm.orientation = pose_goal.orientation
    ocm.absolute_x_axis_tolerance = 3.14
    ocm.absolute_y_axis_tolerance = 0.1
    ocm.absolute_z_axis_tolerance = 0.1
    ocm.weight = 1.0
    constraints = moveit_msgs.msg.Constraints()
    constraints.name = "cable_placement_constraints"
    constraints.orientation_constraints.append(ocm)

    # current_pose = self.group.get_current_pose().pose
    print "current pose: "
    print current_pose

    

    # # set the pose of the robot to a starting state that already meeting the constraints
    # start_pose = geometry_msgs.msg.Pose()
    # start_pose.orientation = pose_goal.orientation
    # start_pose.position = current_pose.position    
    # start_pose.position.x = current_pose.position.x - 0.4
    
    # ### ----- TODO ----- ###
    # # start_state = self.robot.get_current_state()
    # # joint_model_group = start_state->get_joint_model_group()  ## TODO: add python bindings to get the joint_model_group
    # # start_state.set_from_IK(joint_model_group, start_pose)  ## TODO: add python bindings for setting the start state using IK
    # # self.group.set_start_state(start_state)
    # # self.group.set_start_state_to_current_state()
 
    # # # for now, use the set_pose_target function instead, to replace the set_start_state()
    # self.group.set_pose_target(start_pose)
    # plan = self.group.go(wait=True)
    # # Calling `stop()` ensures that there is no residual movement
    # self.group.stop()
    # self.group.clear_pose_targets()

    key = raw_input("============ Hit `Enter` to make a move, or key-in `q` to exit ...")

    if not rospy.is_shutdown() and key != 'q':
      try:           
        self.group.set_pose_target(pose_goal)

        ### ---- set the path constraint ---- ###
        self.group.set_path_constraints(constraints)

      # if not rospy.is_shutdown():
        execute_num = 0
        plan = False
        ## Now, we call the planner to compute the plan and execute it.
        while not plan and execute_num < 10:
          plan = self.group.go(wait=True)
          execute_num += 1
        
        # Calling `stop()` ensures that there is no residual movement
        self.group.stop()

        # It is also always good to clear your constraints after planning.
        self.group.clear_path_constraints()
        self.group.clear_pose_targets()

        # # return to home
        # self.group.set_pose_target(home_pose)
        # plan = self.group.go(wait=True)
        # self.group.stop()
        # self.group.clear_pose_targets()
        
        # For testing:
        # we use the class variable rather than the copied state variable
        current_pose = self.group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

      except rospy.ROSInterruptException:
        return
      except KeyboardInterrupt:
        return

    else:
      print "Quit on user request ..."
      return

  def plan_cartesian_path(self, pose_goal, scale=1):

    ## Cartesian Paths
    ## ^^^^^^^^^^^^^^^
    ## You can plan a Cartesian path directly by specifying a list of waypoints
    ## for the end-effector to go through:
    ## NOTE: the position and orientation must be properly set such that the interpolation creates feasible path for the robot, 
    ## otherwise the robot may crush during execution due to singularity
    waypoints = []

    wpose = self.group.get_current_pose().pose
    # wpose.position.z -= scale * 0.1  # First move up (z)
    # wpose.position.y += scale * 0.2  # and sideways (y)
    # quat = quaternion_from_euler(0.0, 3.14, -1.57079)
    # wpose.orientation.x = quat[0]
    # wpose.orientation.y = quat[1]
    # wpose.orientation.z = quat[2]
    # wpose.orientation.w = quat[3]
    # waypoints.append(copy.deepcopy(wpose))
    
    wpose.position.x -= 0.1   
    waypoints.append(copy.deepcopy(wpose))

    # wpose.position.y += scale * 0.2  # Second move forward/backwards in (x)
    wpose.position.x -= 0.3
    wpose.orientation = pose_goal.orientation
    waypoints.append(copy.deepcopy(wpose))

    wpose.position.z += scale * 0.1
    wpose.position.x -= 0.1
    waypoints.append(copy.deepcopy(wpose))

    wpose.position.y += scale * 0.25  # Third move sideways (y)
    waypoints.append(copy.deepcopy(wpose))

    wpose.position.x -= scale * 0.1  # Third move sideways (y)
    waypoints.append(copy.deepcopy(wpose))

    # We want the Cartesian path to be interpolated at a resolution of 1 cm
    # which is why we will specify 0.01 as the eef_step in Cartesian
    # translation.  We will disable the jump threshold by setting it to 0.0 disabling:
    (plan, fraction) = self.group.compute_cartesian_path(
                                       waypoints,   # waypoints to follow
                                       0.01,        # eef_step
                                       0.0)         # jump_threshold

    # Note: We are just planning, not asking move_group to actually move the robot yet:
    return plan, fraction


  def plan_demo_waypoints(self, demo_path, scale=1):
    
    ## Cartesian paths obtained from demonstrated waypoints
    ## ^^^^^^^^^^^^^^^
    ## You can plan a Cartesian path directly by specifying a list of waypoints obtainted from demonstration or prescribed path
    ## for the end-effector to go through:
    ## NOTE: the position and orientation must be properly set such that the interpolation creates feasible path for the robot, 
    ## otherwise the robot may crush during execution due to singularity

    waypoints = []

    for wpose in demo_path:
      waypoints.append(copy.deepcopy(wpose))

    # We want the Cartesian path to be interpolated at a resolution of 1 cm
    # which is why we will specify 0.01 as the eef_step in Cartesian
    # translation.  We will disable the jump threshold by setting it to 0.0 disabling:
    (plan, fraction) = self.group.compute_cartesian_path(
                                       waypoints,   # waypoints to follow
                                       0.01,        # eef_step
                                       0.0)         # jump_threshold

    # Note: We are just planning, not asking move_group to actually move the robot yet:
    return plan, fraction

  def display_trajectory(self, plan):
    ## Displaying a Trajectory
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
    ## group.plan() method does this automatically so this is not that useful
    ## here (it just displays the same trajectory again):
    ##
    ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
    ## We populate the trajectory_start with our current robot state to copy over
    ## any AttachedCollisionObjects and add our plan to the trajectory.
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = self.robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    # Publish
    self.display_trajectory_publisher.publish(display_trajectory);


  def execute_plan(self, plan):
    ## **Note:** The robot's current joint state must be within some tolerance of the
    ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
    self.group.execute(plan, wait=True)


  def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
    # Copy class variables to local variables to make the web move_ur5es more clear.
    # In practice, you should use the class variables directly unless you have a good
    # reason not to.
    box_name = self.box_name
    scene = self.scene

    ## Ensuring Collision Updates Are Receieved
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ## If the Python node dies before publishing a collision object update message, the message
    ## could get lost and the box will not appear. To ensure that the updates are
    ## made, we wait until we see the changes reflected in the
    ## ``get_known_object_names()`` and ``get_known_object_names()`` lists.
    ## For the purpose of this move_ur5e, we call this function after adding,
    ## removing, attaching or detaching an object in the planning scene. We then wait
    ## until the updates have been made or ``timeout`` seconds have passed
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
      # Test if the box is in attached objects
      attached_objects = self.scene.get_attached_objects([self.box_name])
      is_attached = len(attached_objects.keys()) > 0

      # Test if the box is in the scene.
      # Note that attaching the box will remove it from known_objects
      is_known = self.box_name in self.scene.get_known_object_names()

      # Test if we are in the expected state
      if (box_is_attached == is_attached) and (box_is_known == is_known):
        return True

      # Sleep so that we give other threads time on the processor
      rospy.sleep(0.1)
      seconds = rospy.get_time()

    # If we exited the while loop without returning then we timed out
    return False


def main():
  try:

    move_ur5e = MoveGroupPythonIntefaceUR5e()

    ## We can plan a motion for this group to a desired pose for the
    ## end-effector
    quat = quaternion_from_euler(0.0, 0.0, -1.57079)
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.x = quat[0]
    pose_goal.orientation.y = quat[1]
    pose_goal.orientation.z = quat[2]
    pose_goal.orientation.w = quat[3]
    pose_goal.position.x = 0.05
    pose_goal.position.y = 0.32 #0.27
    pose_goal.position.z = 0.5   
    
    # # print "============ Press `Enter` to execute a movement using a joint state goal ..."
    # # raw_input()
    # # move_ur5e.go_to_joint_state()

    # TODO: if 
    # print "============ Press `Enter` to execute a movement using the pose goal ..."
    # raw_input()
    # # move_ur5e.go_to_pose_goal()
    # move_ur5e.go_to_pose_goal_with_constraints()

    # key = raw_input("============ Hit `Enter` to plan and display a Cartesian path, or key-in `q` to exit ...")
    # if not rospy.is_shutdown() and key != 'q':
    #   cartesian_plan, fraction = move_ur5e.plan_cartesian_path(pose_goal)
    # else:
    #   print "Quit on user request ..."
    #   return
    # # key = raw_input("============ Press `Enter` to display a saved trajectory (this will replay the Cartesian path)  ...")
    # # if not rospy.is_shutdown() and key != 'q':
    # #   move_ur5e.display_trajectory(cartesian_plan)

    # key = raw_input("============ Press `Enter` to execute a saved path ..., or key-in `q` to exit ...")
    # if not rospy.is_shutdown() and key != 'q':
    #   move_ur5e.execute_plan(cartesian_plan)
    # else:
    #   print "Quit on user request ..."
    #   return

    key = raw_input("============ Press `Enter` to execute a movement using the pose goal ..., or key-in `q` to exit ...")
    if not rospy.is_shutdown() and key != 'q':
      # move_ur5e.go_to_pose_goal()
      move_ur5e.go_to_pose_goal_with_constraints(pose_goal)
    else:
      print "Quit on user request ..."
      return

    key = raw_input("============ Press `Enter` to execute a movement using the pose goal ..., or key-in `q` to exit ...")
    if not rospy.is_shutdown() and key != 'q':
      pose_goal.position.y -= 0.05
      move_ur5e.go_to_pose_goal_with_constraints(pose_goal)
    else:
      print "Quit on user request ..."
      return
    rospy.spin()

  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return

if __name__ == '__main__':
  main()
