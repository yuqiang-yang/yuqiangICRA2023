cmake_minimum_required(VERSION 3.1.3)
project(mue_moveit_config)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS scripts/move_ur5e.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
