cmake_minimum_required(VERSION 2.8.3)
project(my_drone)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  message_generation
  genmsg
  roslaunch
)

roslaunch_add_file_check(launch)

# catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/my_python_script
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

add_executable(my_drone_flattrim
  src/Flattrim.cpp
)
add_dependencies(my_drone_flattrim ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(my_drone_flattrim
  ${catkin_LIBRARIES}
)

add_executable(my_drone_takeoff
  src/Takeoff.cpp
)
add_dependencies(my_drone_takeoff ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(my_drone_takeoff
  ${catkin_LIBRARIES}
)

add_executable(my_drone_camcontrol
  src/CamControl.cpp
)
add_dependencies(my_drone_camcontrol ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(my_drone_camcontrol
  ${catkin_LIBRARIES}
)

add_executable(my_drone_altitude
  src/Altitude.cpp
)
add_dependencies(my_drone_altitude ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(my_drone_altitude
  ${catkin_LIBRARIES}
)

add_executable(my_drone_land
  src/Land.cpp
)
add_dependencies(my_drone_land ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(my_drone_land
  ${catkin_LIBRARIES}
)

add_executable(my_drone_position
  src/Position.cpp
)
add_dependencies(my_drone_position ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(my_drone_position
  ${catkin_LIBRARIES}
)

# add_executable(my_drone_height_control
#   src/Height_control.cpp
# )
# add_dependencies(my_drone_height_control ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(my_drone_height_control
#   ${catkin_LIBRARIES}
# )