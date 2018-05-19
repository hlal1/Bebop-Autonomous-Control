#include <ros/ros.h>
#include "geometry_msgs/Twist.h"
geometry_msgs::Twist msg;

int main(int argc, char** argv)
{

	ROS_INFO("CamControl");
	ros::init(argc, argv,"CamControl");
    ros::NodeHandle node;
    ros::Rate loop_rate(50);
	ros::Publisher pub_campos;
	pub_campos = node.advertise<geometry_msgs::Twist>("/bebop/camera_control", 1);

 	while (ros::ok()) 
 				{
				msg.linear.x=0;
                msg.linear.y=0;
                msg.linear.z=0;
                msg.angular.x=0;
                msg.angular.y=-80.0;
                msg.angular.z=0;
                pub_campos.publish(msg);
            
                ros::spinOnce();
                loop_rate.sleep();
				}

}