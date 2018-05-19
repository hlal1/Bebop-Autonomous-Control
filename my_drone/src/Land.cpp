#include <ros/ros.h>
#include "std_msgs/Empty.h"
std_msgs::Empty emp_msg;

int main(int argc, char** argv)
{

	ROS_INFO("Land");
	ros::init(argc, argv,"Land");
    ros::NodeHandle node;
    ros::Rate loop_rate(50);
	ros::Publisher pub_empty;
	pub_empty = node.advertise<std_msgs::Empty>("/bebop/land", 1);

 	while (ros::ok()) 
 				{
				double time_start=(double)ros::Time::now().toSec();
				while ((double)ros::Time::now().toSec()< time_start+5.0)
					{ 
					pub_empty.publish(emp_msg);
					ros::spinOnce();
					loop_rate.sleep();
					}
				ROS_INFO("Landed Successful");
				exit(0);
				}

}


