#include <ros/ros.h>
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovariance.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"

nav_msgs::Odometry alt;
float altx=0,alty=0,altz=0;
geometry_msgs::Twist msg;

//void value(const nav_msgs::Odometry::ConstPtr& msg);
void value(const nav_msgs::Odometry::ConstPtr& msg)
{   
    alt = *msg;
    altx = alt.pose.pose.position.x;
    alty = alt.pose.pose.position.y;
    altz = alt.pose.pose.position.z;
    ROS_INFO("Callback = [%f,%f,%f]",altx,alty,altz);
}

int main(int argc, char** argv)
{
    ROS_INFO("Desired Altitude=3.0");
    ros::init(argc, argv, "Altitude");
    ros::NodeHandle node;
    ros::Subscriber alt_sub;
    alt_sub = node.subscribe<nav_msgs::Odometry>("/bebop/odom", 100, value);
    ros::Publisher altitude_pub;
    altitude_pub = node.advertise<geometry_msgs::Twist>("/bebop/cmd_vel" ,100);

    ros::Rate loop_rate(100);

    while(ros::ok() && node.ok() && abs(altz-3.7) > 0.1)
    {
            msg.linear.x=0;
            msg.linear.y=0;
            msg.linear.z=0.5*(3.7-altz);
            msg.angular.x=0;
            msg.angular.y=0;
            msg.angular.z=0;
            altitude_pub.publish(msg);
        
        ros::spinOnce();
        loop_rate.sleep();
    }
    ROS_INFO("Altitude Reached");
// Take Image.
// Store msg.linear.z in file.
}

