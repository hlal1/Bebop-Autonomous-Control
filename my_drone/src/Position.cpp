#include <ros/ros.h>
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovariance.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

nav_msgs::Odometry pos;
float posx = 0,posy = 0,posz = 0;
geometry_msgs::Twist msg;

void value(const nav_msgs::Odometry::ConstPtr& msg);
void getCoords();

float coordx=0;
float coordy=0;
float coordz=0;

// Get Coordinates.
void getCoords()
{
    ifstream in;
    in.open("Drone_data/coords.csv", ios::app);
    if (!in.is_open()) {"File not open.\n";}
    else
    {
    char delimiter=',';
    in >> coordx >> delimiter >> coordy >> delimiter >> coordz;
    printf("[%f,%f,%f]\n",coordx,coordy,coordz);
    }
in.close();
}

void value(const nav_msgs::Odometry::ConstPtr& msg)
{   
    pos = *msg;
    posx = pos.pose.pose.position.x;
    posy = pos.pose.pose.position.y;
    posz = pos.pose.pose.position.z;
    ROS_INFO("Callback = [%f,%f,%f]",posx,posy,posz);
}

int main(int argc, char** argv)
{   
    ROS_INFO("Desired Position");
    ros::init(argc, argv, "Position");
    getCoords();
    ros::NodeHandle node;
    ros::Subscriber pos_sub;
    pos_sub = node.subscribe<nav_msgs::Odometry>("/bebop/odom", 100, value);
    ros::Publisher altitude_pub;
    altitude_pub = node.advertise<geometry_msgs::Twist>("/bebop/cmd_vel" ,100);

    ros::Rate loop_rate(100);

    while(ros::ok() && node.ok() && abs(posz-coordz) > 0.2 && abs(posx-coordx) > 0.1 && abs(posy-coordy) > 0.1)
    {
            msg.linear.x=0.1*(coordx+0.2-posx);
            msg.linear.y=0.1*(coordy+0.2-posy);
            msg.linear.z=0.5*(coordz+0.2-posz);
            msg.angular.x=0;
            msg.angular.y=0;
            msg.angular.z=0;
            altitude_pub.publish(msg);
        
        ros::spinOnce();
        loop_rate.sleep();
    }
// Take Image.
// Store msg.linear.z in file.
    ofstream myFile;
    myFile.open("Drone_data/alt.csv");
    myFile << posz;
    myFile.close();
}
