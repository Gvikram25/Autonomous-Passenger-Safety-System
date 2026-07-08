#!/usr/bin/env python2
import socket
import rospy
import pandas as pd
import os
import sys
from std_msgs.msg import String, Bool
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# UDP Configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

# Pickle File Path
PICKLE_PATH = r'/home/infy/AGC_ws/intial_points.pkl'

def get_point_stamped(df, goal_name):
    """
    Retrieves the goal pose from the DataFrame based on the goal name.
    """
    if goal_name not in df['name'].values:
        return None
        
    for ind_goal in df.index:
        if df['name'][ind_goal] == str(goal_name):
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.header.stamp = rospy.Time.now()
            # Extract coordinates
            pos = df['position'][ind_goal]
            orient = df['orientation'][ind_goal]
            
            goal.pose.position.x = pos[0]
            goal.pose.position.y = pos[1]
            goal.pose.position.z = pos[2]
            goal.pose.orientation.x = orient[0]
            goal.pose.orientation.y = orient[1]
            goal.pose.orientation.z = orient[2]
            goal.pose.orientation.w = orient[3]
            return goal
    return None

def publish_navigation_goal(pub_list, pub_name, pub_status, pub_brake, df, goal_name):
    """
    Helper function to publish all topics needed for navigation.
    """
    
    # 1. Release Emergency Brake
    brake_msg = Bool()
    brake_msg.data = False
    pub_brake.publish(brake_msg)
    
    if df is not None:
        pose = get_point_stamped(df, goal_name)
        if pose:
            # Publish Path (Required by Behavioral Planner)
            path_msg = Path()
            path_msg.header.frame_id = 'map'
            path_msg.header.stamp = rospy.Time.now()
            path_msg.poses.append(pose)
            pub_list.publish(path_msg)
            
            # Publish Goal Name
            goal_msg = String()
            goal_msg.data = goal_name
            pub_name.publish(goal_msg)
            
            # Publish Status
            status_msg = String()
            status_msg.data = "TO_DESTINATION"
            pub_status.publish(status_msg)
            
            rospy.loginfo("Navigation started successfully for goal: {}".format(goal_name))
            return True
        else:
            rospy.logerr("Goal '{}' not found in database!".format(goal_name))
            return False
    else:
        rospy.logerr("Destination database not loaded! Cannot navigate.")
        return False

def receiver():
    # Initialize ROS Node
    rospy.init_node('vision_bridge', anonymous=True)
    
    # Initialize Publishers
    pub_status = rospy.Publisher('/gcart_status', String, queue_size=1, latch=True)
    pub_brake = rospy.Publisher('/leg_brake', Bool, queue_size=1, latch=True)
    pub_goal_name = rospy.Publisher('/goal_name', String, queue_size=1, latch=True)
    pub_goal_list = rospy.Publisher('/goal_list', Path, queue_size=1, latch=True)
    
    # Initialize UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # Load Destination Data
    df = None
    try:
        if os.path.exists(PICKLE_PATH):
             df = pd.read_pickle(PICKLE_PATH)
             rospy.loginfo("Loaded destinations from {}".format(PICKLE_PATH))
             rospy.loginfo("Available Goals: {}".format(list(df['name'])))
        else:
             rospy.logwarn("Could not find intial_points.pkl at {}".format(PICKLE_PATH))
    except Exception as e:
        rospy.logerr("Error loading pickle file: {}".format(e))

    # Track state to avoid spamming topics
    last_state = None 
    last_target = None

    rospy.loginfo("Vision Bridge Started. Listening on UDP port {}".format(UDP_PORT))
    
    while not rospy.is_shutdown():
        try:
            # Receive Data
            data, addr = sock.recvfrom(1024)
            message = data.decode().strip()
            
            target_goal = None
            current_state_key = None # Used for 'last_state' tracking

            # Determine Target based on Message
            if "Final State: false" in message:
                # ABNORMAL -> VINAY SADAN
                target_goal = "vinay_sadan"
                current_state_key = "FALSE"
                
            elif "Final State: true" in message:
                # NORMAL -> SB EXITING
                target_goal = "sb_exiting_temp2"
                current_state_key = "TRUE"
                
            # Execute Logic if state changed OR target changed
            if target_goal and (current_state_key != last_state or target_goal != last_target):
                
                rospy.loginfo("State Change Detected: {} -> Target: {}".format(current_state_key, target_goal))
                
                success = publish_navigation_goal(pub_goal_list, pub_goal_name, pub_status, pub_brake, df, target_goal)
                
                if success:
                    last_state = current_state_key
                    last_target = target_goal
                else:
                    # If failed (e.g., goal not found), allow retrying on next loop 
                    # by NOT updating last_state, or handle differently?
                    # For now, we log error and don't update state to prevent spam if it's permanent error,
                    # but maybe we should allow retries.
                    # Let's NOT update last_state so it retries if data comes again? 
                    # Actually, better to update partially or just log.
                    pass

        except socket.error as e:
            rospy.logerr("Socket Error: {}".format(e))
            rospy.sleep(1) # Prevent busy loop on socket error
        except Exception as e:
            rospy.logerr("Error: {}".format(e))

if __name__ == '__main__':
    try:
        receiver()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print("Detailed Error: ",e)
