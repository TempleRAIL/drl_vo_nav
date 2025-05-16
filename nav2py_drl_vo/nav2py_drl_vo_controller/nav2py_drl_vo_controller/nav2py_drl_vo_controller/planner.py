import sys
import os
import numpy as np
import numpy.matlib
from stable_baselines3 import PPO

from geometry_msgs.msg import Twist

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from custom_cnn_full import CustomCNN

# from nav2py.interfaces import Controller

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# for reproducibility, we seed the rng
#       
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)


class DrlVoPlanner():
    def __init__(self):
        # super().__init__()
        # Initialize your planner's parameters
        self.vx_limit = None
        self.wz_limit = None
        self.range_limit = None

        self.vx = 0.
        self.wz = 0.
        self.model = None

        # load model:
        if(self.model == None):
            package_share_dir = os.path.dirname(__file__)
            model_path = os.path.join(package_share_dir, '../../../..', 'share', 'model')
            model_file = os.path.join(model_path, 'drl_vo.zip')
            self.model = PPO.load(model_file, device='cpu')

        print("Finish loading model.")
        
    def configure(self, range_limit=30., vx_limit=0.5, wz_limit=0.7):
        """Configure your planner with parameters"""
        self.range_limit = range_limit
        self.vx_limit = vx_limit
        self.wz_limit = wz_limit
        
    def compute_velocity_commands(self, scan_history, current_scan, sub_goal):
        """
        Implement your navigation algorithm here
        
        Args:
            current_pose: Current robot pose (geometry_msgs/Pose)
            current_velocity: Current robot velocity (geometry_msgs/Twist)
            goal_pose: Goal pose (geometry_msgs/Pose)
            
        Returns:
            velocity_command: Desired velocity command (geometry_msgs/Twist)
        """
        # Your navigation logic here
        # minimum distance:
        len_scan = len(current_scan)
        scan = np.array(current_scan[int(len_scan/2-len_scan/9):int(len_scan/2+len_scan/9)])# [360-40:-360+40])
        scan = scan[scan!=0]
        if(scan.size != 0):
            min_scan_dist = np.amin(scan)
        else:
            min_scan_dist = 30.

        cmd_vel = Twist()
        # if the goal is close to the robot:
        # if np.linalg.norm(sub_goal) <= 0.9: 
        #     cmd_vel.linear.x = 0.
        #     cmd_vel.angular.z = 0.
        if min_scan_dist <= 0.45: 
            cmd_vel.linear.x = 0.
            cmd_vel.angular.z = self.wz_limit 
        else:
            # MaxAbsScaler:
            # v_min = -2 
            # v_max = 2 
            # ped_map = np.array(ped_map, dtype=np.float32)
            # ped_map = 2 * (ped_map - v_min) / (v_max - v_min) + (-1)
            ped_map = np.zeros(12800)

            # MaxAbsScaler:
            temp = np.array(scan_history, dtype=np.float32)
            scan_avg = np.zeros((20,80))
            for n in range(10):
                scan_tmp = temp[n*len_scan:(n+1)*len_scan] #temp[n*720:(n+1)*720]
                # scan_tmp = scan_tmp[int(len_scan/2 - len_scan/4):int(len_scan/2 + len_scan/4)]
                if(len(scan_tmp) > 0):
                    for i in range(80):
                        step = int(len_scan / 80)
                        scan_avg[2*n, i] = np.min(scan_tmp[i*step:(i+1)*step]) #np.min(scan_tmp[i*9:(i+1)*9])
                        scan_avg[2*n+1, i] = np.mean(scan_tmp[i*step:(i+1)*step]) #np.mean(scan_tmp[i*9:(i+1)*9])
            
            scan_avg = scan_avg.reshape(1600)
            scan_avg_map = np.matlib.repmat(scan_avg,1,4)
            scan_map = scan_avg_map.reshape(6400)
            s_min = 0.
            s_max = self.range_limit #30.
            scan_map = 2 * (scan_map - s_min) / (s_max - s_min) + (-1)
            
            # goal:
            # MaxAbsScaler:
            g_min = -1.
            g_max = 1.
            goal_orignal = np.array(sub_goal, dtype=np.float32)
            sub_goal = 2 * (goal_orignal - g_min) / (g_max - g_min) + (-1)
            #sub_goal = sub_goal.tolist()

            # observation:
            observation = np.concatenate((ped_map, scan_map, sub_goal), axis=None) 

            #self.inference()
            action, _states = self.model.predict(observation, deterministic=True)

            # calculate the goal velocity of the robot and send the command
            # velocities:
            vx_min = 0
            if(self.vx_limit >= 0.75):
                if(min_scan_dist >= 2.5):  # free space margin
                    vx_max = self.vx_limit #0.75
                else:
                    vx_max = self.vx_limit / 1.5 #0.5 
            else:
                vx_max = self.vx_limit
            
            if(self.wz_limit >= 1):
                vz_min = -self.wz_limit / 1.5 
                vz_max = self.wz_limit  / 1.5 
            else:
                vz_min = -self.wz_limit #-0.7 
                vz_max = self.wz_limit #0.7

            # MaxAbsScaler inverse:
            cmd_vel.linear.x = (action[0] + 1) * (vx_max - vx_min) / 2. + vx_min
            cmd_vel.angular.z = (action[1] + 1) * (vz_max - vz_min) / 2. + vz_min

        velocity_command = cmd_vel
        return velocity_command
