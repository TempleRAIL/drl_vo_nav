import yaml
import numpy as np
import nav2py
import nav2py.interfaces
import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger

from .planner import DrlVoPlanner

NUM_TP = 10

class nav2py_drl_vo_controller(nav2py.interfaces.nav2py_costmap_controller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

         # Create and configure your planner
        print("Load DRLVO")
        self.planner = DrlVoPlanner()
        self.rang_limit = 30.
        self.planner.configure(range_limit=self.rang_limit, vx_limit=0.26)

        # Register callbacks
        self._register_callback('data', self._data_callback)
        self._register_callback('path', self._path_callback)
        self._register_callback('scan', self._scan_callback)
        self._register_callback('scan_history', self._scan_history_callback)
        
        self.logger = get_logger('nav2py_drl_vo_controller')
        self.frame_count = 0
        self.path = None
        self.scan_history = np.zeros(3200)
        self.scan_tmp = np.zeros(360)
        self.sub_goal = np.zeros(2)

        # DRL-VO input:
        # self.scan_input = None
        # self.goal_input = None

        self.logger.info("nav2py_drl_vo_controller initialized")

    def _path_callback(self, path_):
        """
        Process path data from C++ controller
        """
        try:
            if isinstance(path_, list) and len(path_) > 0:
                data_str = path_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                self.path = yaml.safe_load(data_str)
                self.logger.info("Received path data")
                
                # You could extract the goal position from the path if needed
                if self.path and 'poses' in self.path and len(self.path['poses']) > 0:
                    last_pose = self.path['poses'][-1]['pose']
                    goal_x = last_pose['position']['x']
                    goal_y = last_pose['position']['y']
                    self.logger.info(f"Goal position: x={goal_x:.2f}, y={goal_y:.2f}")

        except Exception as e:
            import traceback
            self.logger.error(f"Error processing path data: {e}")
            self.logger.error(traceback.format_exc())
    
    def _scan_callback(self, scan_):
        """
        Process path data from C++ controller
        """
        try:
            if isinstance(scan_, list) and len(scan_) > 0:
                data_str = scan_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                scan = yaml.safe_load(data_str)
                scan_data = np.array(scan['ranges'], dtype=np.float32)
                scan_data[np.isnan(scan_data)] = self.rang_limit 
                scan_data[np.isinf(scan_data)] = self.rang_limit 
                self.scan_tmp = scan_data
                self.logger.info("Received scan data")
        
        except Exception as e:
            import traceback
            self.logger.error(f"Error processing scan data: {e}")
            self.logger.error(traceback.format_exc())

    def _scan_history_callback(self, scan_history_):
        """
        Process path data from C++ controller
        """
        try:
            if isinstance(scan_history_, list) and len(scan_history_) > 0:
                data_str = scan_history_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                scan_history = yaml.safe_load(data_str)
                scan_data = np.array(scan_history['data'], dtype=np.float32)
                scan_data[np.isnan(scan_data)] = self.rang_limit 
                scan_data[np.isinf(scan_data)] = self.rang_limit 
                self.scan_history = scan_data #.tolist()
                self.logger.info("Received scan history data")
                # self.logger.info(f"Scan data size: {scan_data.size}")
                # self.logger.info(f"First 20 elements of scan data: {scan_data[:min(20, scan_data.size)]}")

                # Use your planner to compute velocity commands
                velocity_command = self.planner.compute_velocity_commands(
                    scan_history = self.scan_history,
                    current_scan = self.scan_tmp,
                    sub_goal = self.sub_goal
                )
                
                # Send velocity commands
                self._send_cmd_vel(
                    velocity_command.linear.x,
                    velocity_command.angular.z
                )
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error processing scan history data: {e}")
            self.logger.error(traceback.format_exc())

            # Send a safe stop command in case of error
            self._send_cmd_vel(0.0, 0.0)

    def _data_callback(self, data):
        """
        Process data from C++ controller
        """
        try:
            self.frame_count += 1
            
            # Simple frame delimiter for logging
            frame_delimiter = "=" * 50
            self.logger.info(f"\n{frame_delimiter}")
            self.logger.info(f"PROCESSING FRAME {self.frame_count}")
            
            # Parse the incoming data
            if isinstance(data, list) and len(data) > 0:
                data_str = data[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                parsed_data = yaml.safe_load(data_str)
                self.logger.info("Data decoded successfully")
            else:
                if isinstance(data, bytes):
                    parsed_data = yaml.safe_load(data.decode())
                    self.logger.info("Data decoded from bytes")
                else:
                    self.logger.error(f"Unexpected data type: {type(data)}")
                    # Send a stop command if we can't parse the data
                    self._send_cmd_vel(0.0, 0.0)
                    return
            
            # Extract frame info
            frame_info = parsed_data.get('frame_info', {})
            frame_id = frame_info.get('id', 0)
            timestamp = frame_info.get('timestamp', 0)
            self.logger.info(f"Frame ID: {frame_id}, Timestamp: {timestamp}")
            
            # Extract robot pose
            robot_pose = parsed_data.get('robot_pose', {})
            position = robot_pose.get('position', {})
            x = position.get('x', 0.0)
            y = position.get('y', 0.0)
            self.logger.info(f"Robot position: x={x:.2f}, y={y:.2f}")
            
            # Extract velocity if available
            velocity = parsed_data.get('robot_velocity', {})
            linear_x = velocity.get('linear', {}).get('x', 0.0)
            angular_z = velocity.get('angular', {}).get('z', 0.0)
            self.logger.info(f"Current velocity: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")
            
            # Extract goal pose
            goal_pose = parsed_data.get('goal_pose', {})
            goal_position = goal_pose.get('position', {})
            gx = goal_position.get('x', 0.0)
            gy = goal_position.get('y', 0.0)
            self.sub_goal = np.array([gx, gy])
            self.logger.info(f"Goal position: x={gx:.2f}, y={gy:.2f}")
            
            # Add closing delimiter
            self.logger.info(f"FRAME {self.frame_count} COMPLETED")
            self.logger.info(f"{frame_delimiter}")
                
        except Exception as e:
            import traceback
            self.logger.error(f"Error processing data: {e}")
            self.logger.error(traceback.format_exc())
            
            # Send a safe stop command in case of error
            self._send_cmd_vel(0.0, 0.0)
    
if __name__ == "__main__":
    nav2py.main(nav2py_drl_vo_controller)
    