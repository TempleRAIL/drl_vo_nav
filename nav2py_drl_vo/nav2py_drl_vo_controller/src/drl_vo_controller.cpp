/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Template Controller for nav2py
 */

#include <algorithm>
#include <string>
#include <memory>

#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2py_drl_vo_controller/drl_vo_controller.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

using std::hypot;
using std::min;
using std::max;
using std::abs;
using nav2_util::declare_parameter_if_not_declared;
using nav2_util::geometry_utils::euclidean_distance;

#define NUM_TP 10
 
namespace nav2py_drl_vo_controller
{

  /**
    * Find element in iterator with the minimum calculated value
    */
  template<typename Iter, typename Getter>
  Iter min_by(Iter begin, Iter end, Getter getCompareVal)
  {
    if (begin == end) {
      return end;
    }
    auto lowest = getCompareVal(*begin);
    Iter lowest_it = begin;
    for (Iter it = ++begin; it != end; ++it) {
      auto comp = getCompareVal(*it);
      if (comp < lowest) {
        lowest = comp;
        lowest_it = it;
      }
    }
    return lowest_it;
  }

  void DrlVoController::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
  {
    node_ = parent;
  
    auto node = node_.lock();
  
    costmap_ros_ = costmap_ros;
    tf_ = tf;
    plugin_name_ = name;
    logger_ = node->get_logger();
    clock_ = node->get_clock();

    declare_parameter_if_not_declared(
      node, plugin_name_ + ".lookahead_dist",
      rclcpp::ParameterValue(1.0));
    declare_parameter_if_not_declared(
      node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(
        0.1));

    node->get_parameter(plugin_name_ + ".lookahead_dist", lookahead_dist_);
    double transform_tolerance;
    node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance);
    transform_tolerance_ = rclcpp::Duration::from_seconds(transform_tolerance);
    
    // Initialize nav2py
    std::string nav2py_script = ament_index_cpp::get_package_share_directory("nav2py_drl_vo_controller") + "/../../lib/nav2py_drl_vo_controller/nav2py_run" ;
    nav2py_bootstrap(nav2py_script +
      " --host 127.0.0.1" +
      " --port 0");
  
    // Create publisher for global plan
    global_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);
    
    // Create scan subscriber:
    // Subscribe to new topics
    scan_sub_ = node->create_subscription<sensor_msgs::msg::LaserScan>(
        "scan", 20,
        std::bind(&DrlVoController::scanCallback, this, std::placeholders::_1));

    // Create scan history publisher and subscriber:
    scan_history_pub_ = node->create_publisher<std_msgs::msg::Float32MultiArray>("scan_history_list", 1);
    scan_history_sub_ = node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "scan_history_list", 20,
        std::bind(&DrlVoController::scanHistoryCallback, this, std::placeholders::_1));


    // Create a timer to call the callback function every 1/20 seconds
    timer_ = node->create_wall_timer(
        std::chrono::milliseconds(50), 
        std::bind(&DrlVoController::timerCallback, this));


    RCLCPP_INFO(
      logger_,
      "Configured controller: %s of type nav2py_drl_vo_controller::DrlVoController",
      plugin_name_.c_str());
  }
 
  void DrlVoController::sendData(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    const geometry_msgs::msg::PoseStamped & goal_pose)
  {
    static int frame_count = 0;
    frame_count++;

    // Add frame delimiter for logs
    std::string frame_delimiter(50, '=');
    RCLCPP_INFO(logger_, "\n%s", frame_delimiter.c_str());
    RCLCPP_INFO(logger_, "===== SENDING FRAME %d =====", frame_count);

    // Create a structured data message
    std::stringstream ss;
    ss << "frame_info:\n";
    ss << "  id: " << frame_count << "\n";
    ss << "  timestamp: " << clock_->now().nanoseconds() << "\n";

    // Add robot pose
    ss << "robot_pose:\n";
    ss << "  position:\n";
    ss << "    x: " << pose.pose.position.x << "\n";
    ss << "    y: " << pose.pose.position.y << "\n";
    ss << "    z: " << pose.pose.position.z << "\n";
    ss << "  orientation:\n";
    ss << "    x: " << pose.pose.orientation.x << "\n";
    ss << "    y: " << pose.pose.orientation.y << "\n";
    ss << "    z: " << pose.pose.orientation.z << "\n";
    ss << "    w: " << pose.pose.orientation.w << "\n";

    // Add robot velocity
    ss << "robot_velocity:\n";
    ss << "  linear:\n";
    ss << "    x: " << velocity.linear.x << "\n";
    ss << "    y: " << velocity.linear.y << "\n";
    ss << "    z: " << velocity.linear.z << "\n";
    ss << "  angular:\n";
    ss << "    x: " << velocity.angular.x << "\n";
    ss << "    y: " << velocity.angular.y << "\n";
    ss << "    z: " << velocity.angular.z << "\n";

    // Add goal pose
    ss << "goal_pose:\n";
    ss << "  position:\n";
    ss << "    x: " << goal_pose.pose.position.x << "\n";
    ss << "    y: " << goal_pose.pose.position.y << "\n";
    ss << "    z: " << goal_pose.pose.position.z << "\n";
    // ss << "  orientation:\n";
    // ss << "    x: " << goal_pose.pose.orientation.x << "\n";
    // ss << "    y: " << goal_pose.pose.orientation.y << "\n";
    // ss << "    z: " << goal_pose.pose.orientation.z << "\n";
    // ss << "    w: " << goal_pose.pose.orientation.w << "\n";

    // Send the data
    nav2py_send("data", {ss.str()});

    RCLCPP_INFO(logger_, "Data sent to Python side");
  }
 
  void DrlVoController::cleanup()
  {
    RCLCPP_INFO(
      logger_,
      "Cleaning up controller: %s",
      plugin_name_.c_str());
    nav2py_cleanup();
    global_pub_.reset();
    scan_history_pub_.reset();
  }
  
  void DrlVoController::activate()
  {
    RCLCPP_INFO(
      logger_,
      "Activating controller: %s",
      plugin_name_.c_str());
    global_pub_->on_activate();
    scan_history_pub_->on_activate();
  }
  
  void DrlVoController::deactivate()
  {
    RCLCPP_INFO(
      logger_,
      "Deactivating controller: %s",
      plugin_name_.c_str());
    global_pub_->on_deactivate();
    scan_history_pub_->on_deactivate();
  }

  void DrlVoController::setSpeedLimit(const double& speed_limit, const bool& percentage)
  {
    // Empty implementation for interface compatibility
    (void) speed_limit;
    (void) percentage;
  }

  geometry_msgs::msg::TwistStamped DrlVoController::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker)
  {
    (void)goal_checker;

    // send path in robot's local frame:
    auto transformed_plan = transformGlobalPlan(pose);
    try {
      std::string path_yaml = nav_msgs::msg::to_yaml(transformed_plan, true);
      
      nav2py_send("path", {path_yaml});
      RCLCPP_INFO(logger_, "Sent path data to Python controller");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(
        logger_,
        "Error sending path to Python controller: %s", e.what());
    }

    // send lidar scan data:
    try {
      std::string scan_yaml = sensor_msgs::msg::to_yaml(scan_msg, true);
      
      nav2py_send("scan", {scan_yaml});
      RCLCPP_INFO(logger_, "Sent scan data to Python controller");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(
        logger_,
        "Error sending scan to Python controller: %s", e.what());
    }

    // send lidar scan history data:
    try {
      std::string scan_history_yaml = std_msgs::msg::to_yaml(scan_history_msg, true);
      
      nav2py_send("scan_history", {scan_history_yaml});
      RCLCPP_INFO(logger_, "Sent scan history data to Python controller");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(
        logger_,
        "Error sending history scan to Python controller: %s", e.what());
    }

    // Find the first pose which is at a distance greater than the specified lookahed distance
    // auto goal_pose_it = std::find_if(
    //   transformed_plan.poses.begin(), transformed_plan.poses.end(), [&](const auto & ps) {
    //     return hypot(ps.pose.position.x, ps.pose.position.y) >= lookahead_dist_;
    //   });

    // // If the last pose is still within lookahed distance, take the last pose
    // if (goal_pose_it == transformed_plan.poses.end()) {
    //   goal_pose_it = std::prev(transformed_plan.poses.end());
    // }
    // auto goal_pose = goal_pose_it;

    auto goal_pose = getLookAheadPoint(lookahead_dist_, transformed_plan, true);

    
    // Simple implementation that just sends data to Python and waits for response
    try {
      sendData(pose, velocity, goal_pose);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(
        logger_,
        "Error sending data: %s", e.what());
    }
    
    geometry_msgs::msg::TwistStamped cmd_vel;
    cmd_vel.header.frame_id = pose.header.frame_id;
    cmd_vel.header.stamp = clock_->now();
    
    try {
      RCLCPP_INFO(logger_, "Waiting for velocity command from Python...");
      cmd_vel.twist = wait_for_cmd_vel();
      
      RCLCPP_INFO(
        logger_, 
        "Received velocity command: linear_x=%.2f, angular_z=%.2f", 
        cmd_vel.twist.linear.x, cmd_vel.twist.angular.z);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(
        logger_,
        "Error receiving velocity command: %s", e.what());
      
      // Default to stop if there's an error
      cmd_vel.twist.linear.x = 0.0;
      cmd_vel.twist.angular.z = 0.0;
    }

    return cmd_vel;
  }

  void DrlVoController::setPlan(const nav_msgs::msg::Path & path)
  {
    global_plan_ = path;
    global_pub_->publish(path);
  }

  void DrlVoController::scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) 
  {
    // Store the received scan data
    scan_msg = *msg;
    scan_data_ = msg->ranges;
  }

  void DrlVoController::scanHistoryCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) 
  {
    // Store the received scan data
    scan_history_msg = *msg;
  }

  void DrlVoController::timerCallback()
  {
      // Ensure that we have valid data to process
      if (!scan_data_.empty())
      {
          // Stack the scan range data by removing the first and last 20 values
          std::vector<float> stacked_scan(scan_data_.begin() + 20, scan_data_.end() - 20);
          scan_history_.push_back(stacked_scan);

          ts_cnt++;

          // When we have collected NUM_TP scans, process the data
          if (ts_cnt == NUM_TP)
          {   
              std_msgs::msg::Float32MultiArray msg;
              // Flatten the 2D vector into a 1D vector (data field)
              for (const auto& subvector : scan_history_)
              {
                  msg.data.insert(msg.data.end(), subvector.begin(), subvector.end());
              }
              scan_history_pub_->publish(msg);
              // scan_history_msg.data = msg.data;
              // Set the layout dimensions to represent the 2D structure
              // scan_history_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
              // scan_history_msg.layout.dim[0].label = "rows";
              // scan_history_msg.layout.dim[0].size = scan_history_.size();
              // scan_history_msg.layout.dim[0].stride = scan_history_.size() * scan_history_[0].size();  // stride = total size of all elements in the 1D array
              // scan_history_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
              // scan_history_msg.layout.dim[1].label = "cols";
              // scan_history_msg.layout.dim[1].size = scan_history_[0].size();
              
              // Reset the data for the next collection cycle
              ts_cnt = NUM_TP - 1;
              scan_history_.erase(scan_history_.begin());  // Remove the first (oldest) scan
              //scan_history_ = std::vector<std::vector<float>>(NUM_TP - 1); // Keep only the last NUM_TP-1 scans
          }
      }
  }

  geometry_msgs::msg::Point 
  DrlVoController::circleSegmentIntersection(
  const geometry_msgs::msg::Point & p1,
  const geometry_msgs::msg::Point & p2,
  double r)
  {
    // Formula for intersection of a line with a circle centered at the origin,
    // modified to always return the point that is on the segment between the two points.
    // https://mathworld.wolfram.com/Circle-LineIntersection.html
    // This works because the poses are transformed into the robot frame.
    // This can be derived from solving the system of equations of a line and a circle
    // which results in something that is just a reformulation of the quadratic formula.
    // Interactive illustration in doc/circle-segment-intersection.ipynb as well as at
    // https://www.desmos.com/calculator/td5cwbuocd
    double x1 = p1.x;
    double x2 = p2.x;
    double y1 = p1.y;
    double y2 = p2.y;

    double dx = x2 - x1;
    double dy = y2 - y1;
    double dr2 = dx * dx + dy * dy;
    double D = x1 * y2 - x2 * y1;

    // Augmentation to only return point within segment
    double d1 = x1 * x1 + y1 * y1;
    double d2 = x2 * x2 + y2 * y2;
    double dd = d2 - d1;

    geometry_msgs::msg::Point p;
    double sqrt_term = std::sqrt(r * r * dr2 - D * D);
    p.x = (D * dy + std::copysign(1.0, dd) * dx * sqrt_term) / dr2;
    p.y = (-D * dx + std::copysign(1.0, dd) * dy * sqrt_term) / dr2;
    return p;
  }

  geometry_msgs::msg::PoseStamped 
  DrlVoController::getLookAheadPoint(
    const double & lookahead_dist,
    const nav_msgs::msg::Path & transformed_plan,
    bool interpolate_after_goal)
  {
    // Find the first pose which is at a distance greater than the lookahead distance
    auto goal_pose_it = std::find_if(
      transformed_plan.poses.begin(), transformed_plan.poses.end(), [&](const auto & ps) {
        return hypot(ps.pose.position.x, ps.pose.position.y) >= lookahead_dist;
      });

    // If the no pose is not far enough, take the last pose
    if (goal_pose_it == transformed_plan.poses.end()) {
      if (interpolate_after_goal) {
        auto last_pose_it = std::prev(transformed_plan.poses.end());
        auto prev_last_pose_it = std::prev(last_pose_it);

        double end_path_orientation = atan2(
          last_pose_it->pose.position.y - prev_last_pose_it->pose.position.y,
          last_pose_it->pose.position.x - prev_last_pose_it->pose.position.x);

        // Project the last segment out to guarantee it is beyond the look ahead
        // distance
        auto projected_position = last_pose_it->pose.position;
        projected_position.x += cos(end_path_orientation) * lookahead_dist;
        projected_position.y += sin(end_path_orientation) * lookahead_dist;

        // Use the circle intersection to find the position at the correct look
        // ahead distance
        const auto interpolated_position = circleSegmentIntersection(
          last_pose_it->pose.position, projected_position, lookahead_dist);

        geometry_msgs::msg::PoseStamped interpolated_pose;
        interpolated_pose.header = last_pose_it->header;
        interpolated_pose.pose.position = interpolated_position;
        return interpolated_pose;
      } else {
        goal_pose_it = std::prev(transformed_plan.poses.end());
      }
    } else if (goal_pose_it != transformed_plan.poses.begin()) {
      // Find the point on the line segment between the two poses
      // that is exactly the lookahead distance away from the robot pose (the origin)
      // This can be found with a closed form for the intersection of a segment and a circle
      // Because of the way we did the std::find_if, prev_pose is guaranteed to be inside the circle,
      // and goal_pose is guaranteed to be outside the circle.
      auto prev_pose_it = std::prev(goal_pose_it);
      auto point = circleSegmentIntersection(
        prev_pose_it->pose.position,
        goal_pose_it->pose.position, lookahead_dist);
      geometry_msgs::msg::PoseStamped pose;
      pose.header.frame_id = prev_pose_it->header.frame_id;
      pose.header.stamp = goal_pose_it->header.stamp;
      pose.pose.position = point;
      return pose;
    }

    return *goal_pose_it;
  }


  nav_msgs::msg::Path
  DrlVoController::transformGlobalPlan(
    const geometry_msgs::msg::PoseStamped & pose)
  {
    // Original mplementation taken fron nav2_dwb_controller

    if (global_plan_.poses.empty()) {
      throw nav2_core::PlannerException("Received plan with zero length");
    }

    // let's get the pose of the robot in the frame of the plan
    geometry_msgs::msg::PoseStamped robot_pose;
    if (!transformPose(
        tf_, global_plan_.header.frame_id, pose,
        robot_pose, transform_tolerance_))
    {
      throw nav2_core::PlannerException("Unable to transform robot pose into global plan's frame");
    }

    // We'll discard points on the plan that are outside the local costmap
    nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
    double dist_threshold = std::max(costmap->getSizeInCellsX(), costmap->getSizeInCellsY()) *
      costmap->getResolution() / 2.0;

    // First find the closest pose on the path to the robot
    auto transformation_begin =
      min_by(
      global_plan_.poses.begin(), global_plan_.poses.end(),
      [&robot_pose](const geometry_msgs::msg::PoseStamped & ps) {
        return euclidean_distance(robot_pose, ps);
      });

    // From the closest point, look for the first point that's further then dist_threshold from the
    // robot. These points are definitely outside of the costmap so we won't transform them.
    auto transformation_end = std::find_if(
      transformation_begin, end(global_plan_.poses),
      [&](const auto & global_plan_pose) {
        return euclidean_distance(robot_pose, global_plan_pose) > dist_threshold;
      });

    // Helper function for the transform below. Transforms a PoseStamped from global frame to local
    auto transformGlobalPoseToLocal = [&](const auto & global_plan_pose) {
        // We took a copy of the pose, let's lookup the transform at the current time
        geometry_msgs::msg::PoseStamped stamped_pose, transformed_pose;
        stamped_pose.header.frame_id = global_plan_.header.frame_id;
        stamped_pose.header.stamp = pose.header.stamp;
        stamped_pose.pose = global_plan_pose.pose;
        transformPose(
          tf_, costmap_ros_->getBaseFrameID(),
          stamped_pose, transformed_pose, transform_tolerance_);
        return transformed_pose;
      };

    // Transform the near part of the global plan into the robot's frame of reference.
    nav_msgs::msg::Path transformed_plan;
    std::transform(
      transformation_begin, transformation_end,
      std::back_inserter(transformed_plan.poses),
      transformGlobalPoseToLocal);
    transformed_plan.header.frame_id = costmap_ros_->getBaseFrameID();
    transformed_plan.header.stamp = pose.header.stamp;

    // Remove the portion of the global plan that we've already passed so we don't
    // process it on the next iteration (this is called path pruning)
    global_plan_.poses.erase(begin(global_plan_.poses), transformation_begin);
    global_pub_->publish(transformed_plan);

    if (transformed_plan.poses.empty()) {
      throw nav2_core::PlannerException("Resulting plan has 0 poses in it.");
    }

    return transformed_plan;
  }

  bool DrlVoController::transformPose(
    const std::shared_ptr<tf2_ros::Buffer> tf,
    const std::string frame,
    const geometry_msgs::msg::PoseStamped & in_pose,
    geometry_msgs::msg::PoseStamped & out_pose,
    const rclcpp::Duration & transform_tolerance
  ) const
  {
    // Implementation taken as is fron nav_2d_utils in nav2_dwb_controller

    if (in_pose.header.frame_id == frame) {
      out_pose = in_pose;
      return true;
    }

    try {
      tf->transform(in_pose, out_pose, frame);
      return true;
    } catch (tf2::ExtrapolationException & ex) {
      auto transform = tf->lookupTransform(
        frame,
        in_pose.header.frame_id,
        tf2::TimePointZero
      );
      if (
        (rclcpp::Time(in_pose.header.stamp) - rclcpp::Time(transform.header.stamp)) >
        transform_tolerance)
      {
        RCLCPP_ERROR(
          rclcpp::get_logger("tf_help"),
          "Transform data too old when converting from %s to %s",
          in_pose.header.frame_id.c_str(),
          frame.c_str()
        );
        RCLCPP_ERROR(
          rclcpp::get_logger("tf_help"),
          "Data time: %ds %uns, Transform time: %ds %uns",
          in_pose.header.stamp.sec,
          in_pose.header.stamp.nanosec,
          transform.header.stamp.sec,
          transform.header.stamp.nanosec
        );
        return false;
      } else {
        tf2::doTransform(in_pose, out_pose, transform);
        return true;
      }
    } catch (tf2::TransformException & ex) {
      RCLCPP_ERROR(
        rclcpp::get_logger("tf_help"),
        "Exception in transformPose: %s",
        ex.what()
      );
      return false;
    }
    return false;
  }

}  // namespace nav2py_drl_vo_controller

// Register this controller as a nav2_core plugin
PLUGINLIB_EXPORT_CLASS(nav2py_drl_vo_controller::DrlVoController, nav2_core::Controller)