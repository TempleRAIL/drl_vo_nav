/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Template Controller for nav2py
 */

#ifndef NAV2PY_TEMPLATE_CONTROLLER__TEMPLATE_CONTROLLER_HPP_
#define NAV2PY_TEMPLATE_CONTROLLER__TEMPLATE_CONTROLLER_HPP_

#include <string>
#include <vector>
#include <memory>

#include "nav2py/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pluginlib/class_loader.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
 
namespace nav2py_drl_vo_controller
{

class DrlVoController : public nav2py::Controller
{
public:
    DrlVoController() = default;
    ~DrlVoController() override = default;
  
    void configure(
      const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
      std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
      const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;
  
    void cleanup() override;
    void activate() override;
    void deactivate() override;
    void setSpeedLimit(const double & speed_limit, const bool & percentage) override;
  
    geometry_msgs::msg::TwistStamped computeVelocityCommands(
      const geometry_msgs::msg::PoseStamped & pose,
      const geometry_msgs::msg::Twist & velocity,
      nav2_core::GoalChecker * goal_checker) override;
  
    void setPlan(const nav_msgs::msg::Path & path) override;
 
 protected:
    void sendData(
      const geometry_msgs::msg::PoseStamped & pose,
      const geometry_msgs::msg::Twist & velocity, //= geometry_msgs::msg::Twist(),
      const geometry_msgs::msg::PoseStamped & goal_pose);

    nav_msgs::msg::Path transformGlobalPlan(const geometry_msgs::msg::PoseStamped & pose);

    bool transformPose(
      const std::shared_ptr<tf2_ros::Buffer> tf,
      const std::string frame,
      const geometry_msgs::msg::PoseStamped & in_pose,
      geometry_msgs::msg::PoseStamped & out_pose,
      const rclcpp::Duration & transform_tolerance
    ) const;

    // scan subscriber callback:
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void scanHistoryCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    // Timer:
    void timerCallback();
    
    // pure pursit:
    geometry_msgs::msg::Point circleSegmentIntersection(
    const geometry_msgs::msg::Point & p1,
    const geometry_msgs::msg::Point & p2,
    double r);

    geometry_msgs::msg::PoseStamped getLookAheadPoint(
    const double & lookahead_dist,
    const nav_msgs::msg::Path & transformed_plan,
    bool interpolate_after_goal);
  
    rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::string plugin_name_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    rclcpp::Logger logger_ {rclcpp::get_logger("DrlVoController")};
    rclcpp::Clock::SharedPtr clock_;


    double lookahead_dist_;
  
    rclcpp::Duration transform_tolerance_ {0, 0};

    // global plan pub:
    nav_msgs::msg::Path global_plan_;
    std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>> global_pub_;

    // scan sub:
    sensor_msgs::msg::LaserScan scan_msg;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    // scan history pub & sub:
    std_msgs::msg::Float32MultiArray scan_history_msg;
    std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32MultiArray>> scan_history_pub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr scan_history_sub_;

    // Timer object:
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<float> scan_data_;
    std::vector<std::vector<float>> scan_history_;
    
    int ts_cnt = 0;

  };

}  // namespace nav2py_drl_vo_controller

#endif  // NAV2PY_TEMPLATE_CONTROLLER__TEMPLATE_CONTROLLER_HPP_