/**
 * @file /src/velocity_smoother_nodelet.cpp
 *
 * @brief Velocity smoother implementation.
 *
 * License: BSD
 *   https://raw.github.com/yujinrobot/yujin_ocs/hydro/yocs_velocity_smoother/LICENSE
 **/
/*****************************************************************************
 ** Includes
 *****************************************************************************/

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <dynamic_reconfigure/server.h>
#include <yocs_velocity_smoother/paramsConfig.h>

#include <ecl/threads/thread.hpp>

#include "yocs_velocity_smoother/velocity_smoother_nodelet.hpp"

/*****************************************************************************
 ** Preprocessing
 *****************************************************************************/

#define PERIOD_RECORD_SIZE    5
#define ZERO_VEL_COMMAND      geometry_msgs::Twist();
#define IS_ZERO_VEOCITY(a)   ((a.linear.x == 0.0) && (a.angular.z == 0.0))

/*****************************************************************************
** Namespaces
*****************************************************************************/

namespace yocs_velocity_smoother {

/*********************
** Implementation
**********************/

VelocitySmoother::VelocitySmoother(const std::string &name)
: name(name)
, quiet(false)
, shutdown_req(false)
, input_active(false)
, pr_next(0)
, dynamic_reconfigure_server(NULL)
{
};

void VelocitySmoother::reconfigCB(yocs_velocity_smoother::paramsConfig &config, uint32_t level)
{
  ROS_INFO("Reconfigure request : %f %f %f %f %f",
           config.speed_lim_v, config.speed_lim_w, config.accel_lim_v, config.accel_lim_w, config.decel_factor);

  locker.lock();
  speed_lim_v  = config.speed_lim_v;
  speed_lim_w  = config.speed_lim_w;
  accel_lim_v  = config.accel_lim_v;
  accel_lim_w  = config.accel_lim_w;
  decel_factor = config.decel_factor;
  decel_lim_v  = decel_factor*accel_lim_v;
  decel_lim_w  = decel_factor*accel_lim_w;
  locker.unlock();
}

void VelocitySmoother::velocityCB(const geometry_msgs::Twist::ConstPtr& msg)
{
  // Estimate commands frequency; we do continuously as it can be very different depending on the
  // publisher type, and we don't want to impose extra constraints to keep this package flexible
  if (period_record.size() < PERIOD_RECORD_SIZE)
  {
    period_record.push_back((ros::Time::now() - last_cb_time).toSec());
  }
  else
  {
    period_record[pr_next] = (ros::Time::now() - last_cb_time).toSec();
  }

  pr_next++;
  pr_next %= period_record.size();
  last_cb_time = ros::Time::now();

  if (period_record.size() <= PERIOD_RECORD_SIZE/2)
  {
    // wait until we have some values; make a reasonable assumption (10 Hz) meanwhile
    cb_avg_time = 0.1;
  }
  else
  {
    // enough; recalculate with the latest input
    cb_avg_time = median(period_record);
  }

  input_active = true;

  // Bound speed with the maximum values
  locker.lock();
  target_vel.linear.x  =
      msg->linear.x  > 0.0 ? std::min(msg->linear.x,  speed_lim_v) : std::max(msg->linear.x,  -speed_lim_v);
  target_vel.angular.z =
      msg->angular.z > 0.0 ? std::min(msg->angular.z, speed_lim_w) : std::max(msg->angular.z, -speed_lim_w);
  locker.unlock();
}

void VelocitySmoother::odometryCB(const nav_msgs::Odometry::ConstPtr& msg)
{
  if (robot_feedback == ODOMETRY)
    current_vel = msg->twist.twist;

  // ignore otherwise
}

void VelocitySmoother::robotVelCB(const geometry_msgs::Twist::ConstPtr& msg)
{
  if (robot_feedback == COMMANDS)
    current_vel = *msg;

  // ignore otherwise
}

void VelocitySmoother::spin()
{
  double period = 1.0/frequency;
  ros::Rate spin_rate(frequency);

  while (! shutdown_req && ros::ok())
  {
    locker.lock();
    double accel_lim_v_(accel_lim_v);
    double accel_lim_w_(accel_lim_w);
    double decel_factor(decel_factor);
    double decel_lim_v_(decel_lim_v);
    double decel_lim_w_(decel_lim_w);
    locker.unlock();
    
    if ((input_active == true) && (cb_avg_time > 0.0) &&
        ((ros::Time::now() - last_cb_time).toSec() > std::min(3.0*cb_avg_time, 0.5)))
    {
      // Velocity input no active anymore; normally last command is a zero-velocity one, but reassure
      // this, just in case something went wrong with our input, or he just forgot good manners...
      // Issue #2, extra check in case cb_avg_time is very big, for example with several atomic commands
      // The cb_avg_time > 0 check is required to deal with low-rate simulated time, that can make that
      // several messages arrive with the same time and so lead to a zero median
      input_active = false;
      if (IS_ZERO_VEOCITY(target_vel) == false)
      {
        ROS_WARN_STREAM("Velocity Smoother : input got inactive leaving us a non-zero target velocity ("
              << target_vel.linear.x << ", " << target_vel.angular.z << "), zeroing...[" << name << "]");
        target_vel = ZERO_VEL_COMMAND;
      }
    }

    //check if the feedback is off from what we expect
    //don't care about min / max velocities here, just for rough checking
    double period_buffer = 2.0;

    double v_deviation_lower_bound = last_cmd_vel.linear.x - decel_lim_v_ * period * period_buffer;
    double v_deviation_upper_bound = last_cmd_vel.linear.x + accel_lim_v_ * period * period_buffer;

    double w_deviation_lower_bound = last_cmd_vel.angular.z - decel_lim_w_ * period * period_buffer;
    double angular_max_deviation = last_cmd_vel.angular.z + accel_lim_w_ * period * period_buffer;

    bool v_different_from_feedback = current_vel.linear.x < v_deviation_lower_bound || current_vel.linear.x > v_deviation_upper_bound;
    bool w_different_from_feedback = current_vel.angular.z < w_deviation_lower_bound || current_vel.angular.z > angular_max_deviation;

    if ((robot_feedback != NONE) && (input_active == true) && (cb_avg_time > 0.0) &&
        (((ros::Time::now() - last_cb_time).toSec() > 5.0*cb_avg_time)     || // 5 missing msgs
            v_different_from_feedback || w_different_from_feedback))
    {
      // If the publisher has been inactive for a while, or if our current commanding differs a lot
      // from robot velocity feedback, we cannot trust the former; relay on robot's feedback instead
      // This might not work super well using the odometry if it has a high delay
      if ( !quiet ) {
        // this condition can be unavoidable due to preemption of current velocity control on
        // velocity multiplexer so be quiet if we're instructed to do so
        ROS_WARN_STREAM("Velocity Smoother : using robot velocity feedback " <<
                        std::string(robot_feedback == ODOMETRY ? "odometry" : "end commands") <<
                        " instead of last command: " <<
                        (ros::Time::now() - last_cb_time).toSec() << ", " <<
                        current_vel.linear.x  - last_cmd_vel.linear.x << ", " <<
                        current_vel.angular.z - last_cmd_vel.angular.z << ", [" << name << "]"
                        );
      }
      last_cmd_vel = current_vel;
    }

    geometry_msgs::TwistPtr cmd_vel;

    if ((target_vel.linear.x  != last_cmd_vel.linear.x) ||
        (target_vel.angular.z != last_cmd_vel.angular.z))
    {
      // Try to reach target velocity ensuring that we don't exceed the acceleration limits
      cmd_vel.reset(new geometry_msgs::Twist(target_vel));

      double v_inc, w_inc, max_v_inc, max_w_inc;

      v_inc = target_vel.linear.x - last_cmd_vel.linear.x;
      if ((robot_feedback == ODOMETRY) && (current_vel.linear.x*target_vel.linear.x < 0.0))
      {
        // countermarch (on robots with significant inertia; requires odometry feedback to be detected)
        max_v_inc = decel_lim_v_*period;
      }
      else
      {
        max_v_inc = ((v_inc*target_vel.linear.x > 0.0)?accel_lim_v:decel_lim_v_)*period;
      }

      w_inc = target_vel.angular.z - last_cmd_vel.angular.z;
      if ((robot_feedback == ODOMETRY) && (current_vel.angular.z*target_vel.angular.z < 0.0))
      {
        // countermarch (on robots with significant inertia; requires odometry feedback to be detected)
        max_w_inc = decel_lim_w_*period;
      }
      else
      {
        max_w_inc = ((w_inc*target_vel.angular.z > 0.0)?accel_lim_w_:decel_lim_w_)*period;
      }

      // Calculate and normalise vectors A (desired velocity increment) and B (maximum velocity increment),
      // where v acts as coordinate x and w as coordinate y; the sign of the angle from A to B determines
      // which velocity (v or w) must be overconstrained to keep the direction provided as command
      double MA = sqrt(    v_inc *     v_inc +     w_inc *     w_inc);
      double MB = sqrt(max_v_inc * max_v_inc + max_w_inc * max_w_inc);

      double Av = std::abs(v_inc) / MA;
      double Aw = std::abs(w_inc) / MA;
      double Bv = max_v_inc / MB;
      double Bw = max_w_inc / MB;
      double theta = atan2(Bw, Bv) - atan2(Aw, Av);

      if (theta < 0)
      {
        // overconstrain linear velocity
        max_v_inc = (max_w_inc*std::abs(v_inc))/std::abs(w_inc);
      }
      else
      {
        // overconstrain angular velocity
        max_w_inc = (max_v_inc*std::abs(w_inc))/std::abs(v_inc);
      }

      if (std::abs(v_inc) > max_v_inc)
      {
        // we must limit linear velocity
        cmd_vel->linear.x  = last_cmd_vel.linear.x  + sign(v_inc)*max_v_inc;
      }

      if (std::abs(w_inc) > max_w_inc)
      {
        // we must limit angular velocity
        cmd_vel->angular.z = last_cmd_vel.angular.z + sign(w_inc)*max_w_inc;
      }
      smooth_vel_pub.publish(cmd_vel);
      last_cmd_vel = *cmd_vel;
    }
    else if (input_active == true)
    {
      // We already reached target velocity; just keep resending last command while input is active
      cmd_vel.reset(new geometry_msgs::Twist(last_cmd_vel));
      smooth_vel_pub.publish(cmd_vel);
    }

    spin_rate.sleep();
  }
}

/**
 * Initialise from a nodelet's private nodehandle.
 * @param nh : private nodehandle
 * @return bool : success or failure
 */
bool VelocitySmoother::init(ros::NodeHandle& nh)
{
  // Dynamic Reconfigure
  dynamic_reconfigure_callback = boost::bind(&VelocitySmoother::reconfigCB, this, _1, _2);

  dynamic_reconfigure_server = new dynamic_reconfigure::Server<yocs_velocity_smoother::paramsConfig>(nh);
  dynamic_reconfigure_server->setCallback(dynamic_reconfigure_callback);

  // Optional parameters
  int feedback;
  nh.param("frequency",      frequency,     20.0);
  nh.param("quiet",          quiet,         quiet);
  nh.param("decel_factor",   decel_factor,   1.0);
  nh.param("robot_feedback", feedback, (int)NONE);

  if ((int(feedback) < NONE) || (int(feedback) > COMMANDS))
  {
    ROS_WARN("Invalid robot feedback type (%d). Valid options are 0 (NONE, default), 1 (ODOMETRY) and 2 (COMMANDS)",
             feedback);
    feedback = NONE;
  }

  robot_feedback = static_cast<RobotFeedbackType>(feedback);

  // Mandatory parameters
  if ((nh.getParam("speed_lim_v", speed_lim_v) == false) ||
      (nh.getParam("speed_lim_w", speed_lim_w) == false))
  {
    ROS_ERROR("Missing velocity limit parameter(s)");
    return false;
  }

  if ((nh.getParam("accel_lim_v", accel_lim_v) == false) ||
      (nh.getParam("accel_lim_w", accel_lim_w) == false))
  {
    ROS_ERROR("Missing acceleration limit parameter(s)");
    return false;
  }

  // Deceleration can be more aggressive, if necessary
  decel_lim_v = decel_factor*accel_lim_v;
  decel_lim_w = decel_factor*accel_lim_w;

  // Publishers and subscribers
  odometry_sub    = nh.subscribe("odometry",      1, &VelocitySmoother::odometryCB, this);
  current_vel_sub = nh.subscribe("robot_cmd_vel", 1, &VelocitySmoother::robotVelCB, this);
  raw_in_vel_sub  = nh.subscribe("raw_cmd_vel",   1, &VelocitySmoother::velocityCB, this);
  smooth_vel_pub  = nh.advertise <geometry_msgs::Twist> ("smooth_cmd_vel", 1);

  return true;
}


/*********************
** Nodelet
**********************/

class VelocitySmootherNodelet : public nodelet::Nodelet
{
public:
  VelocitySmootherNodelet()  { }
  ~VelocitySmootherNodelet()
  {
    NODELET_DEBUG("Velocity Smoother : waiting for worker thread to finish...");
    vel_smoother_->shutdown();
    worker_thread_.join();
  }

  std::string unresolvedName(const std::string &name) const {
    size_t pos = name.find_last_of('/');
    return name.substr(pos + 1);
  }


  virtual void onInit()
  {
    ros::NodeHandle ph = getPrivateNodeHandle();
    std::string resolved_name = ph.getUnresolvedNamespace(); // this always returns like /robosem/goo_arm - why not unresolved?
    std::string name = unresolvedName(resolved_name); // unresolve it ourselves
    NODELET_DEBUG_STREAM("Velocity Smoother : initialising nodelet...[" << name << "]");
    vel_smoother_.reset(new VelocitySmoother(name));
    if (vel_smoother_->init(ph))
    {
      NODELET_DEBUG_STREAM("Velocity Smoother : nodelet initialised [" << name << "]");
      worker_thread_.start(&VelocitySmoother::spin, *vel_smoother_);
    }
    else
    {
      NODELET_ERROR_STREAM("Velocity Smoother : nodelet initialisation failed [" << name << "]");
    }
  }

private:
  boost::shared_ptr<VelocitySmoother> vel_smoother_;
  ecl::Thread                        worker_thread_;
};

} // namespace yocs_velocity_smoother

PLUGINLIB_EXPORT_CLASS(yocs_velocity_smoother::VelocitySmootherNodelet, nodelet::Nodelet);
