#ifndef UTIL_H_
#define UTIL_H_

#include "evio.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <deque>
#include <geometry_msgs/PoseStamped.h>
#include <mutex>
#include <prophesee_event_msgs/Event.h>
#include <prophesee_event_msgs/EventArray.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

struct GTData
{
  using Ptr = std::shared_ptr<GTData>;
  Eigen::Vector3d t;
  Eigen::Quaterniond q;
  double time_stamp_;

  GTData() = delete;
  GTData(const geometry_msgs::PoseStampedConstPtr& gt_masage,
         const double offset) {
    t.x()       = gt_masage->pose.position.x;
    t.y()       = gt_masage->pose.position.y;
    t.z()       = gt_masage->pose.position.z;
    q.w()       = gt_masage->pose.orientation.w;
    q.x()       = gt_masage->pose.orientation.x;
    q.y()       = gt_masage->pose.orientation.y;
    q.z()       = gt_masage->pose.orientation.z;
    time_stamp_ = static_cast<double>(gt_masage->header.stamp.nsec) * 1e-9 +
                  gt_masage->header.stamp.sec + offset;
  }
};

struct ImuData
{
  using Ptr = std::shared_ptr<ImuData>;
  Eigen::Vector3d angular_velocity_;
  Eigen::Vector3d linear_acceleration_;
  double time_stamp_;

  ImuData() = delete;
  ImuData(const sensor_msgs::ImuConstPtr& imu_masage, const double offset) {
    time_stamp_ = static_cast<double>(imu_masage->header.stamp.nsec) * 1e-9 +
                  imu_masage->header.stamp.sec + offset;
    angular_velocity_    = {imu_masage->angular_velocity.x,
                         imu_masage->angular_velocity.y,
                         imu_masage->angular_velocity.z};
    linear_acceleration_ = {imu_masage->linear_acceleration.x,
                            imu_masage->linear_acceleration.y,
                            imu_masage->linear_acceleration.z};
  }
  ImuData(const double time_stamp,
          const Eigen::Vector3d angular_velocity,
          const Eigen::Vector3d linear_acceleration)
      : time_stamp_(time_stamp), angular_velocity_(angular_velocity),
        linear_acceleration_(linear_acceleration) {}
};

struct EventRaw
{
  double time_stemp_;
  uint16_t x_;
  uint16_t y_;
  bool polarity_;

  EventRaw() = delete;
  EventRaw(const double ts, const uint16_t x, const uint16_t y, const bool p)
      : time_stemp_(ts), x_(x), y_(y), polarity_(p){};
};

struct EventProcessed
{
  EventProcessed() = default;
  EventProcessed(const double ts, const float x, const float y, const bool p)
      : time_stemp_(ts), x_(x), y_(y), polarity_(p){};
  double time_stemp_;
  float x_;
  float y_;
  bool polarity_;
};

struct MapWarp
{
  float x_;
  float y_;
  Eigen::Matrix<float, 2, 3> j_rot_;
  Eigen::Matrix<float, 2, 3> j_trans_;
};

template <typename T> struct EventKMs
{
  using Ptr = std::shared_ptr<EventKMs<T>>;
  double time_start_;
  double time_end_;
  bool is_finished_ = false;
  typename std::vector<T> event_vector_;

  EventKMs() = delete;
  EventKMs(const double time_start, const T& event) {
    time_start_ = time_start;
    time_end_   = time_start + ENVELOPE_K * 1e-3;
    event_vector_.emplace_back(event);
  }
  // empty envelope case
  EventKMs(const double time_start) {
    time_start_  = time_start;
    time_end_    = time_start + ENVELOPE_K * 1e-3;
    is_finished_ = true;
  }

  bool Enqueue(const T& event) {
    if (event.time_stemp_ < time_end_) {
      if (event.time_stemp_ >= time_start_) {
        event_vector_.emplace_back(event);
      }
    } else {
      is_finished_ = true;
      return false;
    }
    return true;
  }

  void DownSample(void) {
    // std::cout << event_vector_.size() << std::endl;
  }
};

struct ImuEventData
{
  using Ptr = std::shared_ptr<ImuEventData>;
  double time_stamp_mid_;
  EventKMs<EventRaw>::Ptr event_;
  ImuData::Ptr imu_;
  // GT anglular velocity in body frame;
  Eigen::Vector3d gt_anglular_velocity_;
  // GT linear velocity in body frame;
  Eigen::Vector3d gt_linear_velocity_;

  ImuEventData() = default;
  ImuEventData(EventKMs<EventRaw>::Ptr event,
               ImuData::Ptr imu,
               const Eigen::Vector3d& gt_anglular_velocity,
               const Eigen::Vector3d& gt_linear_velocity)
      : event_(event), imu_(imu), gt_anglular_velocity_(gt_anglular_velocity),
        gt_linear_velocity_(gt_linear_velocity) {
    time_stamp_mid_ = event->time_start_ / 2 + event->time_end_ / 2;
  }
};

class PoseManager {
 public:
  using Ptr = std::shared_ptr<PoseManager>;

  PoseManager() = default;
  PoseManager(GTData::Ptr pose) {
    oldest_time_  = pose->time_stamp_;
    lateset_time_ = pose->time_stamp_;
    pose_.emplace_back(pose);
  }

  int OnPose(std::deque<GTData::Ptr>& pose);

  int GetPose(const double time_stamp, Eigen::Affine3d& pose);

 private:
  std::vector<GTData::Ptr> pose_;
  double lateset_time_;
  double oldest_time_;
};

#endif