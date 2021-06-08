#ifndef EVIO_PREPROCESSOR_H_
#define EVIO_PREPROCESSOR_H_

#include <deque>

#include <prophesee_event_msgs/Event.h>
#include <prophesee_event_msgs/EventArray.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>

#include "evio.h"
#include <Eigen/Core>
#include <Eigen/Dense>

namespace EVIO {
namespace FrontEnd {

struct GTData
{
  using Ptr = std::shared_ptr<GTData>;
  Eigen::Vector3d t;
  Eigen::Quaterniond q;
  double time_stamp_;

  GTData() = delete;
  GTData(const geometry_msgs::PoseStampedConstPtr& gt_masage) {
    t.x() = gt_masage->pose.position.x;
    t.y() = gt_masage->pose.position.y;
    t.z() = gt_masage->pose.position.z;
    q.w() = gt_masage->pose.orientation.w;
    q.x() = gt_masage->pose.orientation.x;
    q.y() = gt_masage->pose.orientation.y;
    q.z() = gt_masage->pose.orientation.z;
    time_stamp_ = 1e-9 * gt_masage->header.stamp.nsec +
                  gt_masage->header.stamp.sec;
  }
};

struct ImuData
{
  using Ptr = std::shared_ptr<ImuData>;
  Eigen::Vector3d angular_velocity_;
  Eigen::Vector3d linear_acceleration_;
  double time_stamp_;

  ImuData() = delete;
  ImuData(const sensor_msgs::ImuConstPtr& imu_masage) {
    time_stamp_ = static_cast<double>(imu_masage->header.stamp.nsec * 1e-9) +
                  imu_masage->header.stamp.sec;
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

  ImuEventData() = default;
  ImuEventData(EventKMs<EventRaw>::Ptr event, ImuData::Ptr imu)
      : event_(event), imu_(imu) {
    time_stamp_mid_ = event->time_start_ / 2 + event->time_end_ / 2;
  }
};

// imu topic event topic
class DataSubscriber {
 private:
  ros::NodeHandle nh_;
  ros::Subscriber imu_sub_;
  ros::Subscriber event_sub_;
  ros::Subscriber gt_sub_;
  // ros::Subscriber lidar_sub_;
  void ImuCallback(const sensor_msgs::ImuConstPtr& imu_masage);

  void EventCallback(
      const prophesee_event_msgs::EventArray::ConstPtr& event_buffer_msg);

  void GTCallback(
      const geometry_msgs::PoseStampedConstPtr& gt_msg);

  void
  OnEvent(const double ts, const uint16_t x, const uint16_t y, const bool p);

 public:
  std::deque<EventKMs<EventRaw>::Ptr> event_data_;
  std::deque<ImuData::Ptr> imu_data_;
  std::deque<GTData::Ptr> gt_data_; 

  DataSubscriber() = delete;
  DataSubscriber(const std::string& imu_topic, const std::string& event_topic)
      : nh_("~") {
    imu_sub_ = nh_.subscribe<sensor_msgs::Imu>(
        imu_topic, 2000, &EVIO::FrontEnd::DataSubscriber::ImuCallback, this);
    event_sub_ = nh_.subscribe<prophesee_event_msgs::EventArray>(
        event_topic, 200, &EVIO::FrontEnd::DataSubscriber::EventCallback, this);
    if (USE_GT) {
      gt_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(
        event_topic, 500, &EVIO::FrontEnd::DataSubscriber::GTCallback, this);
    }
    // f_out.open(out_path);
  }

  ~DataSubscriber() {
    nh_.shutdown();
  }
};

// Imu is aligned to the middle time stamp of event envelope
class ImuAlignUnit {
 public:
  bool GetBias (Eigen::Vector3d& gyro_bias, Eigen::Vector3d& acc_g) {
    if (is_initialized_) {
      gyro_bias = gyro_bias_;
      acc_g = acc_g_;
      return true;
    } else {
      return false;
    }
  }

  bool CheckData(std::deque<EventKMs<EventRaw>::Ptr>& event_que,
                 std::deque<ImuData::Ptr>& imu_que);

  bool TryGetData(ImuEventData::Ptr& rtn);


 private:
  std::deque<ImuEventData::Ptr> imu_event_data_;

  // for small interval imu data (just for small K <= 10)
  ImuData::Ptr GetInterpolation(const double time_stamp,
                                const ImuData::Ptr pre,
                                const ImuData::Ptr post);

  // for long interval imu data (just large K > 10)
  ImuData::Ptr GetAverage(const std::vector<ImuData::Ptr> imu_data);
  bool GetImuBias (Eigen::Vector3d& gyro_bias,
                   Eigen::Vector3d& acc_bias,
                   double& timestamp);

  std::vector<ImuData::Ptr> imu_cache_;
  std::vector<double> event_end_time_;
  bool is_initialized_ = true;
  Eigen::Vector3d gyro_bias_;
  Eigen::Vector3d acc_g_;
};

}  // namespace FrontEnd
}  // namespace EVIO

#endif