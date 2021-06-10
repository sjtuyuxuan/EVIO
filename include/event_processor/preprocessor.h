#ifndef EVIO_PREPROCESSOR_H_
#define EVIO_PREPROCESSOR_H_

#include "evio.h"
#include "util.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <mutex>

namespace EVIO {
namespace FrontEnd {

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

  void GTCallback(const geometry_msgs::PoseStampedConstPtr& gt_msg);

  void
  OnEvent(const double ts, const uint16_t x, const uint16_t y, const bool p);

 public:
  std::deque<EventKMs<EventRaw>::Ptr> event_data_;
  std::deque<ImuData::Ptr> imu_data_;
  std::deque<GTData::Ptr> gt_data_;

  std::mutex event_data_lock_;
  std::mutex imu_data_lock_;
  std::mutex gt_data_lock_;

  DataSubscriber() = delete;
  DataSubscriber(const std::string& imu_topic,
                 const std::string& event_topic,
                 const std::string& gt_topic)
      : nh_("~") {
    imu_sub_ = nh_.subscribe<sensor_msgs::Imu>(
        imu_topic, 2000, &EVIO::FrontEnd::DataSubscriber::ImuCallback, this);
    event_sub_ = nh_.subscribe<prophesee_event_msgs::EventArray>(
        event_topic, 200, &EVIO::FrontEnd::DataSubscriber::EventCallback, this);
    if (USE_GT) {
      gt_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(
          gt_topic, 1200, &EVIO::FrontEnd::DataSubscriber::GTCallback, this);
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
  bool GetBias(Eigen::Vector3d& gyro_bias, Eigen::Vector3d& acc_g) {
    if (is_initialized_) {
      gyro_bias = gyro_bias_;
      acc_g     = acc_g_;
      return true;
    } else {
      return false;
    }
  }

  bool CheckData(DataSubscriber& subscriber);

  bool TryGetData(ImuEventData::Ptr& rtn);

 private:
  std::deque<ImuEventData::Ptr> imu_event_data_;

  // for small interval imu data (just for small K <= 10)
  ImuData::Ptr GetInterpolation(const double time_stamp,
                                const ImuData::Ptr pre,
                                const ImuData::Ptr post);

  // for long interval imu data (just large K > 10)
  ImuData::Ptr GetAverage(const std::vector<ImuData::Ptr> imu_data);
  bool GetTimeBias(Eigen::Vector3d& gyro_bias,
                   Eigen::Vector3d& acc_bias,
                   double& time_imu,
                   double& time_gt);

  PoseManager::Ptr gt_manager_ = std::make_shared<PoseManager>();

  // for initialize
  std::vector<ImuData::Ptr> imu_cache_;
  std::vector<double> event_end_time_;
  std::vector<GTData::Ptr> gt_cache_;

  bool is_initialized_       = !ESTIMATE_OFFSET;
  Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_g_     = Eigen::Vector3d::Zero();
};

}  // namespace FrontEnd
}  // namespace EVIO

#endif