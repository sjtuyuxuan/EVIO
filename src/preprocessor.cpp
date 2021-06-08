#include "preprocessor.h"

namespace EVIO {
namespace FrontEnd {

double time_offset = TIME_OFFSET;

void DataSubscriber::GTCallback(
    const geometry_msgs::PoseStampedConstPtr& gt_msg) {
  gt_data_.emplace_back(std::make_shared<GTData>(gt_msg));
}

void DataSubscriber::ImuCallback(const sensor_msgs::ImuConstPtr& imu_masage) {
  imu_data_.emplace_back(std::make_shared<ImuData>(imu_masage));
}

void DataSubscriber::EventCallback(
    const prophesee_event_msgs::EventArray::ConstPtr& event_buffer_msg) {
  for (auto event : event_buffer_msg->events) {
    double ts =
        static_cast<double>(event.ts.nsec) * 1e-9 + event.ts.sec - time_offset;
    // f_out.setf(std::ios::fixed);
    // f_out << std::setprecision(9) << ts
    //       << " " << static_cast<int>(event.x)
    //       << " " << static_cast<int>(event.y)
    //       << " " << static_cast<int>(event.polarity) << std::endl;
    OnEvent(ts, event.x, event.y, static_cast<bool>(event.polarity));
  }
}

void DataSubscriber::OnEvent(const double ts,
                             const uint16_t x,
                             const uint16_t y,
                             const bool p) {
  EventRaw event(ts, x, y, p);
  if (event_data_.empty()) {
    event_data_.emplace_back(
        std::make_shared<EventKMs<EventRaw>>(event.time_stemp_, event));
    return;
  } else if (!event_data_.back()->Enqueue(event)) {
    while (event.time_stemp_ >
           event_data_.back()->time_end_ + ENVELOPE_K * 1e-3) {
      event_data_.emplace_back(
          std::make_shared<EventKMs<EventRaw>>(event_data_.back()->time_end_));
    }
    event_data_.emplace_back(std::make_shared<EventKMs<EventRaw>>(
        event_data_.back()->time_end_, event));
  }
}

bool ImuAlignUnit::GetImuBias (Eigen::Vector3d& gyro_bias,
                               Eigen::Vector3d& acc_g,
                               double& timestamp) {
    while (!imu_cache_.empty() &&
        imu_cache_.front()->time_stamp_ < timestamp - 2) {
      imu_cache_.erase(imu_cache_.begin());
    }
    Eigen::Vector3d acc_g_begin = imu_cache_.front()->linear_acceleration_;
    double count = 0;
    for(auto imu : imu_cache_) {
      if (std::fabs(imu->angular_velocity_.x()) < STATIONARY_IMU_GYRO &&
          std::fabs(imu->angular_velocity_.y()) < STATIONARY_IMU_GYRO &&
          std::fabs(imu->angular_velocity_.z()) < STATIONARY_IMU_GYRO &&
          std::fabs(imu->linear_acceleration_.x() - acc_g_begin.x() ) <
              STATIONARY_IMU_ACC &&
          std::fabs(imu->linear_acceleration_.y() - acc_g_begin.y() ) <
              STATIONARY_IMU_ACC &&
          std::fabs(imu->linear_acceleration_.z() - acc_g_begin.z() ) <
              STATIONARY_IMU_ACC) {
        acc_g += imu->linear_acceleration_;
        gyro_bias += imu->angular_velocity_;
        count += 1;
      } else {
        timestamp = imu->time_stamp_;
        acc_g /= count;
        gyro_bias /= count;
        return true;
      }
    }
    return false;
  }


bool ImuAlignUnit::CheckData(std::deque<EventKMs<EventRaw>::Ptr>& event_que,
                             std::deque<ImuData::Ptr>& imu_que) {
  if (!is_initialized_){
    if (!event_que.empty()) {
      if (event_que.front()->event_vector_.size() < STATIONARY_EVENTS) {
          event_end_time_.emplace_back(event_que.front()->time_end_);
          event_que.pop_front();
          while (!imu_que.empty()){
            imu_cache_.emplace_back(imu_que.front());
            imu_que.pop_front();
          }
        return false;
      } else if (event_end_time_.size() < 50 && imu_cache_.size() < 300) {
        std::cout << 
            "Not initialized! Please LET device STAYIONARY for a while!" <<
            std::endl;
        event_end_time_.clear();
        imu_cache_.clear();
        event_que.clear();
        imu_que.clear();
      } else {
        double time_imu = event_end_time_.back();
        double time_off_set;
        time_off_set /= event_que.front()-> event_vector_.size();
        if (GetImuBias(gyro_bias_, acc_g_, time_imu)) {
          time_offset += event_end_time_.back() - time_imu;
          std::cout << "The time offset is : " << time_offset << std::endl;
          std::cout << "The imu bias is : " << gyro_bias_ << std::endl;
          is_initialized_ = true;
          return false;
        } else {
          time_offset -= 0.2;
          return false;
        }
      }
    }
    return false;
  }
  if (event_que.empty() || (!event_que.front()->is_finished_) ||
      imu_que.empty()) {
    return false;
  }
  while (event_que.front()->time_start_ > imu_que.front()->time_stamp_) {
    imu_que.pop_front();
    if (imu_que.empty()) {
      return false;
    }
  }
  while (event_que.front()->time_end_ < imu_que.front()->time_stamp_) {
    event_que.pop_front();
    if (event_que.empty()) {
      return false;
    }
  }
  if (event_que.front()->time_end_ > imu_que.back()->time_stamp_){
    return false;
  }
  std::vector<ImuData::Ptr> imu_in_envlope;
  while (event_que.front()->time_end_ > imu_que.front()->time_stamp_) {
    imu_in_envlope.emplace_back(imu_que.front());
    imu_que.pop_front();
    if (imu_que.empty()) {
      break;
    }
  }
  if (imu_in_envlope.empty()) {
    return false;
  }
  event_que.front()->DownSample();
  auto imu_event_data = std::make_shared<ImuEventData>(
      event_que.front(), GetAverage(imu_in_envlope));
  imu_event_data_.emplace_back(imu_event_data);
  return true;
}

bool ImuAlignUnit::TryGetData(ImuEventData::Ptr& rtn) {
  if (imu_event_data_.empty()) {
    return false;
  }
  rtn = imu_event_data_.front();
  imu_event_data_.pop_front();
  return true;
}

ImuData::Ptr ImuAlignUnit::GetInterpolation(const double time_stamp,
                                            const ImuData::Ptr pre,
                                            const ImuData::Ptr post) {
  return nullptr;
}

ImuData::Ptr
ImuAlignUnit::GetAverage(const std::vector<ImuData::Ptr> imu_data) {
  if (imu_data.size() == 0) {
    return nullptr;
  }

  Eigen::Vector3d angular_velocity    = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
  for (auto imu : imu_data) {
    angular_velocity += imu->angular_velocity_;
    linear_acceleration += imu->linear_acceleration_;
  }
  angular_velocity /= imu_data.size();
  linear_acceleration /= imu_data.size();
  auto rtn = std::make_shared<ImuData>(
      imu_data.front()->time_stamp_, angular_velocity, linear_acceleration);
  return rtn;
  // trans of cam->imu not opened
}

}  // namespace FrontEnd
}  // namespace EVIO