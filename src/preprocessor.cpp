#include "preprocessor.h"

namespace EVIO {
namespace FrontEnd {

double time_offset_imu = TIME_OFFSET_IMU;
double time_offset_gt  = TIME_OFFSET_GT;

void DataSubscriber::GTCallback(
    const geometry_msgs::PoseStampedConstPtr& gt_msg) {
  std::lock_guard<std::mutex> _(gt_data_lock_);
  gt_data_.emplace_back(std::make_shared<GTData>(gt_msg, time_offset_gt));
}

void DataSubscriber::ImuCallback(const sensor_msgs::ImuConstPtr& imu_masage) {
  std::lock_guard<std::mutex> _(imu_data_lock_);
  imu_data_.emplace_back(
      std::make_shared<ImuData>(imu_masage, time_offset_imu));
}

void DataSubscriber::EventCallback(
    const prophesee_event_msgs::EventArray::ConstPtr& event_buffer_msg) {
  std::lock_guard<std::mutex> _(event_data_lock_);
  for (auto event : event_buffer_msg->events) {
    double ts = static_cast<double>(event.ts.nsec) * 1e-9 + event.ts.sec;
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

bool ImuAlignUnit::GetTimeBias(Eigen::Vector3d& gyro_bias,
                               Eigen::Vector3d& acc_g,
                               double& time_imu,
                               double& time_gt) {
  while (!imu_cache_.empty() &&
         imu_cache_.front()->time_stamp_ < time_imu - 3) {
    imu_cache_.erase(imu_cache_.begin());
  }
  if (USE_GT) {
    while (!gt_cache_.empty() && gt_cache_.front()->time_stamp_ < time_gt - 3) {
      gt_cache_.erase(gt_cache_.begin());
    }
  }
  int rtn_succ                = 0;
  Eigen::Vector3d acc_g_begin = imu_cache_.front()->linear_acceleration_;
  double count                = 0;
  for (auto imu : imu_cache_) {
    if (std::fabs(imu->angular_velocity_.x()) < STATIONARY_IMU_GYRO &&
        std::fabs(imu->angular_velocity_.y()) < STATIONARY_IMU_GYRO &&
        std::fabs(imu->angular_velocity_.z()) < STATIONARY_IMU_GYRO &&
        std::fabs(imu->linear_acceleration_.x() - acc_g_begin.x()) <
            STATIONARY_IMU_ACC &&
        std::fabs(imu->linear_acceleration_.y() - acc_g_begin.y()) <
            STATIONARY_IMU_ACC &&
        std::fabs(imu->linear_acceleration_.z() - acc_g_begin.z()) <
            STATIONARY_IMU_ACC) {
      acc_g += imu->linear_acceleration_;
      gyro_bias += imu->angular_velocity_;
      count += 1;
    } else {
      time_imu = imu->time_stamp_;
      acc_g /= count;
      gyro_bias /= count;
      rtn_succ++;
      break;
    }
  }
  if (USE_GT) {
    if (!gt_cache_.empty()) {
      Eigen::Quaterniond rot_start_inv = gt_cache_[10]->q.inverse();
      for (auto gt : gt_cache_) {
        // rot < 3 deg
        if ((rot_start_inv * gt->q).w() < cos(STATIONARY_GT_ROT * M_PI / 180)) {
          time_gt = gt->time_stamp_;
          rtn_succ++;
          break;
        }
      }
    }
  } else {
    rtn_succ++;
  }

  return rtn_succ == 2;
}

bool ImuAlignUnit::CheckData(DataSubscriber& subscriber) {
  std::lock_guard<std::mutex> imu_(subscriber.imu_data_lock_);
  std::lock_guard<std::mutex> event_(subscriber.event_data_lock_);
  std::lock_guard<std::mutex> gt_(subscriber.gt_data_lock_);
  if (!is_initialized_) {
    if (!subscriber.event_data_.empty()) {
      if (subscriber.event_data_.front()->event_vector_.size() <
          STATIONARY_EVENTS) {
        event_end_time_.emplace_back(subscriber.event_data_.front()->time_end_);
        subscriber.event_data_.pop_front();
        while (!subscriber.imu_data_.empty()) {
          imu_cache_.emplace_back(subscriber.imu_data_.front());
          subscriber.imu_data_.pop_front();
        }
        if (USE_GT) {
          while (!subscriber.gt_data_.empty()) {
            gt_cache_.emplace_back(subscriber.gt_data_.front());
            subscriber.gt_data_.pop_front();
          }
        }
        return false;
      } else if (event_end_time_.size() < 50 && imu_cache_.size() < 300 &&
                 gt_cache_.size() < 180) {
        std::cout << "Not initialized! Please LET the device be STAYIONARY for "
                     "a while!"
                  << std::endl;
        event_end_time_.clear();
        imu_cache_.clear();
        gt_cache_.clear();
        subscriber.event_data_.clear();
        subscriber.imu_data_.clear();
        subscriber.gt_data_.clear();
      } else {
        double time_imu = event_end_time_.back();
        double time_gt  = event_end_time_.back();
        if (GetTimeBias(gyro_bias_, acc_g_, time_imu, time_gt)) {
          time_offset_imu += event_end_time_.back() - time_imu;
          std::cout << "The imu time offset is : " << time_offset_imu
                    << std::endl;
          if (USE_GT) {
            time_offset_gt += event_end_time_.back() - time_gt;
            std::cout << "The gt time offset is : " << time_offset_gt
                      << std::endl;
          }
          std::cout << "The imu bias is : " << std::endl
                    << gyro_bias_ << std::endl;
          is_initialized_ = true;
          subscriber.event_data_.clear();
          subscriber.imu_data_.clear();
          subscriber.gt_data_.clear();
          return false;
        } else {
          time_offset_imu -= 0.2;
          time_offset_gt -= 0.2;
          return false;
        }
      }
    }
    return false;
  }

  if (USE_GT) {
    gt_manager_->OnPose(subscriber.gt_data_);
  }

  if (subscriber.event_data_.empty() ||
      (!subscriber.event_data_.front()->is_finished_) ||
      subscriber.imu_data_.empty()) {
    return false;
  }
  while (subscriber.event_data_.front()->time_start_ >
         subscriber.imu_data_.front()->time_stamp_) {
    subscriber.imu_data_.pop_front();
    if (subscriber.imu_data_.empty()) {
      return false;
    }
  }
  while (subscriber.event_data_.front()->time_end_ <
         subscriber.imu_data_.front()->time_stamp_) {
    subscriber.event_data_.pop_front();
    if (subscriber.event_data_.empty()) {
      return false;
    }
  }
  if (subscriber.event_data_.front()->time_end_ >
      subscriber.imu_data_.back()->time_stamp_) {
    return false;
  }
  std::vector<ImuData::Ptr> imu_in_envlope;
  while (subscriber.event_data_.front()->time_end_ >
         subscriber.imu_data_.front()->time_stamp_) {
    imu_in_envlope.emplace_back(subscriber.imu_data_.front());
    subscriber.imu_data_.pop_front();
    if (subscriber.imu_data_.empty()) {
      break;
    }
  }
  if (imu_in_envlope.empty()) {
    return false;
  }
  Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d linear_velocity  = Eigen::Vector3d::Zero();
  if (USE_GT) {
    Eigen::Affine3d start_pose;
    Eigen::Affine3d end_pose;
    if (gt_manager_->GetPose(subscriber.event_data_.front()->time_start_,
                             start_pose) == SUCC &&
        gt_manager_->GetPose(subscriber.event_data_.front()->time_end_,
                             end_pose) == SUCC) {
      Eigen::Affine3d delta_pose = start_pose.inverse() * end_pose;
      Eigen::AngleAxisd rot(delta_pose.rotation());
      angular_velocity = rot.axis() * rot.angle() * 1e3 / ENVELOPE_K;
      linear_velocity  = delta_pose.translation() * 1e3 / ENVELOPE_K;
    } else {
      return false;
    }
  }
  subscriber.event_data_.front()->DownSample();
  auto imu_event_data =
      std::make_shared<ImuEventData>(subscriber.event_data_.front(),
                                     GetAverage(imu_in_envlope),
                                     angular_velocity,
                                     linear_velocity);
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