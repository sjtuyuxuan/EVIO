#include "util.h"

int PoseManager::OnPose(std::deque<GTData::Ptr>& pose) {
  if (pose_.empty() && !pose.empty()) {
    oldest_time_  = pose.front()->time_stamp_;
    lateset_time_ = pose.front()->time_stamp_;
  }
  while (!pose.empty()) {
    if (pose.front()->time_stamp_ < lateset_time_ - 0.001) {
      return TIME_INVERSE;
    } else if (pose.front()->time_stamp_ > lateset_time_ + 0.1) {
      return HUGE_GAP;
    } else {
      pose_.emplace_back(pose.front());
      lateset_time_ = pose.front()->time_stamp_;
    }
    pose.pop_front();
  }
  return SUCC;
}

int PoseManager::GetPose(const double time_stamp, Eigen::Affine3d& pose) {
  if (time_stamp < oldest_time_ || time_stamp > lateset_time_) {
    return OUT_OF_RANGE;
  } else {
    int index = std::floor((time_stamp - oldest_time_) /
                           (lateset_time_ - oldest_time_) * (pose_.size() - 1));
    while (pose_[index]->time_stamp_ > time_stamp)
      index--;
    while (pose_[index + 1]->time_stamp_ < time_stamp)
      index++;
    double ratio = (time_stamp - pose_[index]->time_stamp_) /
                   (pose_[index + 1]->time_stamp_ - pose_[index]->time_stamp_);

    pose = Eigen::Translation3d((1 - ratio) * pose_[index]->t +
                                ratio * pose_[index + 1]->t) *
           pose_[index]->q.slerp(ratio, pose_[index + 1]->q);
    return SUCC;
  }
}