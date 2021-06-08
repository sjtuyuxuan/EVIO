#ifndef EVIO_FRAME_H_
#define EVIO_FRAME_H_

#include "evio.h"
#include "preprocessor.h"

#include <boost/functional/hash.hpp>
#include <deque>
#include <fstream>
#include <opencv2/flann.hpp>
#include <unordered_map>

typedef std::pair<uint16_t, uint16_t> uvpair;

namespace EVIO {
namespace FrontEnd {

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

int GenMapWarp(const Eigen::Matrix3d intrinsic_mat,
               std::unordered_map<uvpair, MapWarp, boost::hash<uvpair>>& map);

struct JRot
{
  double fx, fy, cx, cy;
  double temp_1_fx2, temp_2cx, temp_cx2;
  double temp_1_fy2, temp_2cy, temp_cy2;

  JRot() = delete;
  JRot(const Eigen::Matrix3d& intrinsic_mat)
      : fx(intrinsic_mat(0, 0)), fy(intrinsic_mat(1, 1)),
        cx(intrinsic_mat(0, 2)), cy(intrinsic_mat(1, 2)) {
    temp_1_fx2 = 1 / (fx * fx);
    temp_2cx   = 2 * cx;
    temp_cx2   = cx * cx;
    temp_1_fy2 = 1 / (fy * fy);
    temp_2cy   = 2 * cy;
    temp_cy2   = cy * cy;
  }
  Eigen::Matrix<double, 2, 3> operator()(uint16_t u, uint16_t v) {
    Eigen::Matrix<double, 2, 3> rtn;
    rtn << 0, -(1 + temp_1_fx2 * (u * u - temp_2cx * u + temp_cx2)),
        -(v - cy) / fy, (1 + temp_1_fy2 * (v * v - temp_2cy * v + temp_cy2)), 0,
        (u - cx) / fx;
    rtn.block<1, 3>(0, 0) *= fx;
    rtn.block<1, 3>(1, 0) *= fy;
    return rtn;
  }
};

struct JTrans
{
  double fx, fy, cx, cy;
  JTrans() = delete;
  JTrans(const Eigen::Matrix3d& intrinsic_mat)
      : fx(intrinsic_mat(0, 0)), fy(intrinsic_mat(1, 1)),
        cx(intrinsic_mat(0, 2)), cy(intrinsic_mat(1, 2)) {}
  Eigen::Matrix<double, 2, 3> operator()(uint16_t u, uint16_t v) {}
};

struct PatchInfo
{
};


class PoseManager {
 public:
  using Ptr = std::shared_ptr<PoseManager>;

  PoseManager() = default;
  PoseManager(GTData::Ptr pose) {
    oldest_time_ = pose->time_stamp_;
    lateset_time_ = pose->time_stamp_;
    pose_.emplace_back(pose);
  }

  int OnPose(std::deque<GTData::Ptr>& pose) {
    while (!pose.empty()){
      pose_.emplace_back(pose.front());
      if (pose.front()->time_stamp_ < lateset_time_) {
        return TIME_INVERSE;
      } else if (pose.front()->time_stamp_ > lateset_time_ + 0.05) {
        return HUGE_GAP;
      } else {
        lateset_time_ = pose.front()->time_stamp_;
      }
      pose.pop_front();
    }
    return SUCC;
  }

  int GetPose (const double time_stamp, Eigen::Affine3d& pose) {
    if (time_stamp < oldest_time_ || time_stamp > lateset_time_ ) {
      return OUT_OF_RANGE;
    } else {
      int index = std::floor((time_stamp - oldest_time_) /
          (lateset_time_ - oldest_time_) * (pose_.size() - 1));
      while (pose_[index]->time_stamp_ > time_stamp) index--;
      while (pose_[index + 1]->time_stamp_ < time_stamp) index++;
      double ratio = (time_stamp - pose_[index]->time_stamp_) /
          (pose_[index + 1]->time_stamp_ - pose_[index]->time_stamp_);
      
      pose = Eigen::Translation3d((1 - ratio) * pose_[index]->t +
          ratio * pose_[index + 1]->t) *
          pose_[index]->q.slerp(ratio, pose_[index + 1]->q);
      return SUCC;
    }
  }

 private:
  std::vector<GTData::Ptr> pose_;
  double lateset_time_;
  double oldest_time_;

};




class Grid {
 public:
  using Ptr = std::shared_ptr<Grid>;

  float center_x_;
  float center_y_;
  bool polarity_;
  float grid_width_ = GRID_SIZE;
  std::vector<EventProcessed> events_in_grid_;
  Grid() = delete;
  Grid(float u, float v, bool polarity)
      : center_x_(u), center_y_(v), polarity_(polarity) {
    // up_u_   = center_x_ + grid_width_ / 2;
    // up_v_   = center_y_ + grid_width_ / 2;
    // down_u_ = up_u_ - grid_width_;
    // down_v_ = up_v_ - grid_width_;
  }

  int AddPoint(const EventProcessed& event);

  int Process(const Eigen::Vector3d tans = Eigen::Vector3d::Zero());

  // the mean is in global coordinate
  Eigen::Vector2f GetMean(void) {
    return mean_;
  }

  // the dorection vector is normalized
  Eigen::Vector2f GetDirection(void) {
    return direction_;
  }

  int Depth(double& depth, double& inverse_depth_var_);

 private:
  // float up_u_, up_v_, down_u_, down_v_;
  float sum_x_        = 0;
  float sum_y_        = 0;
  float sum_suqare_x_ = 0;
  float sum_suqare_y_ = 0;
  float sum_xy_       = 0;

  double inverse_depth_;
  double inverse_depth_var_;
  double depth_;

  Eigen::Vector2f mean_;
  Eigen::Vector2f direction_;
  Eigen::Matrix2f statistic_matrix_;
};

#define UNREACHED -1
#define NOISE -2

struct DBPoint
{
  using Ptr = std::shared_ptr<DBPoint>;
  EventProcessed event_;
  std::vector<float> coordinate_;
  std::vector<int> reach_index_;
  int cluster_ = UNREACHED;

  DBPoint(EventProcessed& event) {
    event_ = event;
    coordinate_.emplace_back(event.x_);
    coordinate_.emplace_back(event.y_);
    reach_index_.reserve(DBSCAN_RESERVE);
  }
};

using ClusterVector = std::vector<std::vector<DBPoint::Ptr>>;

class DBScan {
 public:
  using Ptr = std::shared_ptr<DBScan>;

  DBScan() = delete;
  DBScan(EventKMs<EventProcessed>::Ptr event_kms, ClusterVector* result) :
      result_(result) {
    for (EventProcessed event : event_kms->event_vector_) {
      dbpoint_list_.emplace_back(std::make_shared<DBPoint>(event));
      points_in_tree_.emplace_back(cv::Point2f(event.x_, event.y_));
    }
    if (!points_in_tree_.empty()){
      Kdtree_.build(cv::Mat(points_in_tree_).reshape(1),
                    cv::flann::KDTreeIndexParams(1),
                    cvflann::FLANN_DIST_EUCLIDEAN);
    }
  }

  int CountNear(void) {
    std::vector<float> dis;
    dis.reserve(DBSCAN_RESERVE);
    for (auto dbpoint : dbpoint_list_) {
      cv::flann::SearchParams params(32);
      Kdtree_.radiusSearch(dbpoint->coordinate_,
                           dbpoint->reach_index_,
                           dis,
                           DBSCAN_DIS,
                           DBSCAN_RESERVE,
                           params);
      if (dbpoint->reach_index_.size() > DBSCAN_COUNT) {
        valid_center_points_.emplace_back(dbpoint);
      } else {
        dbpoint->cluster_ = NOISE;
      }
    }
    return SUCC;
  }

  void DFS(DBPoint::Ptr dbpoint, const int times) {
    if (DBSCAN_DFS_DEPTH && times == DBSCAN_DFS_DEPTH) return;
    for (int index : dbpoint->reach_index_) {
      if (dbpoint_list_[index]->cluster_ == UNREACHED) {
        dbpoint_list_[index]->cluster_ = dbpoint->cluster_;
        (*result_)[dbpoint->cluster_].emplace_back(dbpoint_list_[index]);
        DFS(dbpoint_list_[index], times + 1);
      }
    }
  }

  int process(void) {
    int cluster_count = 0;
    for (auto dbpoint : valid_center_points_) {
      if (dbpoint->cluster_ == UNREACHED) {
        std::vector<DBPoint::Ptr> cluster{dbpoint};
        result_->emplace_back(cluster);
        dbpoint->cluster_ = cluster_count;
        DFS(dbpoint, 0);
        cluster_count++;
      }
    }
    auto it = result_->begin();
    while (it != result_->end()) {
      if (it->size() < DBSCAN_MIN_SIZE) {
        it = result_->erase(it);
      } else {
        it++;
      }
    }
    return SUCC;
  }

 private:
  ClusterVector* result_;
  std::vector<DBPoint::Ptr> valid_center_points_;
  cv::flann::Index Kdtree_;
  std::vector<cv::Point2f> points_in_tree_;
  std::vector<DBPoint::Ptr> dbpoint_list_;
};

struct DBLine {
  int size            = 0;
  float sum_x_        = 0;
  float sum_y_        = 0;
  float sum_suqare_x_ = 0;
  float sum_suqare_y_ = 0;
  float sum_xy_       = 0;

  Eigen::Vector2f meam;
  float length;
  cv::Point2f point_1;
  cv::Point2f point_2;
};

class Frame {
 public:
  using Ptr = std::shared_ptr<Frame>;
  Frame()   = delete;
  Frame(ImuEventData::Ptr& imu_event_data,
        const Eigen::Matrix3d intrinsic_mat,
        const Eigen::Affine3d c2i,
        const Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero(),
        const Eigen::Vector3d acc_bias  = Eigen::Vector3d::Zero())
      : intrinsic_mat_(intrinsic_mat), imu_event_raw_data_(imu_event_data),
        cam2imu_(c2i), gyro_bias_(gyro_bias), acc_bias_(acc_bias) {}

  int WarpRotation(
      const std::unordered_map<uvpair, MapWarp, boost::hash<uvpair>>& map);
  
  int WarpRotation(
      const std::unordered_map<uvpair, MapWarp, boost::hash<uvpair>>& map,
      const Eigen::AngleAxisd& diff_pose_q);

  int GridSegment(void);

  int GridProcess(void);

  int DBScanSegment(void);

  int DBScanLineDetct(void);

  EventKMs<EventProcessed>::Ptr undistorted_roation_warp_ = nullptr;
  EventKMs<EventProcessed>::Ptr undistorted_all_warp_     = nullptr;
  std::unordered_map<uvpair, Grid::Ptr, boost::hash<uvpair>> grid_frame_pos_;
  std::unordered_map<uvpair, Grid::Ptr, boost::hash<uvpair>> grid_frame_neg_;
  ClusterVector cluster_result_;
  ClusterVector cluster_result_pos_;
  ClusterVector cluster_result_neg_;

  float time_for_grid_process_;
  float time_for_cluster_process_;
  float time_for_warp_process_;

 private:
  ImuEventData::Ptr imu_event_raw_data_;
  Eigen::Matrix3d intrinsic_mat_;
  Eigen::Affine3d cam2imu_;
  Eigen::Vector3d gyro_bias_;
  Eigen::Vector3d acc_bias_;
};

}  // namespace FrontEnd
}  // namespace EVIO
#endif