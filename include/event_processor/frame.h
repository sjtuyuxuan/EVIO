#ifndef EVIO_FRAME_H_
#define EVIO_FRAME_H_

#include "evio.h"
#include "util.h"

#include <boost/functional/hash.hpp>
#include <deque>
#include <fstream>
#include <opencv2/flann.hpp>
#include <unordered_map>

typedef std::pair<uint16_t, uint16_t> uvpair;

namespace EVIO {
namespace FrontEnd {

int GenMapWarp(const Eigen::Matrix3d intrinsic_mat,
               std::unordered_map<uvpair, MapWarp, boost::hash<uvpair>>& map);

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
  DBScan(EventKMs<EventProcessed>::Ptr event_kms, ClusterVector* result)
      : result_(result) {
    for (EventProcessed event : event_kms->event_vector_) {
      dbpoint_list_.emplace_back(std::make_shared<DBPoint>(event));
      points_in_tree_.emplace_back(cv::Point2f(event.x_, event.y_));
    }
    if (!points_in_tree_.empty()) {
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
    if (DBSCAN_DFS_DEPTH && times == DBSCAN_DFS_DEPTH)
      return;
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

class DBLine {
 public:
  using Ptr = std::shared_ptr<DBLine>;

  std::vector<EventProcessed> result;
  Eigen::Vector2d line_direction_;
  Eigen::Vector2d line_vec_;
  float mean_t;
  Eigen::Vector2d mean_;

  DBLine() = delete;
  DBLine(std::vector<DBPoint::Ptr>* points, double time_stamp)
      : points_(points), time_stamp_(time_stamp) {}

  int Process(void);

 private:
  std::vector<DBPoint::Ptr>* points_;

  double time_stamp_;
  int size             = 0;
  double sum_x_        = 0;
  double sum_y_        = 0;
  double sum_suqare_x_ = 0;
  double sum_suqare_y_ = 0;
  double sum_xy_       = 0;
  double sum_xt_       = 0;
  double sum_yt_       = 0;
  double sum_t_        = 0;

  Eigen::Matrix2d xy_cov_;
  Eigen::Vector2d xy_t_;
};
using DBLines = std::vector<std::shared_ptr<DBLine>>;

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

  int DBScanLineProcess(void);

  EventKMs<EventProcessed>::Ptr undistorted_roation_warp_ = nullptr;
  EventKMs<EventProcessed>::Ptr undistorted_all_warp_     = nullptr;
  std::unordered_map<uvpair, Grid::Ptr, boost::hash<uvpair>> grid_frame_pos_;
  std::unordered_map<uvpair, Grid::Ptr, boost::hash<uvpair>> grid_frame_neg_;
  ClusterVector cluster_result_;
  ClusterVector cluster_result_pos_;
  ClusterVector cluster_result_neg_;
  DBLines dblines_;

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