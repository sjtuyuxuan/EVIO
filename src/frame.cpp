#include "frame.h"
#include <time.h>

namespace EVIO {
namespace FrontEnd {

int GenMapWarp(const Eigen::Matrix3d intrinsic_mat,
               std::unordered_map<uvpair, MapWarp, boost::hash<uvpair>>& map) {
  float fx = intrinsic_mat(0, 0);
  float fy = intrinsic_mat(1, 1);
  float cx = intrinsic_mat(0, 2);
  float cy = intrinsic_mat(1, 2);
  for (uint16_t u = 0; u < WIDTH; u++) {
    for (uint16_t v = 0; v < HEIGHT; v++) {
      MapWarp map_warp;
      map_warp.x_ = static_cast<float>(u);
      map_warp.y_ = static_cast<float>(v);
      map_warp.j_rot_ << (u - cx) * (v - cy) / fy,
          -fx * (1 + std::pow((u - cx) / fx, 2)), fx * (v - cy) / fy,
          fy * (1 + std::pow((v - cy) / fy, 2)), -(u - cx) * (v - cy) / fx,
          -fy * (u - cx) / fx;
      float chi = std::sqrt(std::pow(((u - cx) / fx), 2) +
                            std::pow(((v - cy) / fy), 2) + 1);
      map_warp.j_trans_ << -fx * chi, 0, (u - cx) * chi, 0, -fy * chi,
          (v - cy) * chi;
      map[std::make_pair(u, v)] = map_warp;
    }
  }
  return SUCC;
}

int Grid::AddPoint(const EventProcessed& event) {
  // closed for speed
  // if (event.polarity_ != polarity_ || event.x_ >= up_u_ ||
  //     event.x_ <= down_u_ || event.y_ >= up_v_ || event.y_ <= down_v_) {
  //   return UNEXPECT_ERROR;
  // }
  events_in_grid_.emplace_back(event);
  float x = event.x_ - center_x_;
  float y = event.y_ - center_y_;
  sum_x_ += x;
  sum_y_ += y;
  sum_suqare_x_ += x * x;
  sum_suqare_y_ += y * y;
  sum_xy_ += x * y;
  return SUCC;
};

int Grid::Process(const Eigen::Vector3d tans) {
  mean_ << sum_x_ / events_in_grid_.size(), sum_y_ / events_in_grid_.size();
  statistic_matrix_ << sum_suqare_x_ - sum_x_ * mean_.x(),
      sum_xy_ - sum_x_ * mean_.y(), sum_xy_ - sum_y_ * mean_.x(),
      sum_suqare_y_ - sum_y_ * mean_.y();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(statistic_matrix_);
  if (solver.info() != Eigen::Success) {
    return UNEXPECT_ERROR;
  }
  solver.eigenvalues()(0) >= solver.eigenvalues()(1) ?
      direction_ = solver.eigenvectors().col(0) :
      direction_ = solver.eigenvectors().col(1);
  direction_.y() > 0 ? direction_ = direction_ : direction_ = -direction_;
}

int DBLine::Process(void) {
  for (auto point : *points_) {
    sum_x_ += point->event_.x_;
    sum_y_ += point->event_.y_;
    sum_suqare_x_ += point->event_.x_ * point->event_.x_;
    sum_suqare_y_ += point->event_.y_ * point->event_.y_;
    sum_xy_ += point->event_.x_ * point->event_.y_;
    sum_xt_ += point->event_.x_ * (point->event_.time_stemp_ - time_stamp_);
    sum_yt_ += point->event_.y_ * (point->event_.time_stemp_ - time_stamp_);
    sum_t_ += point->event_.time_stemp_ - time_stamp_;
  }
  mean_t = sum_t_ / (*points_).size();
  mean_ << sum_x_ / (*points_).size(), sum_y_ / (*points_).size();
  xy_cov_ << sum_suqare_x_ - sum_x_ * mean_.x(), sum_xy_ - sum_x_ * mean_.y(),
      sum_xy_ - sum_y_ * mean_.x(), sum_suqare_y_ - sum_y_ * mean_.y();
  xy_t_ << sum_xt_ - sum_x_ * mean_t, sum_yt_ - sum_y_ * mean_t;
  line_direction_ = xy_cov_.inverse() * xy_t_;
  line_vec_       = -line_direction_ / std::pow(line_direction_.norm(), 2);
  std::cout << line_vec_ << std::endl;
  for (auto point : *points_) {
    result.emplace_back(EventProcessed(
        time_stamp_,
        point->event_.x_ +
            (point->event_.time_stemp_ - time_stamp_) * line_vec_.x(),
        point->event_.y_ +
            (point->event_.time_stemp_ - time_stamp_) * line_vec_.y(),
        point->event_.polarity_));
  }
  return SUCC;
}

int Frame::WarpRotation(
    const std::unordered_map<uvpair, MapWarp, boost::hash<uvpair>>& map) {
  if (undistorted_roation_warp_ != nullptr) {
    return REPEAT_PROCESS;
  }
  // std::cout << "imu:" << std::endl
  //           << cam2imu_ * imu_event_raw_data_->imu_->angular_velocity_
  //           << std::endl
  //           << "GT:" << std::endl
  //           << imu_event_raw_data_->gt_anglular_velocity_ << std::endl;
  undistorted_roation_warp_ = std::make_shared<EventKMs<EventProcessed>>(
      imu_event_raw_data_->event_->time_start_);
  clock_t start, end;
  start = clock();
  for (auto event : imu_event_raw_data_->event_->event_vector_) {
    auto map_pixel = map.find(std::make_pair(event.x_, event.y_))->second;
    Eigen::Vector2f warp =
        static_cast<float>(imu_event_raw_data_->time_stamp_mid_ -
                           event.time_stemp_) *
        map_pixel.j_rot_ *
        // Eigen::Vector3f(
        //     static_cast<float>(imu_event_raw_data_->gt_anglular_velocity_.x()),
        //     static_cast<float>(imu_event_raw_data_->gt_anglular_velocity_.y()),
        //     static_cast<float>(imu_event_raw_data_->gt_anglular_velocity_.z()));
        Eigen::Vector3f(static_cast<float>(
                            -imu_event_raw_data_->imu_->angular_velocity_.y()),
                        static_cast<float>(
                            -imu_event_raw_data_->imu_->angular_velocity_.z()),
                        static_cast<float>(
                            imu_event_raw_data_->imu_->angular_velocity_.x()));
    float u = warp(0, 0) + map_pixel.x_;
    float v = warp(1, 0) + map_pixel.y_;
    if (u > 0 && v > 0 && u < WIDTH && v < HEIGHT) {
      EventProcessed rotation_warped(event.time_stemp_, u, v, event.polarity_);
      undistorted_roation_warp_->Enqueue(rotation_warped);
    }
  }
  end                    = clock();
  time_for_warp_process_ = static_cast<float>(end - start) / CLOCKS_PER_SEC;
  return SUCC;
}

int Frame::GridSegment(void) {
  for (auto event : undistorted_roation_warp_->event_vector_) {
    uint16_t index_x = event.x_ / GRID_SIZE;
    uint16_t index_y = event.y_ / GRID_SIZE;
    auto pair        = std::make_pair(index_x, index_y);
    if (event.polarity_) {
      if (grid_frame_pos_.find(pair) == grid_frame_pos_.end()) {
        grid_frame_pos_[pair] = std::make_shared<Grid>(
            static_cast<float>(index_x * GRID_SIZE + GRID_SIZE / 2),
            static_cast<float>(index_y * GRID_SIZE + GRID_SIZE / 2),
            true);
      }
      grid_frame_pos_[pair]->AddPoint(event);
    } else {
      if (grid_frame_neg_.find(pair) == grid_frame_neg_.end()) {
        grid_frame_neg_[pair] = std::make_shared<Grid>(
            static_cast<float>(index_x * GRID_SIZE + GRID_SIZE / 2),
            static_cast<float>(index_y * GRID_SIZE + GRID_SIZE / 2),
            true);
      }
      grid_frame_neg_[pair]->AddPoint(event);
    }
  }
}

int Frame::GridProcess(void) {
  for (auto grid = grid_frame_pos_.begin(); grid != grid_frame_pos_.end();) {
    if (grid->second->events_in_grid_.size() > 80) {
      grid->second->Process();
      grid++;
    } else {
      grid = grid_frame_pos_.erase(grid);
    }
  }
  for (auto grid = grid_frame_neg_.begin(); grid != grid_frame_neg_.end();) {
    if (grid->second->events_in_grid_.size() > 80) {
      grid->second->Process();
      grid++;
    } else {
      grid = grid_frame_neg_.erase(grid);
    }
  }
  return SUCC;
}

int Frame::DBScanSegment(void) {
  if (!DBSCAN_DEVIDE_POLARITY) {
    clock_t start, end;
    start = clock();
    DBScan db_scan(undistorted_roation_warp_, &cluster_result_);
    db_scan.CountNear();
    db_scan.process();
    end = clock();
    time_for_cluster_process_ =
        static_cast<float>(end - start) / CLOCKS_PER_SEC;
    return SUCC;
  } else {
    auto pos_kms = std::make_shared<EventKMs<EventProcessed>>(
        undistorted_roation_warp_->time_start_);
    auto neg_kms = std::make_shared<EventKMs<EventProcessed>>(
        undistorted_roation_warp_->time_start_);
    for (auto event : undistorted_roation_warp_->event_vector_) {
      event.polarity_ ? pos_kms->event_vector_.emplace_back(event) :
                        neg_kms->event_vector_.emplace_back(event);
    }
    clock_t start, end;
    start = clock();
    DBScan db_scan_pos(pos_kms, &cluster_result_pos_);
    db_scan_pos.CountNear();
    db_scan_pos.process();
    DBScan db_scan_neg(neg_kms, &cluster_result_neg_);
    db_scan_neg.CountNear();
    db_scan_neg.process();
    end = clock();
    time_for_cluster_process_ =
        static_cast<float>(end - start) / CLOCKS_PER_SEC;
  }
}

int Frame::DBScanLineProcess(void) {
  for (auto cluster : cluster_result_pos_) {
    dblines_.emplace_back(std::make_shared<DBLine>(
        &cluster, imu_event_raw_data_->time_stamp_mid_));
    dblines_.back()->Process();
  }
  for (auto cluster : cluster_result_neg_) {
    dblines_.emplace_back(std::make_shared<DBLine>(
        &cluster, imu_event_raw_data_->time_stamp_mid_));
    dblines_.back()->Process();
  }
}
}  // namespace FrontEnd
}  // namespace EVIO