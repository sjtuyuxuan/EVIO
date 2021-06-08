#include "evio.h"
#include "frame.h"
#include "preprocessor.h"

#include <boost/functional/hash.hpp>
#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using EVIO::FrontEnd::DataSubscriber;
using EVIO::FrontEnd::EventKMs;
using EVIO::FrontEnd::EventProcessed;
using EVIO::FrontEnd::EventRaw;
using EVIO::FrontEnd::Frame;
using EVIO::FrontEnd::GenMapWarp;
using EVIO::FrontEnd::Grid;
using EVIO::FrontEnd::ImuAlignUnit;
using EVIO::FrontEnd::ImuData;
using EVIO::FrontEnd::ImuEventData;
using EVIO::FrontEnd::JRot;
using EVIO::FrontEnd::MapWarp;
using EVIO::FrontEnd::PoseManager;

int PrintImage(EventKMs<EventRaw>::Ptr raw_event, Frame::Ptr frame) {
  if (raw_event->event_vector_.size() < 5) {
    return STATIONARY;
  } else if (raw_event->event_vector_.size() < 1500) {
    std::cout << "NOT ENOUGH" << std::endl;
    return SUCC;
  } else {
    std::cout << "ENOUGH" << std::endl;
    cv::RNG rng;
    cv::Mat debug_image_raw(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    cv::Mat debug_image_pro(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    // cv::Mat debug_image_arrored_pos(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    // cv::Mat debug_image_arrored_neg(HEIGHT, WIDTH, CV_8UC1, cv::Scalar(0));
    cv::Mat debug_image_cluster(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat debug_image_cluster_neg(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    for (EventRaw pixel : raw_event->event_vector_) {
      debug_image_raw.at<uint8_t>(pixel.y_, pixel.x_) =
          std::min(static_cast<uint8_t>(
                       (debug_image_raw.at<uint8_t>(pixel.y_, pixel.x_) + 83)),
                   static_cast<uint8_t>(255));
    }
    for (EventProcessed pixel :
         frame->undistorted_roation_warp_->event_vector_) {
      debug_image_pro.at<cv::Vec3b>(pixel.y_, pixel.x_)[0] +=
          (pixel.time_stemp_ - frame->undistorted_roation_warp_->time_start_) * 1600;

      debug_image_pro.at<cv::Vec3b>(pixel.y_, pixel.x_)[2] +=
          (- pixel.time_stemp_ + frame->undistorted_roation_warp_->time_start_) * 1600;
    }
    cv::putText( debug_image_pro, "Time(ms) = " +
        std::to_string(frame->time_for_warp_process_ * 1000),
        cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));

    // for (auto grid : frame->grid_frame_pos_) {
    //   cv::Point center_point(
    //       grid.second->center_x_ + grid.second->GetMean().x(),
    //       grid.second->center_y_ + grid.second->GetMean().y());
    //   cv::Point direct_point(center_point);
    //   direct_point.x += 30 * grid.second->GetDirection().x();
    //   direct_point.y += 30 * grid.second->GetDirection().y();
    //   int size = grid.second->events_in_grid_.size() / 4;
    //   cv::arrowedLine(debug_image_arrored_pos,
    //                   center_point,
    //                   direct_point,
    //                   cv::Scalar(std::min(255, size)),
    //                   2);
    // }
    // for (auto grid : frame->grid_frame_neg_) {
    //   cv::Point center_point(
    //       grid.second->center_x_ + grid.second->GetMean().x(),
    //       grid.second->center_y_ + grid.second->GetMean().y());
    //   cv::Point direct_point(center_point);
    //   direct_point.x += 30 * grid.second->GetDirection().x();
    //   direct_point.y += 30 * grid.second->GetDirection().y();
    //   int size = grid.second->events_in_grid_.size() / 3;
    //   cv::arrowedLine(debug_image_arrored_neg,
    //                   center_point,
    //                   direct_point,
    //                   cv::Scalar(std::min(255, size)),
    //                   2);
    // }
    for (auto cluster : frame->cluster_result_pos_) {
      cv::Vec3b color{static_cast<unsigned char>(rng.uniform(0, 255)),
                      static_cast<unsigned char>(rng.uniform(0, 255)),
                      static_cast<unsigned char>(rng.uniform(0, 255))};
      for (auto point : cluster) {
        debug_image_cluster.at<cv::Vec3b>(point->event_.y_, point->event_.x_) =
            color;
      }
    }
    for (auto cluster : frame->cluster_result_neg_) {
      cv::Vec3b color{static_cast<unsigned char>(rng.uniform(0, 255)),
                      static_cast<unsigned char>(rng.uniform(0, 255)),
                      static_cast<unsigned char>(rng.uniform(0, 255))};
      for (auto point : cluster) {
        debug_image_cluster_neg.at<cv::Vec3b>(point->event_.y_, point->event_.x_) =
            color;
      }
    }
    cv::putText( debug_image_cluster, "Time(ms) = " +
        std::to_string(frame->time_for_cluster_process_ * 1000),
        cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));

    std::string path_raw =
        "/mnt/750ssd_1/temp/" +
        std::to_string(frame->undistorted_roation_warp_->time_start_) + "a.png";
    std::string path_processed =
        "/mnt/750ssd_1/temp/" +
        std::to_string(frame->undistorted_roation_warp_->time_start_) + "b.png";
    // std::string path_arrored_pos =
    //     "/mnt/750ssd_1/temp/" +
    //     std::to_string(frame->undistorted_roation_warp_->time_start_) +
    //     "c.png";
    // std::string path_arrored_neg =
    //     "/mnt/750ssd_1/temp/" +
    //     std::to_string(frame->undistorted_roation_warp_->time_start_) +
    //     "d.png";
    std::string path_cluster =
        "/mnt/750ssd_1/temp/" +
        std::to_string(frame->undistorted_roation_warp_->time_start_) + "e.png";
    std::string path_cluster_neg =
        "/mnt/750ssd_1/temp/" +
        std::to_string(frame->undistorted_roation_warp_->time_start_) + "f.png";
    for (int index = 1; index < HEIGHT / GRID_SIZE; index++) {
      cv::Point left_point(0, index * GRID_SIZE);
      cv::Point right_point(WIDTH, index * GRID_SIZE);
      cv::line(debug_image_raw, left_point, right_point, cv::Scalar(100));
    }
    for (int index = 1; index < WIDTH / GRID_SIZE; index++) {
      cv::Point up_point(index * GRID_SIZE, 0);
      cv::Point down_point(index * GRID_SIZE, HEIGHT);
      cv::line(debug_image_raw, up_point, down_point, cv::Scalar(100));
    }
    for (int index = 1; index < HEIGHT / GRID_SIZE; index++) {
      cv::Point left_point(0, index * GRID_SIZE);
      cv::Point right_point(WIDTH, index * GRID_SIZE);
      cv::line(debug_image_pro, left_point, right_point, cv::Scalar(100));
    }
    for (int index = 1; index < WIDTH / GRID_SIZE; index++) {
      cv::Point up_point(index * GRID_SIZE, 0);
      cv::Point down_point(index * GRID_SIZE, HEIGHT);
      cv::line(debug_image_pro, up_point, down_point, cv::Scalar(100));
    }
    cv::imwrite(path_raw, debug_image_raw);
    cv::imwrite(path_processed, debug_image_pro);
    // cv::imwrite(path_arrored_pos, debug_image_arrored_pos);
    // cv::imwrite(path_arrored_neg, debug_image_arrored_neg);
    cv::imwrite(path_cluster, debug_image_cluster);
    cv::imwrite(path_cluster_neg, debug_image_cluster_neg);
  }

  return SUCC;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "event_printer");
  DataSubscriber data_subscriber("/imu/data",
                                 "/prophesee/camera/cd_events_buffer");
  Eigen::Matrix3d r_ci;
  r_ci << 0, -1, 0, 0, 0, -1, 1, 0, 0;
  Eigen::Translation3d t_ci(0, 0, 0);
  Eigen::Affine3d c2i = t_ci * r_ci;
  Eigen::Matrix3d intrinsic_mat;
  intrinsic_mat << 551.0904959933358, 0, 312.5664494105658,
                   0, 551.1346807591603, 236.6593903974883,
                   0, 0, 1;
  ImuAlignUnit imu_align_unit;
  ros::Rate loop_rate(200);
  int stay_count = 0;
  std::vector<ImuData::Ptr> imu_cache;
  JRot j_rot(intrinsic_mat);
  std::unordered_map<uvpair, MapWarp, boost::hash<uvpair>> map;
  GenMapWarp(intrinsic_mat, map);
  PoseManager::Ptr gt_pose_manager = nullptr;
  if (USE_GT) {
    if (!data_subscriber.gt_data_.empty()) {
      gt_pose_manager = std::make_shared<PoseManager>(
          data_subscriber.gt_data_.front());
    } else {
      gt_pose_manager = std::make_shared<PoseManager>();
    }
  }
  while (ros::ok()) {
    gt_pose_manager->OnPose(data_subscriber.gt_data_);
    imu_align_unit.CheckData(data_subscriber.event_data_,
                             data_subscriber.imu_data_);
    ImuEventData::Ptr imu_event_data;
    if (imu_align_unit.TryGetData(imu_event_data)) {
      auto frame = std::make_shared<Frame>(imu_event_data, intrinsic_mat, c2i);
      frame->WarpRotation(map);
      // frame->GridSegment();
      // frame->GridProcess();
      frame->DBScanSegment();

      PrintImage(imu_event_data->event_, frame) == STATIONARY ? stay_count++ :
                                                                stay_count = 0;
      stay_count ? imu_cache.emplace_back(imu_event_data->imu_) :
                   imu_cache.clear();
    }

    loop_rate.sleep();
    ros::spinOnce();
  }
}
