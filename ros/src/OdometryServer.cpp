// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <Eigen/Core>
#include <memory>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

// KISS-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS 2 headers
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>


/**
 * @brief LookupTransform: retrieves the transformation between two coordinate frames (source_frame and target_frame)
 *                         using the tf2 buffer. It converts the transformation to a Sophus::SE3d object. If the transformation 
 *                         cannot be found or an error occurs, it logs a warning and returns an identity transformation.

 * @param target_frame  : the name of the frame to which the transformation is desired
 * @param source_frame  : the name of the frame from which the transformation id being required
 * @param tf2_buffer    : unique pointer to a tf2_ros::Buffer object, which is used to store and look up transformations
 * @return Sophus::SE3d : A Sophus::SE3d object representing the transformation from the source_frame to the target_frame
*/
namespace {
Sophus::SE3d LookupTransform(const std::string &target_frame,
                             const std::string &source_frame,
                             const std::unique_ptr<tf2_ros::Buffer> &tf2_buffer) {
    std::string err_msg;
    // check if the trasformation from source_frame to target_frame is available
    if (tf2_buffer->canTransform(target_frame, source_frame, tf2::TimePointZero, &err_msg)) {
        try {
            // lookup the transform and convert it to Sophus::SE3d
            auto tf = tf2_buffer->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
            return tf2::transformToSophus(tf);
        } catch (tf2::TransformException &ex) {
            // log a warning if an exception occurs
            RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "%s", ex.what());
        }
    }
    // log a warning if the transform is not available and return identity transform
    RCLCPP_WARN(rclcpp::get_logger("LookupTransform"), "Failed to find tf. Reason=%s",
                err_msg.c_str());
    // default construction is the identity
    return Sophus::SE3d();
}
}  // namespace

namespace kiss_icp_ros {

// Utility functions from the utils namespace
using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

/**
 * @brief OdometryServer: Constructor for the OdometryServer class. This constructor initializes the OdometryServer node 
 *                        with specified options, declares and sets up parameters, constructs the KISS-ICP odometry node, and
 *                        initializes the necessary subscribers, publishers, and transform broadcaster.
 * @param options: The options to configure the node, such as parameter overrides and context (note: here no options are 
 *                 specified so the default ones are used)
 */
OdometryServer::OdometryServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("kiss_icp_node", options) {
    
    // Declare parameters
    base_frame_             = declare_parameter<std::string>("base_frame", base_frame_);
    odom_frame_             = declare_parameter<std::string>("odom_frame", odom_frame_);
    publish_odom_tf_        = declare_parameter<bool>("publish_odom_tf", publish_odom_tf_);
    publish_debug_clouds_   = declare_parameter<bool>("publish_debug_clouds", false);
    position_covariance_    = declare_parameter<double>("position_covariance", 0.1);
    orientation_covariance_ = declare_parameter<double>("orientation_covariance", 0.1);

    // Set up KISS-ICP configuration parameters
    kiss_icp::pipeline::KISSConfig config;
    config.max_range             = declare_parameter<double>("max_range", config.max_range);
    config.min_range             = declare_parameter<double>("min_range", config.min_range);
    config.deskew                = declare_parameter<bool>("deskew", config.deskew);
    config.voxel_size            = declare_parameter<double>("voxel_size", config.max_range / 100.0);
    config.max_points_per_voxel  = declare_parameter<int>("max_points_per_voxel", config.max_points_per_voxel);
    config.initial_threshold     = declare_parameter<double>("initial_threshold", config.initial_threshold);
    config.min_motion_th         = declare_parameter<double>("min_motion_th", config.min_motion_th);
    config.max_num_iterations    = declare_parameter<int>("max_num_iterations", config.max_num_iterations);
    config.convergence_criterion = declare_parameter<double>("convergence_criterion", config.convergence_criterion);
    config.max_num_threads       = declare_parameter<int>("max_num_threads", config.max_num_threads);

    // Ensure max_range is not smaller than min_range. If not, set min_range to 0
    if (config.max_range < config.min_range) {
        RCLCPP_WARN(get_logger(), "[WARNING] max_range is smaller than min_range, settng min_range to 0.0");
        config.min_range = 0.0;
    }

    // Construct the main KISS-ICP odometry node with the configured parameters
    kiss_icp_ = std::make_unique<kiss_icp::pipeline::KissICP>(config);

    // Initialize the subscriber for point cloud data: when the message of type sensor_msgs::msg::PointCloud2 is published on
    // the specified pointcloud_topic, the subscribers call the method RegisterFrame
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud_topic", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){
            this->RegisterFrame(msg);
        });

    // Initialize publishers for odometry and optionally for debug point clouds
    rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("/kiss/odometry", qos);
    if (publish_debug_clouds_) {
        frame_publisher_   = create_publisher<sensor_msgs::msg::PointCloud2>("/kiss/frame", qos);
        kpoints_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/kiss/keypoints", qos);
        map_publisher_     = create_publisher<sensor_msgs::msg::PointCloud2>("/kiss/local_map", qos);
    }

    // Initialize the transform broadcaster and listener for handling transformations
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf2_buffer_     = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf2_buffer_->setUsingDedicatedThread(true);
    tf2_listener_   = std::make_unique<tf2_ros::TransformListener>(*tf2_buffer_);

    // Log node initialization
    RCLCPP_INFO(this->get_logger(), "KISS-ICP ROS 2 odometry node initialized");
   
}

/**
 * @brief RegisterFrame: Processes an incoming PointCloud2 message to an Eigen format, retrieves timestamps, and 
 *                       registers the frame using the KISS-ICP pipeline. It then publishes the estimated pose and, 
 *                       if enabled, the debug point clouds.
 * @param msg: A constant shared pointer to the incoming PointCloud2 message.
 */
void OdometryServer::RegisterFrame(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    const auto cloud_frame_id = msg->header.frame_id;     // Retrieve the frame ID from the message header
    const auto points         = PointCloud2ToEigen(msg);  // Convert the PointCloud2 message to Eigen format
    const auto timestamps     = GetTimestamps(msg);       // Get timestamps from the PointCloud2 message

    // Register frame, main entry point to KISS-ICP pipeline
    const auto &[frame, keypoints] = kiss_icp_->RegisterFrame(points, timestamps);

    // Extract the last KISS-ICP pose, ego-centric to the LiDAR
    const Sophus::SE3d kiss_pose = kiss_icp_->pose();

    // Spit the current estimated pose to ROS msgs handling the desired target frame
    PublishOdometry(kiss_pose, msg->header);
    // Publishing these clouds is a bit costly, so do it only if we are debugging
    if (publish_debug_clouds_) {
        PublishClouds(frame, keypoints, msg->header);
    }
}

void OdometryServer::PublishOdometry(const Sophus::SE3d &kiss_pose,
                                     const std_msgs::msg::Header &header) {
    // If necessary, transform the ego-centric pose to the specified base_link/base_footprint frame
    const auto cloud_frame_id = header.frame_id;
    const auto egocentric_estimation = (base_frame_.empty() || base_frame_ == cloud_frame_id);
    const auto pose = [&]() -> Sophus::SE3d {
        if (egocentric_estimation) return kiss_pose;
        const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id, tf2_buffer_);
        return cloud2base * kiss_pose * cloud2base.inverse();
    }();

    // Broadcast the tf ---
    if (publish_odom_tf_) {
        geometry_msgs::msg::TransformStamped transform_msg;
        transform_msg.header.stamp = header.stamp;
        transform_msg.header.frame_id = odom_frame_;
        transform_msg.child_frame_id = egocentric_estimation ? cloud_frame_id : base_frame_;
        transform_msg.transform = tf2::sophusToTransform(pose);
        tf_broadcaster_->sendTransform(transform_msg);
    }

    // publish odometry msg
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = header.stamp;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.pose.pose = tf2::sophusToPose(pose);
    odom_msg.pose.covariance.fill(0.0);
    odom_msg.pose.covariance[0] = position_covariance_;
    odom_msg.pose.covariance[7] = position_covariance_;
    odom_msg.pose.covariance[14] = position_covariance_;
    odom_msg.pose.covariance[21] = orientation_covariance_;
    odom_msg.pose.covariance[28] = orientation_covariance_;
    odom_msg.pose.covariance[35] = orientation_covariance_;
    odom_publisher_->publish(std::move(odom_msg));
}

void OdometryServer::PublishClouds(const std::vector<Eigen::Vector3d> frame,
                                   const std::vector<Eigen::Vector3d> keypoints,
                                   const std_msgs::msg::Header &header) {
    const auto kiss_map = kiss_icp_->LocalMap();
    const auto kiss_pose = kiss_icp_->pose().inverse();

    frame_publisher_->publish(std::move(EigenToPointCloud2(frame, header)));
    kpoints_publisher_->publish(std::move(EigenToPointCloud2(keypoints, header)));
    map_publisher_->publish(std::move(EigenToPointCloud2(kiss_map, kiss_pose, header)));
}
}  // namespace kiss_icp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(kiss_icp_ros::OdometryServer)
