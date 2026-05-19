#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include "visualization_msgs/msg/marker.hpp"
#include <pcl/filters/radius_outlier_removal.h>

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inliers_pub;

std::shared_ptr<rclcpp::Node> node;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

typedef pcl::PointXYZ PointT;

int marker_id = 0;
bool verbose = true;

void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    rclcpp::Time now = msg->header.stamp;

    // Transform entire cloud to map frame
    sensor_msgs::msg::PointCloud2 cloud_map_msg;
    try {
        auto tss = tf_buffer_->lookupTransform("map", msg->header.frame_id, now);
        tf2::doTransform(*msg, cloud_map_msg, tss);
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN(node->get_logger(), "%s", ex.what());
        return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(cloud_map_msg, *cloud);

    if (verbose)
        RCLCPP_INFO(node->get_logger(), "PointCloud has: %zu points", cloud->points.size());

    // Filter by height in map frame (z = up, unambiguous)
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::PassThrough<PointT> passz;
    passz.setInputCloud(cloud);
    passz.setFilterFieldName("z");
    passz.setFilterLimits(0.7, 2.0);
    passz.filter(*cloud_filtered);

    if (cloud_filtered->points.empty())
        return;

    // Remove dense surfaces (walls, furniture) - keep sparse edge points
    pcl::PointCloud<PointT>::Ptr cloud_no_planes(new pcl::PointCloud<PointT>);
    pcl::RadiusOutlierRemoval<PointT> ror;
    ror.setInputCloud(cloud_filtered);
    ror.setRadiusSearch(0.05);
    ror.setMinNeighborsInRadius(50);
    ror.setNegative(true);
    ror.filter(*cloud_no_planes);

    if (verbose)
        RCLCPP_INFO(node->get_logger(), "PointCloud after filtering: %zu points", cloud_no_planes->points.size());

    // Publish filtered cloud for debug
    sensor_msgs::msg::PointCloud2 filtered_out_msg;
    pcl::toROSMsg(*cloud_no_planes, filtered_out_msg);
    filtered_out_msg.header.frame_id = "map";
    filtered_out_msg.header.stamp = now;
    filtered_pub->publish(filtered_out_msg);

    if (cloud_no_planes->points.empty())
        return;

    // Try fitting circles of each target size
    // {radius, margin}
    std::vector<std::pair<float,float>> targets = {{0.20f, 0.02f}, {0.075f, 0.02f}};

    pcl::SACSegmentation<PointT> seg_circle;
    seg_circle.setOptimizeCoefficients(true);
    seg_circle.setModelType(pcl::SACMODEL_CIRCLE3D);
    seg_circle.setMethodType(pcl::SAC_RANSAC);
    seg_circle.setMaxIterations(1000);
    seg_circle.setDistanceThreshold(0.02);
    seg_circle.setInputCloud(cloud_no_planes);

    for (const auto& [target_r, margin] : targets) {
        seg_circle.setRadiusLimits(target_r - margin, target_r + margin);

        pcl::ModelCoefficients::Ptr coefficients_circle(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers_circle(new pcl::PointIndices);
        seg_circle.segment(*inliers_circle, *coefficients_circle);

        if (coefficients_circle->values.size() == 0)
            continue;

        // coefficients: [cx, cy, cz, radius, nx, ny, nz]
        Eigen::Vector3f center(coefficients_circle->values[0],
                               coefficients_circle->values[1],
                               coefficients_circle->values[2]);
        float radius = coefficients_circle->values[3];
        Eigen::Vector3f normal(coefficients_circle->values[4],
                               coefficients_circle->values[5],
                               coefficients_circle->values[6]);

        // Always publish debug disk so you can see what was fit even if rejected
        visualization_msgs::msg::Marker debug_circle;
        debug_circle.header.frame_id = "map";
        debug_circle.header.stamp = now;
        debug_circle.ns = "debug_fit";
        debug_circle.id = 0;
        debug_circle.type = visualization_msgs::msg::Marker::CYLINDER;
        debug_circle.action = visualization_msgs::msg::Marker::ADD;
        debug_circle.pose.position.x = center[0];
        debug_circle.pose.position.y = center[1];
        debug_circle.pose.position.z = center[2];
        debug_circle.pose.orientation.w = 1.0;
        debug_circle.scale.x = radius * 2;
        debug_circle.scale.y = radius * 2;
        debug_circle.scale.z = 0.01;
        debug_circle.color.r = 1.0f;
        debug_circle.color.a = 0.5f;
        debug_circle.lifetime = rclcpp::Duration(1, 0);
        marker_pub->publish(debug_circle);

        // Inlier count sanity
        size_t n_inliers = inliers_circle->indices.size();
        if (n_inliers < 35) {
            if (verbose) RCLCPP_INFO(node->get_logger(), "[r=%.2f] Rejected: too few inliers (%zu)", target_r, n_inliers);
            continue;
        }
        if (n_inliers > 1500) {
            if (verbose) RCLCPP_INFO(node->get_logger(), "[r=%.2f] Rejected: too many inliers (%zu)", target_r, n_inliers);
            continue;
        }

        // Reject horizontal circles - floor/ceiling have normal parallel to Z
        Eigen::Vector3f up(0.0f, 0.0f, 1.0f);
        if (std::abs(normal.dot(up)) > 0.8f) {
            if (verbose) RCLCPP_INFO(node->get_logger(), "[r=%.2f] Rejected: circle is horizontal", target_r);
            continue;
        }

        // Extract inliers from cloud_no_planes (same cloud RANSAC ran on)
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud_no_planes);
        extract.setIndices(inliers_circle);
        extract.setNegative(false);
        pcl::PointCloud<PointT>::Ptr cloud_circle(new pcl::PointCloud<PointT>());
        extract.filter(*cloud_circle);

        if (cloud_circle->points.empty())
            continue;

        // Publish inliers for debug
        sensor_msgs::msg::PointCloud2 inliers_msg;
        pcl::toROSMsg(*cloud_circle, inliers_msg);
        inliers_msg.header.frame_id = "map";
        inliers_msg.header.stamp = now;
        inliers_pub->publish(inliers_msg);

        // Annulus validation
        float mean_dist = 0.0f;
        for (const auto& pt : cloud_circle->points)
            mean_dist += std::abs((Eigen::Vector3f(pt.x, pt.y, pt.z) - center).norm() - radius);
        mean_dist /= cloud_circle->points.size();

        if (verbose)
            RCLCPP_INFO(node->get_logger(), "[r=%.2f] mean_dist: %.4f radius: %.4f", target_r, mean_dist, radius);

        if (mean_dist > 0.2f) {
            if (verbose) RCLCPP_INFO(node->get_logger(), "[r=%.2f] Rejected: mean_dist %.4f > 0.02", target_r, mean_dist);
            continue;
        }

        // Arc coverage + hollow check in one pass over cloud_no_planes
        std::vector<double> angles;
        int inner_points = 0;

        for (const auto& pt : cloud_no_planes->points) {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f v = p - center;
            float dist_from_plane = std::abs(v.dot(normal));

            if (dist_from_plane > 0.05f)
                continue;

            Eigen::Vector3f v_projected = v - v.dot(normal) * normal;
            float dist_from_center = v_projected.norm();

            if (dist_from_center < radius * 0.8f)
                inner_points++;

            float dist_from_circumference = std::abs(dist_from_center - radius);
            if (dist_from_circumference < 0.02f)
                angles.push_back(std::atan2(v_projected[1], v_projected[0]));
        }

        if (inner_points > 10) {
            if (verbose) RCLCPP_INFO(node->get_logger(), "[r=%.2f] Rejected: %d points inside ring", target_r, inner_points);
            continue;
        }

        if (angles.size() < 3) {
            if (verbose) RCLCPP_INFO(node->get_logger(), "[r=%.2f] Rejected: too few angle samples", target_r);
            continue;
        }

        std::sort(angles.begin(), angles.end());
        double max_gap = 0;
        for (size_t i = 1; i < angles.size(); i++)
            max_gap = std::max(max_gap, angles[i] - angles[i-1]);
        max_gap = std::max(max_gap, (angles.front() + 2*M_PI) - angles.back());

        if (max_gap > M_PI) {
            if (verbose) RCLCPP_INFO(node->get_logger(), "[r=%.2f] Rejected: arc gap %.1f deg", target_r, max_gap * 180/M_PI);
            continue;
        }

        // Passed all checks - publish marker
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = now;
        marker.ns = "ring";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = center[0];
        marker.pose.position.y = center[1];
        marker.pose.position.z = center[2];
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;
        marker.lifetime = rclcpp::Duration(10, 0);
        marker_pub->publish(marker);

        // Publish circle cloud
        sensor_msgs::msg::PointCloud2 circle_out_msg;
        pcl::toROSMsg(*cloud_circle, circle_out_msg);
        circle_out_msg.header.frame_id = "map";
        circle_out_msg.header.stamp = now;
        cylinder_pub->publish(circle_out_msg);

        if (verbose)
            RCLCPP_INFO(node->get_logger(), "Ring detected at map (%.2f, %.2f, %.2f) r=%.3f inliers=%zu",
                        center[0], center[1], center[2], radius, n_inliers);
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::cout << "cylinder_segmentation" << std::endl;

    node = rclcpp::Node::make_shared("cylinder_segmentation");

    // create subscriber
    node->declare_parameter<std::string>("topic_pointcloud_in", "/oakd/rgb/preview/depth/points");
    std::string param_topic_pointcloud_in = node->get_parameter("topic_pointcloud_in").as_string();
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription = node->create_subscription<sensor_msgs::msg::PointCloud2>(param_topic_pointcloud_in, 10, &cloud_cb);

    // setup tf listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // create publishers
    cylinder_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("cylinder", 1);
    marker_pub = node->create_publisher<visualization_msgs::msg::Marker>("detected_cylinder", 1);
    filtered_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", 1);
    inliers_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("inliers_circle", 1);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
