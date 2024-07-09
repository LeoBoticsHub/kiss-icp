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
#include "Preprocessing.hpp"

#include <tbb/parallel_for.h>
#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <sophus/se3.hpp>
#include <vector>

namespace {
using Voxel = Eigen::Vector3i;
struct VoxelHash {
    size_t operator()(const Voxel &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};
}  // namespace

namespace kiss_icp {
/**
 * @brief VoxelDownsample: Downsamples a point cloud using a voxel grid filter.
 *                         This function performs voxel grid downsampling on an input point cloud.
 *                         Each voxel represents a cube of side length `voxel_size`. Points falling
 *                         within the same voxel are reduced to a single representative point, 
 *                         effectively reducing the number of points in the output cloud.
 * @param frame:      The input point cloud as a vector of Eigen::Vector3d points.
 * @param voxel_size: The size of each voxel in the grid.
 * @return vector<Eigen::Vector3d> frame_dowsampled: A vector of Eigen::Vector3d points representing the downsampled point cloud.
 */
std::vector<Eigen::Vector3d> VoxelDownsample(const std::vector<Eigen::Vector3d> &frame,
                                             double voxel_size) {
    // Create a hash map to store unique voxels and their representative points
    tsl::robin_map<Voxel, Eigen::Vector3d, VoxelHash> grid;
    grid.reserve(frame.size());  // Reserve memory for efficiency
    
    // Iterate over each point in the input frame
    for (const auto &point : frame) {
        // Calculate the voxel index for the current point
        const auto voxel = Voxel((point / voxel_size).cast<int>());
        // If the voxel already exists in the grid, skip it
        if (grid.contains(voxel)) continue;
        // Insert the voxel and its corresponding point into the grid
        grid.insert({voxel, point});
    }

    // Create a vector to store the downsampled point cloud
    std::vector<Eigen::Vector3d> frame_dowsampled;
    frame_dowsampled.reserve(grid.size()); // Reserve memory for efficiency
    // Iterate over the voxels in the grid
    for (const auto &[voxel, point] : grid) {
        (void)voxel; // Avoid unused variable warning for the voxel key
        frame_dowsampled.emplace_back(point); // Add the representative point to the downsampled frame
    }

    // Return the downsampled point cloud
    return frame_dowsampled;
}

/**
 * @brief Preprocess: Preprocess a frame of 3D points by filtering points based on range limits. 
 *                    This function filters the input 3D points to include only those within the specified
 *                    range limits (between min_range and max_range). Points outside this range are discarded.
 * @param frame:     The input frame of 3D points.
 * @param max_range: The maximum allowable distance for points to be included.
 * @param min_range: The minimum allowable distance for points to be included.
 * @return std::vector<Eigen::Vector3d> corrected_frame: The filtered frame of 3D points within the specified range.
 */
std::vector<Eigen::Vector3d> Preprocess(const std::vector<Eigen::Vector3d> &frame,
                                        double max_range,
                                        double min_range) {
    std::vector<Eigen::Vector3d> inliers;
    // Copy points from frame to inliers if their norm (distance) is within the specified range
    std::copy_if(frame.cbegin(), frame.cend(), std::back_inserter(inliers), [&](const auto &pt) {
        // Calculate the distance of the point from the origin (norm)
        const double norm = pt.norm();
        // Include the point if it is within range
        return norm < max_range && norm > min_range;
    });
    return inliers;
}

std::vector<Eigen::Vector3d> CorrectKITTIScan(const std::vector<Eigen::Vector3d> &frame) {
    constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        const auto &pt = frame[i];
        const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
        corrected_frame[i] =
            Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
    });
    return corrected_frame;
}
}  // namespace kiss_icp
