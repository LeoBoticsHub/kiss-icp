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

#include "KissICP.hpp"

#include <Eigen/Core>
#include <vector>

#include "kiss_icp/core/Deskew.hpp"
#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"

namespace kiss_icp::pipeline {

/**
 * @brief RegisterFrame: Register a frame of 3D points with optional deskewing based on timestamps.
                         If deskewing is enabled and timestamps are provided, the frame is deskewed
 *                       using the DeSkewScan method. The resulting frame (deskewed or original)
 *                       is then passed to another RegisterFrame method for further processing.
 * @param frame The input frame of 3D points.
 * @param timestamps The timestamps associated with each 3D point in the frame.
 * @return KissICP::Vector3dVectorTuple The result of the registration process.
 */
KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<double> &timestamps) {
    // Lambda function to conditionally deskew the frame
    const auto &deskew_frame = [&]() -> std::vector<Eigen::Vector3d> {
        // If deskewing is disabled or if timestamps are empty
        if (!config_.deskew || timestamps.empty()) return frame;
        // Otherwise deskew the frame
        return DeSkewScan(frame, timestamps, last_delta_);
    }();
    // Register the (potentially deskewed) frame using another RegisterFrame method
    return RegisterFrame(deskew_frame);
}

/**
 * @brief RegisterFrame: Register a frame of 3D points by preprocessing, voxelizing, and running ICP 
 *                      (Iterative Closest Point) registration to align the frame with a local map. 
 *                       It updates the pose and model deviation accordingly.
 * @param frame: The input frame of 3D points.
 * @return KissICP::Vector3dVectorTuple {frame, source}: The deskewed input raw scan and the points used for registration.
 */
KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame) {
    // Preprocess the input cloud to include only 3D points included within the specified range limits [min_range, max_range]
    const auto &cropped_frame = Preprocess(frame, config_.max_range, config_.min_range);

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(cropped_frame);

    // Get adaptive_threshold
    const double sigma = adaptive_threshold_.ComputeThreshold();

    // Compute initial_guess for ICP
    const auto initial_guess = last_pose_ * last_delta_;

    // Run ICP
    const auto new_pose = registration_.AlignPointsToMap(source,         // frame
                                                         local_map_,     // voxel_map
                                                         initial_guess,  // initial_guess
                                                         3.0 * sigma,    // max_correspondence_dist
                                                         sigma / 3.0);   // kernel

    // Compute the difference between the prediction and the actual estimate
    const auto model_deviation = initial_guess.inverse() * new_pose;

    // Update step: threshold, local map, delta, and the last pose
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(frame_downsample, new_pose);
    last_delta_ = last_pose_.inverse() * new_pose;
    last_pose_ = new_pose;

    // Return the (deskew) input raw scan (frame) and the points used for registration (source)
    return {frame, source};
}

KissICP::Vector3dVectorTuple KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame) const {
    
    // define number of downsample to apply (0, 1 or 2)
    int nDownsample = 0;

    // init variables
    const auto voxel_size = config_.voxel_size;
    std::vector<Eigen::Vector3d> frame_downsample;
    std::vector<Eigen::Vector3d> source;

    switch (nDownsample) {
        // No downsample 
        case 0:
            frame_downsample = frame;
            source           = frame;
            break;
        // Downsample one time
        case 1:
            frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
            source           = frame;
            break;
        // Downsample two times
        case 2:
            frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
            source           = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
            break;
        default:
            frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
            source           = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 1.5);
            break;
    }
    
    return {source, frame_downsample};
}

}  // namespace kiss_icp::pipeline
