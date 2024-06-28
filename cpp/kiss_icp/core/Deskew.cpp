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
#include "Deskew.hpp"

#include <tbb/parallel_for.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

namespace {
/// TODO(Nacho) Explain what is the very important meaning of this param
constexpr double mid_pose_timestamp{0.5};
}  // namespace

namespace kiss_icp {

/**
 * @brief DeSkewScan: it deskews a frame of 3D points based on their timestamps and a given motion delta.
 *                    This function corrects the points in the input frame to account for motion during the scan,
 *                    using the timestamps and a specified delta pose. The correction is performed in parallel
 *                    for efficiency. 
 * @param  frame:      The input frame of 3D points.
 * @param  timestamps: The timestamps associated with each 3D point.
 * @param  delta:      The pose change during the scan.
 * @return std::vector<Eigen::Vector3d> corrected_frame: The deskewed frame of 3D points.
 */
std::vector<Eigen::Vector3d> DeSkewScan(const std::vector<Eigen::Vector3d> &frame,
                                        const std::vector<double> &timestamps,
                                        const Sophus::SE3d &delta) {
    // Convert the delta pose to its logarithmic form
    const auto delta_pose = delta.log(); 
    // Prepare a vector for the corrected frame.
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    // Parallel loop to correct each point in the frame.
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        // Calculate the motion of the sensor for the timestamp of the current point.
        const auto motion = Sophus::SE3d::exp((timestamps[i] - mid_pose_timestamp) * delta_pose);
        // Apply the motion to the current point to get its corrected position.
        corrected_frame[i] = motion * frame[i];
    });
    return corrected_frame;
}
}  // namespace kiss_icp
