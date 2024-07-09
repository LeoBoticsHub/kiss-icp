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
#include "Threshold.hpp"

#include <Eigen/Core>
#include <cmath>

namespace kiss_icp {

/**
 * @brief AdaptiveThreshold: defines the struct constructor with specified initial threshold,
 *                           minimum motion threshold, and maximum range. The initial sum of 
 *                           squared errors (SSE) is computed as the square of the initial threshold, 
 *                           and the number of samples is set to 1.
 * @param initial_threshold:    The initial value used to calculate the starting model sum of squared errors (SSE).
 * @param min_motion_threshold: The minimum motion threshold used for adaptive thresholding.
 * @param max_range:            The maximum range used for adaptive thresholding.
 */
AdaptiveThreshold::AdaptiveThreshold(double initial_threshold,
                                     double min_motion_threshold,
                                     double max_range)
    : min_motion_threshold_(min_motion_threshold),
      max_range_(max_range),
      model_sse_(initial_threshold * initial_threshold),
      num_samples_(1) {}



/**
 * @brief AdaptiveThreshold::UpdateModelDeviation     updates the current belief of the deviation from the prediction model.
 *
 * This method calculates the model error based on the provided current deviation,
 * which is a transformation represented by `Sophus::SE3d`. The model error is a
 * combination of the translational and rotational components of the deviation.
 * If the model error exceeds the minimum motion threshold, the method updates
 * the sum of squared errors (SSE) and increments the number of samples.
 *
 * @param current_deviation The current deviation from the prediction model, represented as a `Sophus::SE3d` transformation.
 */
void AdaptiveThreshold::UpdateModelDeviation(const Sophus::SE3d &current_deviation) {
    // lambda function to compute model error
    const double model_error = [&]() {
        const double theta = Eigen::AngleAxisd(current_deviation.rotationMatrix()).angle();
        const double delta_rot = 2.0 * max_range_ * std::sin(theta / 2.0);
        const double delta_trans = current_deviation.translation().norm();
        return delta_trans + delta_rot;
    }();
    if (model_error > min_motion_threshold_) {
        model_sse_ += model_error * model_error;
        num_samples_++;
    }
}

}  // namespace kiss_icp
