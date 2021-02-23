/**
 * @copyright Copyright (C) 2021  Miklas Riechmann, Ashwin Maliampurakal
 *
 *  This program is free software: you can blueistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include "inference_core.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "posture_in.h"  // Hardcoded image for testing
#include "pre_processor.h"

#define MODEL_INPUT_X 224
#define MODEL_INPUT_Y 224

void displayImage(cv::Mat originalImage, Inference::InferenceResults results) {
  std::string windowName = "Inference Results";

  cv::namedWindow(windowName);
  cv::Scalar blue(255, 0, 0);

  cv::circle(originalImage,
             cv::Point((int)(results.head_top.x * originalImage.cols),
                       (int)(results.head_top.y * originalImage.rows)),
             5, blue, -1);
  cv::circle(originalImage,
             cv::Point((int)(results.upper_neck.x * originalImage.cols),
                       (int)(results.upper_neck.y * originalImage.rows)),
             5, blue, -1);
  cv::circle(originalImage,
             cv::Point((int)(results.right_shoulder.x * originalImage.cols),
                       (int)(results.right_shoulder.y * originalImage.rows)),
             5, blue, -1);
  cv::circle(originalImage,
             cv::Point((int)(results.pelvis.x * originalImage.cols),
                       (int)(results.pelvis.y * originalImage.rows)),
             5, blue, -1);
  cv::circle(originalImage,
             cv::Point((int)(results.right_knee.x * originalImage.cols),
                       (int)(results.right_knee.y * originalImage.rows)),
             5, blue, -1);

  cv::imshow(windowName, originalImage);

  cv::waitKey(5);
}

int main(int argc, char const *argv[]) {
  // Setup the camera input
  cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    printf("Error opening camera");
    return 0;
  }

  // Initialise the PreProcessor and InferenceCore
  PreProcessing::PreProcessor preprocessor(MODEL_INPUT_X, MODEL_INPUT_Y);
  Inference::InferenceCore core("models/EfficientPoseRT_LITE.tflite",
                                MODEL_INPUT_X, MODEL_INPUT_Y);

  // Each frame
  cv::Mat frame;

  while (1) {
    // Try to read frame from camera input
    bool frameSuccess = cap.read(frame);

    if (!frameSuccess) {
      printf("Unable to get frame");
      return 0;
    }

    PreProcessing::PreProcessedImage preprocessed_image =
        preprocessor.run(frame);

    Inference::InferenceResults results = core.run(preprocessed_image);

    // Display image with detected points
    displayImage(frame, results);
  }
}
