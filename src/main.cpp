/**
 * @copyright Copyright (C) 2021  Miklas Riechmann, Ashwin Maliampurakal
 *
 *  This program is free software: you can redistribute it and/or modify
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
#include "posture_in.h"  // Hardcoded image for testing
#include "pre_processor.h"

#define MODEL_INPUT_X 224
#define MODEL_INPUT_Y 224

int main(int argc, char const *argv[]) {
  PreProcessing::PreProcessor preprocessor(500, 500, 224, 224);

  PreProcessing::PreProcessedImage preprocessed_image =
      preprocessor.run(image.pixel_data);

  Inference::InferenceCore core("models/EfficientPoseRT_LITE.tflite",
                                MODEL_INPUT_X, MODEL_INPUT_Y);

  Inference::InferenceResults results = core.run(preprocessed_image);

  printf("%f, %f\n", results.head_top.x, results.head_top.y);
  printf("%f, %f\n", results.upper_neck.x, results.upper_neck.y);
  printf("%f, %f\n", results.right_shoulder.x, results.right_shoulder.y);
  printf("%f, %f\n", results.right_elbow.x, results.right_elbow.y);
  printf("%f, %f\n", results.right_wrist.x, results.right_wrist.y);
  printf("%f, %f\n", results.thorax.x, results.thorax.y);
  printf("%f, %f\n", results.left_shoulder.x, results.left_shoulder.y);
  printf("%f, %f\n", results.left_elbow.x, results.left_elbow.y);
  printf("%f, %f\n", results.left_wrist.x, results.left_wrist.y);
  printf("%f, %f\n", results.pelvis.x, results.pelvis.y);
  printf("%f, %f\n", results.right_hip.x, results.right_hip.y);
  printf("%f, %f\n", results.right_knee.x, results.right_knee.y);
  printf("%f, %f\n", results.right_ankle.x, results.right_ankle.y);
  printf("%f, %f\n", results.left_hip.x, results.left_hip.y);
  printf("%f, %f\n", results.left_knee.x, results.left_knee.y);
  printf("%f, %f\n", results.left_ankle.x, results.left_ankle.y);
}
