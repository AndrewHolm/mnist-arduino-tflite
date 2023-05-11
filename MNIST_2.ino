// Created by Andy Holm
// 5-01-23
//
// Description: Testing efficiency and efficacy of TFLu models running on edge devices. 
// Model trained in Google Colab and exported as new_model.h
// Using TensorFlowLite library for inference
#include <TensorFlowLite.h>
#include <Arduino.h>
#include <ArduinoBLE.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "new_model.h" // This is the header file containing the model's data (weights and biases)

//variable required by TFLu
//-----------------------------------------------------------------------------

// Model parsed by the TFLu parser
const tflite::Model* model = nullptr;

// The pointer to the interpreter
tflite::MicroInterpreter* interpreter = nullptr; 
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Input and output tensors
TfLiteTensor* tflu_i_tensor = nullptr;
TfLiteTensor* tflu_o_tensor = nullptr;

// Input shape and number of classes for MNIST
// (28 * 28 pixel image and 10 classes, 0 - 9)
const int kImageSize = 28;
const int kNumPixels = kImageSize * kImageSize;
const int kNumClasses = 10;

// Tensor arena size, memory required by interpreter TFLu does not use dynamic
//  allocation. Arena size is determined by model size through experiments.
const int kTensorArenaSize = 15* 1024;

// allocation of memory 
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Create a resolver to load the model's operators
static tflite::AllOpsResolver resolver;

// Setup
//-----------------------------------------------------------------------------

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);

  //load the TFLite model from the C-byte array
model = tflite::GetModel(model_tflite);

// make sure model schema version is compatible (from tflite website)
if (model->version() != TFLITE_SCHEMA_VERSION) {
  TF_LITE_REPORT_ERROR(error_reporter,
  "Model provided is schema version %d not equal not equal to supported version "
  "  %d. \n", model->version(), TFLITE_SCHEMA_VERSION);  
}

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("BLE initialization failed!");
    while (1);
  }
  BLE.setLocalName("MNIST Classifier");


  // Initialize TensorFlow Lite
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize); //, error_reporter
  interpreter = &static_interpreter;


  // Allocate memory for the model's input and output tensors
  interpreter->AllocateTensors();

  // Get pointers to the model's input and output tensors
  tflu_i_tensor = interpreter->input(0);
  tflu_o_tensor = interpreter->output(0);

  // create array to store the test image 
  float flat_image[kImageSize * kImageSize];

    // Flatten the test image (from new_model.h)
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            flat_image[i * IMAGE_SIZE + j] = zero_test[i][j];
        }
    }

  // Copy the test image to the input tensor
  for (int i = 0; i < kNumPixels; i++) {
    tflu_i_tensor->data.f[i] = flat_image[i];
  }


}
void loop(){

  // timer to test inference times
  unsigned long start_time = millis();  
  interpreter->Invoke();
  unsigned long end_time = millis();
  unsigned long inference_time = end_time - start_time;

  // Print the output (the predicted digit)
  float max_prob = 0.0;
  int max_index = 0;
  Serial.println(" ");

  for (int i = 0; i < kNumClasses; i++) {
    float prob = tflu_o_tensor->data.f[i];
    if (prob > max_prob) {
      max_prob = prob;
      max_index = i;
    }
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(prob);
  }
  
  // Display infernce time
  Serial.print("Inference time: ");
  Serial.println(inference_time);   
  delay(5000);
 

}

