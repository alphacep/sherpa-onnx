// Copyright (c) 2023 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "speaker-model.h"
#include "onnx-utils.h"
#include "log.h"

namespace sherpa_onnx {

Ort::Env SpeakerModel::env_ = Ort::Env(
  ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
Ort::SessionOptions SpeakerModel::session_options_ = Ort::SessionOptions();

void SpeakerModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
}

#ifdef USE_GPU
void SpeakerModel::SetGpuDeviceId(int gpu_id) {
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
    session_options_, gpu_id));
}
#endif

SpeakerModel::SpeakerModel(void *model_data, size_t model_data_length) {
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  speaker_session_ = std::make_unique<Ort::Session>(env_, 
     model_data, model_data_length, session_options_);
  GetInputNames(speaker_session_.get(), &input_names_, &input_names_ptr_);
  GetOutputNames(speaker_session_.get(), &output_names_, &output_names_ptr_);
}

void SpeakerModel::ExtractEmbedding(
  const std::vector<std::vector<float>>& feats,
  std::vector<float>* embed) {
  Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // prepare onnx required data
  unsigned int num_frames = feats.size();
  unsigned int feat_dim = feats[0].size();
  std::vector<float> feats_onnx(num_frames * feat_dim, 0.0);
  for (size_t i = 0; i < num_frames; ++i) {
    for (size_t j = 0; j < feat_dim; ++j) {
      feats_onnx[i * feat_dim + j] = feats[i][j];
    }
  }
  // NOTE(cdliang): batchsize = 1
  const int64_t feats_shape[3] = {1, num_frames, feat_dim};
  Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
      memory_info, feats_onnx.data(), feats_onnx.size(), feats_shape, 3);
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(std::move(feats_ort));
  std::vector<Ort::Value> ort_outputs = speaker_session_->Run(
      Ort::RunOptions{nullptr}, input_names_ptr_.data(), inputs.data(),
      inputs.size(), output_names_ptr_.data(), output_names_.size());
  // output
  float* outputs = ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();

  embed->reserve(type_info.GetElementCount());
  for (size_t i = 0; i < type_info.GetElementCount(); ++i) {
    embed->emplace_back(outputs[i]);
  }
}

}  // namespace sherpa_onnx
