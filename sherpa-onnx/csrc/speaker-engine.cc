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

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <cmath>

#include "macros.h"
#include "log.h"
#include "speaker-engine.h"
#include "speaker-model.h"
#include "onnx-utils.h"

namespace sherpa_onnx {


#if __ANDROID_API__ >= 9
SpeakerEngine::SpeakerEngine(AAssetManager *mgr, const std::string& model_path) {
    auto buf = ReadFile(mgr, model_path);
    Init(buf.data(), buf.size(), 80, 16000, 256);
}
#endif

SpeakerEngine::SpeakerEngine(const std::string& model_path,
                             const int feat_dim,
                             const int sample_rate,
                             const int embedding_size) {
    auto buf = ReadFile(model_path);
    Init(buf.data(), buf.size(), feat_dim, sample_rate, embedding_size);
}

void SpeakerEngine::Init(void *model_data, size_t model_data_length,
                    const int feat_dim,
                    const int sample_rate,
                    const int embedding_size) {
  // NOTE(cdliang): default num_threads = 1
  const int kNumGemmThreads = 1;
  embedding_size_ = embedding_size;
  sample_rate_ = sample_rate;
  feature_config_ = std::make_shared<FeatureExtractorConfig>();
  feature_config_->feature_dim = feat_dim;
  feature_config_->sampling_rate = sample_rate;

  feature_extractor_ = std::make_shared<FeatureExtractor>(*feature_config_);
  SpeakerModel::InitEngineThreads(kNumGemmThreads);
  #ifdef USE_GPU
  // NOTE(cdliang): default gpu_id = 0
  SpeakerModel::SetGpuDeviceId(0);
  #endif
  model_ = std::make_shared<SpeakerModel>(model_data, model_data_length);
}

int SpeakerEngine::EmbeddingSize() {
  return embedding_size_;
}

void SpeakerEngine::ApplyMean(std::vector<std::vector<float>>* feat,
                              unsigned int feat_dim) {
  std::vector<float> mean(feat_dim, 0);
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), mean.begin(),
                   std::plus<>{});
  }
  std::transform(mean.begin(), mean.end(), mean.begin(),
                 [&](const float d) {return d / feat->size();});
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), i.begin(), std::minus<>{});
  }
}

void SpeakerEngine::ExtractFeature(int32_t sampling_rate, const float* data, int32_t data_size,
    std::vector<std::vector<float>>& feats) {

  if (data != nullptr) {
    feature_extractor_->AcceptWaveform(sampling_rate, data, data_size);
    feature_extractor_->InputFinished();
    int nframes = feature_extractor_->NumFramesReady();
    feats.resize(nframes);
    std::vector<float> row_feats = feature_extractor_->GetFrames(0, nframes);
    int feature_dim = feature_config_->feature_dim;
    for (int32_t i = 0; i < nframes; ++i) {
        for (int32_t j = 0; j < feature_dim; ++j) {
            feats[i].push_back(row_feats[i * feature_dim + j]);
        }
    }
  } else {
    SHERPA_ONNX_LOGE("Input is nullptr!");
  }
}

void SpeakerEngine::ExtractEmbedding(int32_t sampling_rate, const float *data, int32_t data_size,
                                     std::vector<float> *emb) {

  std::vector<std::vector<float>> feats;
  this->ExtractFeature(sampling_rate, data, data_size, feats);
  int feature_dim = feature_config_->feature_dim;
  this->ApplyMean(&feats, feature_dim);
  model_->ExtractEmbedding(feats, emb);
}

float SpeakerEngine::CosineSimilarity(const std::vector<float>& emb1,
                                      const std::vector<float>& emb2) {
  SHERPA_ONNX_CHECK_EQ(emb1.size(), emb2.size());
  float dot = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0);
  float emb1_sum = std::inner_product(emb1.begin(), emb1.end(),
                                      emb1.begin(), 0.0);
  float emb2_sum = std::inner_product(emb2.begin(), emb2.end(),
                                      emb2.begin(), 0.0);
  dot /= std::max(std::sqrt(emb1_sum) * std::sqrt(emb2_sum),
                  std::numeric_limits<float>::epsilon());
  return dot;
}

}  // namespace sherpa_onnx
