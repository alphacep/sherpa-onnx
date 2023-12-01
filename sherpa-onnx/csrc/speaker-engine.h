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

#ifndef SPEAKER_SPEAKER_ENGINE_H_
#define SPEAKER_SPEAKER_ENGINE_H_
#include <stdint.h>

#include <string>
#include <vector>
#include <memory>

#include "speaker-model.h"
#include "features.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

namespace sherpa_onnx {

class SpeakerEngine {
 public:
  explicit SpeakerEngine(const std::string& model_path,
                         const int feat_dim,
                         const int sample_rate,
                         const int embedding_size);

#if __ANDROID_API__ >= 9
  SpeakerEngine(AAssetManager *mgr, const std::string& model_path);
#endif

  void Init(void *model_data, size_t model_data_length,
            const int feat_dim,
            const int sample_rate,
            const int embedding_size);

  // return embedding_size
  int EmbeddingSize();
  // extract fbank
  void ExtractFeature(FeatureExtractor &extractor, int32_t sampling_rate, const float *data, int32_t data_size,
    std::vector<std::vector<float>>& feats);
  // extract embedding
  void ExtractEmbedding(int32_t sampling_rate, const float *data, int32_t data_size,
                        std::vector<float>* emb);

  float CosineSimilarity(const std::vector<float>& emb1,
                        const std::vector<float>& emb2);

 private:
  void ApplyMean(std::vector<std::vector<float>>* feats,
                 unsigned int feat_dim);
  std::shared_ptr<sherpa_onnx::SpeakerModel> model_ = nullptr;
  std::shared_ptr<FeatureExtractorConfig> feature_config_ = nullptr;
  int embedding_size_ = 0;
  int sample_rate_ = 16000;
};

}  // namespace sherpa_onnx

#endif  // SPEAKER_SPEAKER_ENGINE_H_
