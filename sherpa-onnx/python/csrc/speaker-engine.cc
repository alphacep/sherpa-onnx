// sherpa-onnx/python/csrc/voice-activity-detector.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/speaker-engine.h"

#include <vector>

#include "sherpa-onnx/csrc/speaker-engine.h"

namespace sherpa_onnx {

void PybindSpeakerEngine(py::module *m) {
  using PyClass = SpeakerEngine;
  py::class_<PyClass>(*m, "SpeakerEngine")
      .def(py::init<const std::string&, int, int, int>(),
           py::arg("model"), py::arg("feat_dim") = 80,
           py::arg("sample_rate") = 16000,
           py::arg("embedding_size") = 256)
      .def(
          "extract_embedding",
         [](PyClass &self, int sampling_rate, py::array_t<float> samples) {
             std::vector<float> embed;
             self.ExtractEmbedding(sampling_rate, samples.data(), samples.size(), &embed);
             auto result = py::array_t<float>(embed.size());
             py::buffer_info buf = result.request();
             memcpy(buf.ptr, embed.data(), embed.size() * sizeof(float));
             return result;
          }, 
          py::arg("sampling_rate"),
          py::arg("samples"))
       .def("similarity",
         [](PyClass &self, py::array_t<float> emb1, py::array_t<float> emb2) {
             return 0.0;
          },
          py::arg("emb1"),
          py::arg("emb2"));
}

}  // namespace sherpa_onnx
