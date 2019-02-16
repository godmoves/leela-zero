/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef TRTNETWORK_H_INCLUDED
#define TRTNETWORK_H_INCLUDED

#include "config.h"

#include <deque>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include <NvInfer.h>
#include <NvUffParser.h>
#include <cuda_runtime_api.h>

#include "NNCache.h"
#include "FastState.h"
#include "GameState.h"
#include "ForwardPipe.h"


class TRTNetwork {
public:
    static constexpr auto NUM_SYMMETRIES = 8;
    static constexpr auto IDENTITY_SYMMETRY = 0;
    enum Ensemble {
        DIRECT, RANDOM_SYMMETRY, AVERAGE
    };
    using PolicyVertexPair = std::pair<float,int>;
    using Netresult = NNCache::Netresult;

    // used in GTP.cpp and UCTNode.cpp to get NN result
    Netresult get_output(const GameState* const state,
                         const Ensemble ensemble,
                         const int symmetry = -1,
                         const bool read_cache = true,
                         const bool write_cache = true,
                         const bool force_selfcheck = false);

    static constexpr auto INPUT_MOVES = 8;
    static constexpr auto INPUT_CHANNELS = 2 * INPUT_MOVES + 2;

    // init the LZ network in Leela.cpp
    void initialize(int playouts, const std::string & weightsfile);

    // used for benchmark in GTP.cpp
    float benchmark_time(int centiseconds);
    void benchmark(const GameState * const state,
                   const int iterations = 1600);
    // show heatmap in GTP.cpp
    static void show_heatmap(const FastState * const state,
                             const Netresult & netres, const bool topmoves);

    // used in training.cpp to get training feature.
    static std::vector<float> gather_features(const GameState* const state,
                                              const int symmetry);
    // used in board/state to get symmetry
    static std::pair<int, int> get_symmetry(const std::pair<int, int>& vertex,
                                            const int symmetry,
                                            const int board_size = BOARD_SIZE);
    // get the network size
    size_t get_estimated_size();
    // something related to NN cache
    size_t get_estimated_cache_size();
    void nncache_resize(int max_count);

private:
    Netresult get_output_internal(const GameState* const state,
                                  const int symmetry, bool selfcheck = false);
    static void fill_input_plane_pair(const FullBoard& board,
                                      std::vector<float>::iterator black,
                                      std::vector<float>::iterator white,
                                      const int symmetry);
    bool probe_cache(const GameState* const state, TRTNetwork::Netresult& result);

    NNCache m_nncache;

    // nvidia brothers
    nvuffparser::IUffParser *m_parser;
    nvinfer1::IBuilder *m_builder;
    nvinfer1::INetworkDefinition *m_net;
    nvinfer1::ICudaEngine *m_engine;
    nvinfer1::IExecutionContext *m_context;
    std::vector<void *> m_cuda_buf;

    // TODO: fix gpu selection, not sure how leela do that
    int m_gpu{0};
    // TODO: not sure if we need this, seems this is used to store the network size
    size_t m_estimated_size{0};
    // TODO: fix elf network later
    bool m_value_head_not_stm{false}; // this is for elf network
    // TODO: fix batch size later
    int m_batch_size = 1;
};
#endif
