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

#include "config.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <boost/utility.hpp>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>
#ifndef USE_BLAS
#include <Eigen/Dense>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#include "zlib.h"

#include "TRTNetwork.h"
#include "CPUPipe.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#include "UCTNode.h"
#endif
#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GameState.h"
#include "GTP.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"

using namespace Utils;
using namespace nvinfer1;
using namespace nvuffparser;


// Symmetry helper
static std::array<std::array<int, NUM_INTERSECTIONS>,
                  TRTNetwork::NUM_SYMMETRIES> symmetry_nn_idx_table;

// TensorRT logger
class Logger : public ILogger {
    void log(Severity severity, const char *msg) override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: myprintf("kINTERNAL_ERROR: %s", msg); break;
            case Severity::kERROR: myprintf("kERROR: %s", msg); break;
            case Severity::kWARNING: myprintf("kWARNING: %s", msg); break;
            // not log info
            // case Severity::kINFO: LOG(INFO) << msg; break;
        }
    }
} g_logger;

float TRTNetwork::benchmark_time(int centiseconds) {
    const auto cpus = cfg_num_threads;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    GameState state;
    state.init_game(BOARD_SIZE, 7.5);

    // As a sanity run, try one run with self check.
    // Isn't enough to guarantee correctness but better than nothing,
    // plus for large nets self-check takes a while (1~3 eval per second)
    get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, false, true, true);

    const Time start;
    for (auto i = size_t{0}; i < cpus; i++) {
        tg.add_task([this, &runcount, start, centiseconds, state]() {
            while (true) {
                runcount++;
                get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, false);
                const Time end;
                const auto elapsed = Time::timediff_centis(start, end);
                if (elapsed >= centiseconds) {
                    break;
                }
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_centis(start, end);
    return 100.0f * runcount.load() / elapsed;
}

void TRTNetwork::benchmark(const GameState* const state, const int iterations) {
    const auto cpus = cfg_num_threads;
    const Time start;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    for (auto i = size_t{0}; i < cpus; i++) {
        tg.add_task([this, &runcount, iterations, state]() {
            while (runcount < iterations) {
                runcount++;
                get_output(state, Ensemble::RANDOM_SYMMETRY, -1, false);
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_seconds(start, end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             runcount.load(), elapsed, int(runcount.load() / elapsed));
}

void TRTNetwork::initialize(int playouts, const std::string & weightsfile) {
    myprintf("Using TensorRT kernel on GPU %d\n.", m_gpu);
#ifdef USE_HALF
    myprintf("Using half precision.");
#endif

    // set which GPU to use
    cudaSetDevice(m_gpu);

    // Make a guess at a good size as long as the user doesn't
    // explicitly set a maximum memory usage.
    m_nncache.set_size_from_playouts(playouts);

    // Prepare symmetry table
    for (auto s = 0; s < NUM_SYMMETRIES; ++s) {
        for (auto v = 0; v < NUM_INTERSECTIONS; ++v) {
            const auto newvtx =
                get_symmetry({v % BOARD_SIZE, v / BOARD_SIZE}, s);
            symmetry_nn_idx_table[s][v] =
                (newvtx.second * BOARD_SIZE) + newvtx.first;
            assert(symmetry_nn_idx_table[s][v] >= 0
                   && symmetry_nn_idx_table[s][v] < NUM_INTERSECTIONS);
        }
    }

    // use uff model, this will configure the network automatically.
    // TODO: fix the input/output names
    const std::string input_name = "inputs";
    const std::string policy_name = "policy";
    const std::string value_name = "value";

    // create the parser
    m_parser = createUffParser();
    m_parser->registerInput(input_name.c_str(), DimsCHW(18, 19, 19), UffInputOrder::kNCHW);
    m_parser->registerOutput(policy_name.c_str());
    m_parser->registerOutput(value_name.c_str());

    m_builder = createInferBuilder(g_logger);
    m_net = m_builder->createNetwork();
    // parse the uff file
    m_parser->parse(weightsfile.c_str(), *m_net, DataType::kFLOAT);

    // build engine
    // set max batch size
    // TODO: batch size need fix later
    m_builder->setMaxBatchSize(m_batch_size);
    // set work space size (1 << 20 ~ 1 << 30 is known good)
    m_builder->setMaxWorkspaceSize(1 << 30);
    // set half precision
#ifdef USE_HALF
    m_builder->setHalf2Mode(true);
#endif

    m_engine = m_builder->buildCudaEngine(*m_net);
    if (m_engine == nullptr) {
        myprintf("load cuda engine error\n");
        return;
    }

    m_context = m_engine->createExecutionContext();

    for (int i = 0; i < m_engine->getNbBindings(); ++i) {
        auto dim = m_engine->getBindingDimensions(i);
        std::string dim_str = "(";
        int size = 1;
        for (int i = 0; i < dim.nbDims; ++i) {
            if (i)
                dim_str += ", ";
            dim_str += std::to_string(dim.d[i]);
            size *= dim.d[i];
        }
        dim_str += ")";
        myprintf("tensorrt binding: %s %s\n", m_engine->getBindingName(i), dim_str);

        void *buf;
        int ret = cudaMalloc(&buf, m_batch_size * size * sizeof(float));
        if (ret != 0) {
            myprintf("cuda malloc err %d\n", ret);
            return;
        }
        m_cuda_buf.push_back(buf);
    }
}

bool TRTNetwork::probe_cache(const GameState* const state,
                             TRTNetwork::Netresult& result) {
    if (m_nncache.lookup(state->board.get_hash(), result)) {
        return true;
    }
    // If we are not generating a self-play game, try to find
    // symmetries if we are in the early opening.
    if (!cfg_noise && !cfg_random_cnt
        && state->get_movenum()
           < (state->get_timecontrol().opening_moves(BOARD_SIZE) / 2)) {
        for (auto sym = 0; sym < TRTNetwork::NUM_SYMMETRIES; ++sym) {
            if (sym == TRTNetwork::IDENTITY_SYMMETRY) {
                continue;
            }
            const auto hash = state->get_symmetry_hash(sym);
            if (m_nncache.lookup(hash, result)) {
                decltype(result.policy) corrected_policy;
                for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; ++idx) {
                    const auto sym_idx = symmetry_nn_idx_table[sym][idx];
                    corrected_policy[idx] = result.policy[sym_idx];
                }
                result.policy = std::move(corrected_policy);
                return true;
            }
        }
    }
    return false;
}

TRTNetwork::Netresult TRTNetwork::get_output(
    const GameState* const state, const Ensemble ensemble, const int symmetry,
    const bool read_cache, const bool write_cache, const bool force_selfcheck) {
    Netresult result;
    if (state->board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    if (read_cache) {
        // See if we already have this in the cache.
        if (probe_cache(state, result)) {
            return result;
        }
    }

    if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
        result = get_output_internal(state, symmetry);
    } else if (ensemble == AVERAGE) {
        for (auto sym = 0; sym < NUM_SYMMETRIES; ++sym) {
            auto tmpresult = get_output_internal(state, sym);
            result.winrate +=
                tmpresult.winrate / static_cast<float>(NUM_SYMMETRIES);
            result.policy_pass +=
                tmpresult.policy_pass / static_cast<float>(NUM_SYMMETRIES);

            for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
                result.policy[idx] +=
                    tmpresult.policy[idx] / static_cast<float>(NUM_SYMMETRIES);
            }
        }
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        const auto rand_sym = Random::get_Rng().randfix<NUM_SYMMETRIES>();
        result = get_output_internal(state, rand_sym);
    }

    // v2 format (ELF Open Go) returns black value, not stm
    if (m_value_head_not_stm) {
        if (state->board.get_to_move() == FastBoard::WHITE) {
            result.winrate = 1.0f - result.winrate;
        }
    }

    if (write_cache) {
        // Insert result into cache.
        m_nncache.insert(state->board.get_hash(), result);
    }

    return result;
}

TRTNetwork::Netresult TRTNetwork::get_output_internal(
    const GameState* const state, const int symmetry, bool selfcheck) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;

    Netresult result;

    // TODO: we support batch size 1 only
    const auto inputs_data = gather_features(state, symmetry);

    int ret = cudaMemcpy(m_cuda_buf[0], inputs_data.data(),
                         inputs_data.size() * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (ret != 0) {
        myprintf("input cuda memcpy err %d", ret);
        return result;
    }

    m_context->execute(m_batch_size, m_cuda_buf.data());

    std::vector<float> value(1);
    std::vector<float> policy(NUM_INTERSECTIONS + 1);

    // calculate value
    ret = cudaMemcpy(value.data(), m_cuda_buf[1],
                   value.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (ret != 0) {
        myprintf("value cuda memcpy err %d", ret);
        return result;
    }

    // calculate policy
    ret = cudaMemcpy(policy.data(), m_cuda_buf[2],
                   policy.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (ret != 0) {
        myprintf("policy cuda memcpy err %d", ret);
        return result;
    }

    // Map value output range [-1..1] to [0..1] range
    const auto winrate = (1.0f + value[0]) / 2.0f;

    for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        result.policy[sym_idx] = policy[idx];
    }

    result.policy_pass = policy[NUM_INTERSECTIONS];
    result.winrate = winrate;

    return result;
}

void TRTNetwork::show_heatmap(const FastState* const state,
                           const Netresult& result,
                           const bool topmoves) {
    std::vector<std::string> display_map;
    std::string line;

    for (unsigned int y = 0; y < BOARD_SIZE; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE; x++) {
            auto policy = 0;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_state(vertex) == FastBoard::EMPTY) {
                policy = result.policy[y * BOARD_SIZE + x] * 1000;
            }

            line += boost::str(boost::format("%3d ") % policy);
        }

        display_map.push_back(line);
        line.clear();
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    const auto pass_policy = int(result.policy_pass * 1000);
    myprintf("pass: %d\n", pass_policy);
    myprintf("winrate: %f\n", result.winrate);

    if (topmoves) {
        std::vector<TRTNetwork::PolicyVertexPair> moves;
        for (auto i=0; i < NUM_INTERSECTIONS; i++) {
            const auto x = i % BOARD_SIZE;
            const auto y = i / BOARD_SIZE;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_state(vertex) == FastBoard::EMPTY) {
                moves.emplace_back(result.policy[i], vertex);
            }
        }
        moves.emplace_back(result.policy_pass, FastBoard::PASS);

        std::stable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        for (const auto& move : moves) {
            if (cum > 0.85f || move.first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                    move.first,
                    state->board.move_to_text(move.second).c_str());
            cum += move.first;
        }
    }
}

void TRTNetwork::fill_input_plane_pair(const FullBoard& board,
                                    std::vector<float>::iterator black,
                                    std::vector<float>::iterator white,
                                    const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto color = board.get_state(x, y);
        if (color == FastBoard::BLACK) {
            black[idx] = float(true);
        } else if (color == FastBoard::WHITE) {
            white[idx] = float(true);
        }
    }
}

std::vector<float> TRTNetwork::gather_features(const GameState* const state,
                                            const int symmetry) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    auto input_data = std::vector<float>(INPUT_CHANNELS * NUM_INTERSECTIONS);

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;

    const auto black_it = blacks_move ?
                          begin(input_data) :
                          begin(input_data) + INPUT_MOVES * NUM_INTERSECTIONS;
    const auto white_it = blacks_move ?
                          begin(input_data) + INPUT_MOVES * NUM_INTERSECTIONS :
                          begin(input_data);
    const auto to_move_it = blacks_move ?
        begin(input_data) + 2 * INPUT_MOVES * NUM_INTERSECTIONS :
        begin(input_data) + (2 * INPUT_MOVES + 1) * NUM_INTERSECTIONS;

    const auto moves = std::min<size_t>(state->get_movenum() + 1, INPUT_MOVES);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_board(h),
                              black_it + h * NUM_INTERSECTIONS,
                              white_it + h * NUM_INTERSECTIONS,
                              symmetry);
    }

    std::fill(to_move_it, to_move_it + NUM_INTERSECTIONS, float(true));

    return input_data;
}

std::pair<int, int> TRTNetwork::get_symmetry(const std::pair<int, int>& vertex,
                                          const int symmetry,
                                          const int board_size) {
    auto x = vertex.first;
    auto y = vertex.second;
    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    if ((symmetry & 4) != 0) {
        std::swap(x, y);
    }

    if ((symmetry & 2) != 0) {
        x = board_size - x - 1;
    }

    if ((symmetry & 1) != 0) {
        y = board_size - y - 1;
    }

    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry != IDENTITY_SYMMETRY || vertex == std::make_pair(x, y));
    return {x, y};
}

// TODO: seems we don't need this 
size_t TRTNetwork::get_estimated_size() {
  // get the size of the network
  if (m_estimated_size != 0) {
      return m_estimated_size;
  }
  return m_estimated_size;
}

size_t TRTNetwork::get_estimated_cache_size() {
  return m_nncache.get_estimated_size();
}

void TRTNetwork::nncache_resize(int max_count) {
  return m_nncache.resize(max_count);
}
