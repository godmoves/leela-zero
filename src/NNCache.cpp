/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Michael O and contributors

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
#include <functional>
#include <memory>

#include "NNCache.h"
#include "Utils.h"
#include "UCTSearch.h"
#include "GTP.h"

const int NNCache::MAX_CACHE_COUNT;
const int NNCache::MIN_CACHE_COUNT;
const size_t NNCache::ENTRY_SIZE;

NNCache::NNCache(int size) : m_size(size) {}

std::shared_ptr<NNCache::Entry> NNCache::lookup_and_insert(std::uint64_t hash, 
    bool insert, bool lookup, BackupData& bd, bool& ready) {
    // !lookup implies insert

    if (lookup) {
        ++m_lookups;
        auto iter = m_cache.find(hash);
        if (iter != m_cache.end()) {
            ++m_hits;
            auto result = iter->second;
            /*bool expected = false;
            if (result->ready.compare_exchange_strong(expected, true)) {
                result->num_mods++;
                result->backup_obligations.emplace_back(std::move(bd));
                result->ready = false;
                //expected = true;
                //if (!result->ready.compare_exchange_strong(expected, false)) { Utils::myprintf("Strange ready status!\n"); }
            }*/
            if (result->ready.test_and_set())
                ready = true;
            else {
                result->backup_obligations.emplace_back(std::move(bd));
                result->ready.clear();
            }
            return result;
        }
    }

    // skip-cache or not found
    if (insert) {
        // If the cache is too large, remove the oldest entry.
        if (m_order.size() >= m_size) {
            m_cache.erase(m_order.front());
            m_order.pop_front();
        }
        auto result = std::make_shared<Entry>();
        result->backup_obligations.emplace_back(std::move(bd));
        
        m_cache.emplace(hash, result);
        m_order.push_back(hash);
        ++m_inserts;
        return result;
    }
    else {
        return nullptr;
    }
}

void NNCache::resize(int size) {
    std::lock_guard<std::mutex> lk(m_mutex);
    m_size = size;
    while (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::clear() {
    std::lock_guard<std::mutex> lk(m_mutex);
    m_cache.clear();
    m_order.clear();
}

void NNCache::set_size_from_playouts(int max_playouts) {
    // cache hits are generally from last several moves so setting cache
    // size based on playouts increases the hit rate while balancing memory
    // usage for low playout instances. 150'000 cache entries is ~208 MiB
    constexpr auto num_cache_moves = 3;
    auto max_playouts_per_move =
        std::min(max_playouts,
                 UCTSearch::UNLIMITED_PLAYOUTS / num_cache_moves);
    auto max_size = num_cache_moves * max_playouts_per_move;
    max_size = std::min(MAX_CACHE_COUNT, std::max(MIN_CACHE_COUNT, max_size));
    m_cache.reserve(max_size);
    resize(max_size);
}

void NNCache::clear_stats() {
    m_hits = m_lookups = m_inserts = 0;
}

void NNCache::dump_stats() {
    Utils::myprintf(
        "NNCache: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %u size\n",
        m_hits, m_lookups, 100. * m_hits / (m_lookups + 1),
        m_inserts, m_cache.size());
}

size_t NNCache::get_estimated_size() {
    return m_order.size() * NNCache::ENTRY_SIZE;
}
