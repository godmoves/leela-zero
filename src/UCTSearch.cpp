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
#include "UCTSearch.h"

#include <boost/format.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <algorithm>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "TimeControl.h"
#include "Timing.h"
#include "Training.h"
#include "Utils.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif

using namespace Utils;

constexpr int UCTSearch::UNLIMITED_PLAYOUTS;

class OutputAnalysisData {
public:
    OutputAnalysisData(const std::string& move, int visits,
                       float winrate, float policy_prior, std::string pv,
                       float lcb, bool lcb_ratio_exceeded)
    : m_move(move), m_visits(visits), m_winrate(winrate),
      m_policy_prior(policy_prior), m_pv(pv), m_lcb(lcb),
      m_lcb_ratio_exceeded(lcb_ratio_exceeded) {};

    std::string get_info_string(int order) const {
        auto tmp = "info move " + m_move
                 + " visits " + std::to_string(m_visits)
                 + " winrate "
                 + std::to_string(static_cast<int>(m_winrate * 10000))
                 + " prior "
                 + std::to_string(static_cast<int>(m_policy_prior * 10000.0f))
                 + " lcb "
                 + std::to_string(static_cast<int>(std::max(0.0f, m_lcb) * 10000));
        if (order >= 0) {
            tmp += " order " + std::to_string(order);
        }
        tmp += " pv " + m_pv;
        return tmp;
    }

    friend bool operator<(const OutputAnalysisData& a,
                          const OutputAnalysisData& b) {
        if (a.m_lcb_ratio_exceeded && b.m_lcb_ratio_exceeded) {
            if (a.m_lcb != b.m_lcb) {
                return a.m_lcb < b.m_lcb;
            }
        }
        if (a.m_visits == b.m_visits) {
            return a.m_winrate < b.m_winrate;
        }
        return a.m_visits < b.m_visits;
    }

private:
    std::string m_move;
    int m_visits;
    float m_winrate;
    float m_policy_prior;
    std::string m_pv;
    float m_lcb;
    bool m_lcb_ratio_exceeded;
};


UCTSearch::UCTSearch(GameState& g, Network& network)
    : m_gtpstate(g), m_network(network), m_delete_futures(thread_pool), m_search_threads(thread_pool) {
    set_playout_limit(cfg_max_playouts);
    set_visit_limit(cfg_max_visits);

    m_rootstate = m_gtpstate;
    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    m_network.set_search(this);
}

UCTSearch::~UCTSearch() {
    m_network.destruct();
    m_delete_futures.wait_all();
}

bool UCTSearch::advance_to_new_rootstate(std::list<UCTNode*>& to_delete) {

    if (!m_root || !m_last_rootstate) {
        // No current state
        return false;
    }

    if (m_rootstate.get_komi() != m_last_rootstate->get_komi()) {
        return false;
    }

    auto depth =
        int(m_rootstate.get_movenum() - m_last_rootstate->get_movenum());

    if (depth < 0) {
        return false;
    }


    auto test = std::make_unique<GameState>(m_rootstate);
    for (auto i = 0; i < depth; i++) {
        test->undo_move();
    }

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // m_rootstate and m_last_rootstate don't match
        return false;
    }

    // why necessary?
    // Make sure that the nodes we destroyed the previous move are
    // in fact destroyed
    /*while (!m_delete_futures.empty()) {
        m_delete_futures.front().wait_all();
        m_delete_futures.pop_front();
    }*/

    myprintf("entered going forward in tree\n");
    // Try to replay moves advancing m_root
    for (auto i = 0; i < depth; i++) {
        test->forward_move();
        const auto move = test->get_last_move();

        auto oldroot = std::move(m_root);
        m_root = oldroot->find_child(move);

        // Lazy tree destruction.  Instead of calling the destructor of the
        // old root node on the main thread, send the old root to a separate
        // thread and destroy it from the child thread.  This will save a
        // bit of time when dealing with large trees.
        to_delete.emplace_back(oldroot.release());

        if (!m_root) {
            myprintf("tree hasn't expanded this far\n");
            // Tree hasn't been expanded this far
            return false;
        }
        m_last_rootstate->play_move(move);
    }

    assert(m_rootstate.get_movenum() == m_last_rootstate->get_movenum());

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // Can happen if user plays multiple moves in a row by same player
        return false;
    }

    return true;
}

void UCTSearch::acquire_reader() {
    while (true) {
        if (m_root_lock >= 128) continue;
        if (m_root_lock.fetch_add(1) >= 128) {
            --m_root_lock;
            continue;
        }
        return;
    }
}

void UCTSearch::release_reader() {
    --m_root_lock;
}

void UCTSearch::acquire_writer() {
    // only the main thread may attempt this
    m_root_lock += 128;
    while (m_root_lock != 128) {}
}

void UCTSearch::release_writer() {
    m_root_lock -= 128;
}

void UCTSearch::update_root() {

#ifndef NDEBUG
    //auto start_nodes = m_root->count_nodes_and_clear_expand_state();
#endif

    m_network.clear_stats();

    acquire_writer();
    m_rootstate = m_gtpstate;

    std::list<UCTNode*> to_delete;
    if (!advance_to_new_rootstate(to_delete) || !m_root) {
        if (m_root) {
            to_delete.emplace_back(m_root.release());
        }
        m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    }
    myprintf("to delete: %d nodes\n", to_delete.size());
    if (m_pending_counter && !to_delete.empty()) {
        //ThreadGroup tg(thread_pool);
        m_delete_futures.add_task([](std::list<UCTNode*> to_delete, std::atomic<int>* pending_counter) {
            auto root = to_delete.front();
            to_delete.pop_front();
            ThreadGroup tg0(thread_pool);
            while (*pending_counter > 0) {
                myprintf("pending count: %d\n", pending_counter->load());
                myprintf("root vl: %d\n", root->m_virtual_loss.load());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            myprintf("root virtual loss at deletion: %d\n", root->m_virtual_loss.load());
            for (auto node : to_delete) {
                tg0.add_task([node]() { delete node; });
            }
            delete root;
            //tg0.wait_all();
            delete pending_counter;
            myprintf("deleted!\n");
        }, to_delete, m_pending_counter);
        //m_delete_futures.push_back(std::move(tg));
    }

    // Clear last_rootstate to prevent accidental use.
    m_last_rootstate.reset(nullptr);

    // Check how big our search tree (reused or new) is.
    // m_nodes = m_root->count_nodes_and_clear_expand_state();

#ifndef NDEBUG
    /*if (m_nodes > 0) {
        myprintf("update_root, %d -> %d nodes (%.1f%% reused)\n",
            start_nodes, m_nodes.load(), 100.0 * m_nodes.load() / start_nodes);
    }*/
#endif

    // Definition of m_playouts is playouts per search call.
    // So reset this count now.
    // However these aren't well protected by m_root_lock.
    m_playouts = 0;
    m_positions = 0;
#ifdef ACCUM_DEBUG
    failed_simulations = 0;
    max_pending_backups = 0;
    max_pending_w_mult = 0;
    max_vl = 0;
    max_leaf_vl = 0;
    max_pending_netresults = 0;
    min_pending_netresults = INT_MAX;
#endif
    // This is protected.
    m_pending_counter = new std::atomic<int>(0);
    release_writer();

    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_run = true;
        m_root_prepared = false;
    }
    m_cv.notify_all();

    while (!m_root_prepared) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void UCTSearch::backup(BackupData& bd, Netresult_ptr netresult) {
    auto& path = bd.path;
    auto node = path.back().node;
    auto is_root = bd.path.size() == 1;
    auto min_psa_ratio = is_root ? 0.0 : UCTNode::get_min_psa_ratio();
    auto first_visit = node->get_visits() == 0.0;
    
    node->create_children(netresult->result, bd.symmetry, *bd.state, min_psa_ratio);
    auto vl = node->m_accumulated_vl.exchange(0);
    if (first_visit) {
        auto eval = netresult->result.winrate;
        bd.eval = (bd.state->get_to_move() == FastBoard::BLACK ? eval : 1.0f - eval);
        // writer is responsible for removing all virtual losses injected by concurrent readers
        node->update(bd.eval, vl, 1.0f, path.back().factor);
    }
    if (is_root && !m_root_prepared) {
        // create a sorted list of legal moves (make sure we
        // play something legal and decent even in time trouble)
        node->prepare_root_node(*bd.state);
        m_root_prepared = true;
    }
    node->release_writer();

    if (first_visit) {
        backup(bd, vl);
    }
    else {
        failed_simulation(bd, vl);
    }
    ++m_playouts;
#ifdef ACCUM_DEBUG
    if (!is_root && first_visit) {
        max_leaf_vl = std::max(max_leaf_vl.load(), vl);
        --pending_backups;
        pending_w_mult -= vl;
    }
#endif
}

void UCTSearch::backup(BackupData& bd, uint16_t vl) {
    auto& path = bd.path;
    auto factor = path.back().factor;
    for (auto nf = path.rbegin() + 1; nf != path.rend(); ++nf) {
        auto sel_factor = factor * nf->factor;
        nf->node->update(bd.eval, vl, factor, sel_factor);
        factor = sel_factor;
    }
    --(*bd.pending_counter);
}

void UCTSearch::failed_simulation(BackupData& bd, uint16_t vl, bool incr) {
    auto& path = bd.path;
    for (auto nf = path.rbegin(); nf != path.rend(); ++nf) {
        if (incr) {
            nf->node->virtual_loss(vl);
        }
        else {
            nf->node->virtual_loss_undo(vl);
        }
    }
    --(*bd.pending_counter);
}

float eval_from_score(float board_score) {
    if (board_score > 0.0f) { return 1.0f; }
    else if (board_score < 0.0f) { return 0.0f; }
    else { return 0.5f; }
}

void UCTSearch::play_simulation(std::unique_ptr<GameState> currstate,
                                        UCTNode* node,
                                        std::atomic<int>* pending_counter,
                                        int gnum, int i) {
    auto factor = 1.0f;
    BackupData bd;
    bd.pending_counter = pending_counter;
    bool is_root = true;
    while (true) {
        bd.path.emplace_back(node, factor);

        // end of game
        if (currstate->get_passes() >= 2) {
            bd.eval = eval_from_score(currstate->final_score());
            node->update(bd.eval, 0, 1.0f, factor);
            backup(bd, 1);
            return;
        }
#ifdef ACCUM_DEBUG
        if(!is_root) max_vl = std::max(max_vl.load(), uint16_t(node->m_virtual_loss + 1));
#endif
        switch (node->get_action(is_root && !m_root_prepared)) {

        case UCTNode::WRITE: // expand the node
#ifdef ACCUM_DEBUG
            ++pending_backups;
            ++pending_w_mult;
            max_pending_backups = std::max(pending_backups.load(), max_pending_backups.load());
            if (!is_root) { max_pending_w_mult = std::max(pending_w_mult.load(), max_pending_w_mult.load()); }
#endif
            bd.state = std::move(currstate);
            m_network.get_output0(gnum, i, bd, Network::Ensemble::RANDOM_SYMMETRY);
            return;

        case UCTNode::FAIL: // virtual loss accumulated, return
#ifdef ACCUM_DEBUG
            if (!is_root) {
                ++pending_w_mult;
                max_pending_w_mult = std::max(pending_w_mult.load(), max_pending_w_mult.load());
                ++failed_simulations;
            }
#endif
            //failed_simulation(bd, node->m_virtual_loss, true);
            --(*pending_counter);
            return;

        case UCTNode::READ: // select a child 
        {
            auto child_factor = node->uct_select_child(currstate->get_to_move(), is_root);
            node->release_reader();
            auto new_node = child_factor.first;
            ////if (is_root) { m_debug_string += currstate->move_to_text(new_node->get_move()) + " \n"; } //
            //   if (is_root) { myprintf("%s\n", currstate->move_to_text(new_node->get_move()).c_str()); } //
            if (new_node != nullptr) {
                node = new_node;
                factor = child_factor.second;
                auto move = node->get_move();
                currstate->play_move(move);
                if (move != FastBoard::PASS && currstate->superko()) {
                    node->invalidate();
                    failed_simulation(bd, 1);
                    return;
                }
                break;
            }
            else {
                myprintf("All children are invalidated! ");
#ifdef LOCK_DEBUG
                myprintf("%d, %d, %d", node->get_children().size(), node->m_lock.load(), is_root);
#endif
                myprintf("\n");
                // backup, instead of expand further..
            }
        }

        case UCTNode::BACKUP:
            bd.eval = node->get_net_eval(FastBoard::BLACK);
            // print the sequence of moves from bd.path ...
            node->update(bd.eval, 1, 1.0f, factor);
            backup(bd, 1);
            return;
        }
        is_root = false;
    }
}

void UCTSearch::dump_stats(FastState & state, UCTNode & parent) {
    if (cfg_quiet || !parent.has_children()) {
        return;
    }

    const int color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children()) {
        max_visits = std::max(max_visits, (int)node->get_visits());
    }

    // sort children, put best move on top
    parent.sort_children(color, cfg_lcb_min_visit_ratio * max_visits);

    parent.acquire_reader();
    if (parent.get_first_child()->first_visit()) {
        parent.release_reader();
        return;
    }

    int movecount = 0;
    for (const auto& node : parent.get_children()) {
        // Always display at least two moves. In the case there is
        // only one move searched the user could get an idea why.
        if (++movecount > 2 && !node->get_visits()) break;

        auto move = state.move_to_text(node->get_move());
        auto tmpstate = FastState{state};
        tmpstate.play_move(node->get_move());
        auto pv = move + " " + get_pv(tmpstate, *node);

        myprintf("%4s -> %7d (V: %5.2f%%) (LCB: %5.2f%%) (N: %5.2f%%) PV: %s\n",
            move.c_str(),
            (int)node->get_visits(),
            node->get_visits() ? node->get_raw_eval(color)*100.0f : 0.0f,
            std::max(0.0f, node->get_eval_lcb(color) * 100.0f),
            node->get_policy() * 100.0f,
            pv.c_str());
    }
    //tree_stats(parent);
    parent.release_reader();
}

void UCTSearch::output_analysis(FastState & state, UCTNode & parent) {
    // We need to make a copy of the data before sorting
    auto sortable_data = std::vector<OutputAnalysisData>();

    parent.acquire_reader();
    if (!parent.has_children()) {
        parent.release_reader();
        return;
    }

    const auto color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children()) {
        max_visits = std::max(max_visits, (int)node->get_visits());
    }

    for (const auto& node : parent.get_children()) {
        // Send only variations with visits, unless more moves were
        // requested explicitly.
        if (!node->get_visits()
            && sortable_data.size() >= cfg_analyze_tags.post_move_count()) {
            continue;
        }
        auto move = state.move_to_text(node->get_move());
        auto tmpstate = FastState{state};
        tmpstate.play_move(node->get_move());
        auto rest_of_pv = get_pv(tmpstate, *node);
        auto pv = move + (rest_of_pv.empty() ? "" : " " + rest_of_pv);
        auto move_eval = node->get_visits() ? node->get_raw_eval(color) : 0.0f;
        auto policy = node->get_policy();
        auto lcb = node->get_eval_lcb(color);
        auto visits = node->get_visits();
        // Need at least 2 visits for valid LCB.
        auto lcb_ratio_exceeded = visits > 2 &&
            visits > max_visits * cfg_lcb_min_visit_ratio;
        // Store data in array
        sortable_data.emplace_back(move, visits,
                                   move_eval, policy, pv, lcb, lcb_ratio_exceeded);
    }
    parent.release_reader();
    // Sort array to decide order
    std::stable_sort(rbegin(sortable_data), rend(sortable_data));

    auto i = 0;
    // Output analysis data in gtp stream
    for (const auto& node : sortable_data) {
        if (i > 0) {
            gtp_printf_raw(" ");
        }
        gtp_printf_raw(node.get_info_string(i).c_str());
        i++;
    }
    gtp_printf_raw("\n");
}

//should be abandoned? occupy reader for too long ...
void UCTSearch::tree_stats(UCTNode& node) {
    size_t nodes = 0;
    size_t non_leaf_nodes = 0;
    size_t depth_sum = 0;
    size_t max_depth = 0;
    size_t children_count = 0;

    std::function<void(UCTNode& node, size_t)> traverse =
          [&](UCTNode& node, size_t depth) {
        nodes += 1;
        non_leaf_nodes += node.get_visits() > 1;
        depth_sum += depth;
        if (depth > max_depth) max_depth = depth;

        node.acquire_reader();
        for (const auto& child : node.get_children()) {
            if (child.get_visits() > 0) {
                children_count += 1;
                traverse(*(child.get()), depth+1);
            } else {
                nodes += 1;
                depth_sum += depth+1;
                if (depth >= max_depth) max_depth = depth+1;
            }
        }
        node.release_reader();
    };

    traverse(node, 0);

    if (nodes > 0) {
        myprintf("%.1f average depth, %d max depth\n",
                 (1.0f*depth_sum) / nodes, max_depth);
        myprintf("%d non leaf nodes, %.2f average children\n",
                 non_leaf_nodes, (1.0f*children_count) / non_leaf_nodes);
    }
}

bool UCTSearch::should_resign(passflag_t passflag, float besteval) {
    if (passflag & UCTSearch::NORESIGN) {
        // resign not allowed
        return false;
    }

    if (cfg_resignpct == 0) {
        // resign not allowed
        return false;
    }

    const size_t num_intersections = m_rootstate.board.get_boardsize()
                                   * m_rootstate.board.get_boardsize();
    const auto move_threshold = num_intersections / 4;
    const auto movenum = m_rootstate.get_movenum();
    if (movenum <= move_threshold) {
        // too early in game to resign
        return false;
    }

    const auto color = m_rootstate.board.get_to_move();

    const auto is_default_cfg_resign = cfg_resignpct < 0;
    const auto resign_threshold =
        0.01f * (is_default_cfg_resign ? 10 : cfg_resignpct);
    if (besteval > resign_threshold) {
        // eval > cfg_resign
        return false;
    }

    if ((m_rootstate.get_handicap() > 0)
            && (color == FastBoard::WHITE)
            && is_default_cfg_resign) {
        const auto handicap_resign_threshold =
            resign_threshold / (1 + m_rootstate.get_handicap());

        // Blend the thresholds for the first ~215 moves.
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        auto blended_resign_threshold = blend_ratio * resign_threshold
            + (1 - blend_ratio) * handicap_resign_threshold;
        if (besteval > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    if (!m_rootstate.is_move_legal(color, FastBoard::RESIGN)) {
        return false;
    }

    return true;
}

int UCTSearch::get_best_move(passflag_t passflag) {
    int color = m_rootstate.board.get_to_move();

    auto max_visits = 0;
    for (const auto& node : m_root->get_children()) {
        max_visits = std::max(max_visits, (int)node->get_visits());
    }

    // Make sure best is first
    m_root->sort_children(color,  cfg_lcb_min_visit_ratio * max_visits);

    // Check whether to randomize the best move proportional
    // to the playout counts, early game only.
    auto movenum = int(m_rootstate.get_movenum());
    if (movenum < cfg_random_cnt) {
        m_root->randomize_first_proportionally();
    }

    auto first_child = m_root->get_first_child();
    assert(first_child != nullptr);

    auto bestmove = first_child->get_move();
    auto besteval = first_child->first_visit() ? 0.5f : first_child->get_raw_eval(color);

    // do we want to fiddle with the best move because of the rule set?
    if (passflag & UCTSearch::NOPASS) {
        // were we going to pass?
        if (bestmove == FastBoard::PASS) {
            UCTNode * nopass = m_root->get_nopass_child(m_rootstate);

            if (nopass != nullptr) {
                myprintf("Preferring not to pass.\n");
                bestmove = nopass->get_move();
                if (nopass->first_visit()) {
                    besteval = 1.0f;
                } else {
                    besteval = nopass->get_raw_eval(color);
                }
            } else {
                myprintf("Pass is the only acceptable move.\n");
            }
        }
    } else if (!cfg_dumbpass) {
        const auto relative_score =
            (color == FastBoard::BLACK ? 1 : -1) * m_rootstate.final_score();
        if (bestmove == FastBoard::PASS) {
            // Either by forcing or coincidence passing is
            // on top...check whether passing loses instantly
            // do full count including dead stones.
            // In a reinforcement learning setup, it is possible for the
            // network to learn that, after passing in the tree, the two last
            // positions are identical, and this means the position is only won
            // if there are no dead stones in our own territory (because we use
            // Trump-Taylor scoring there). So strictly speaking, the next
            // heuristic isn't required for a pure RL network, and we have
            // a commandline option to disable the behavior during learning.
            // On the other hand, with a supervised learning setup, we fully
            // expect that the engine will pass out anything that looks like
            // a finished game even with dead stones on the board (because the
            // training games were using scoring with dead stone removal).
            // So in order to play games with a SL network, we need this
            // heuristic so the engine can "clean up" the board. It will still
            // only clean up the bare necessity to win. For full dead stone
            // removal, kgs-genmove_cleanup and the NOPASS mode must be used.

            // Do we lose by passing?
            if (relative_score < 0.0f) {
                myprintf("Passing loses :-(\n");
                // Find a valid non-pass move.
                UCTNode * nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr) {
                    myprintf("Avoiding pass because it loses.\n");
                    bestmove = nopass->get_move();
                    if (nopass->first_visit()) {
                        besteval = 1.0f;
                    } else {
                        besteval = nopass->get_raw_eval(color);
                    }
                } else {
                    myprintf("No alternative to passing.\n");
                }
            } else if (relative_score > 0.0f) {
                myprintf("Passing wins :-)\n");
            } else {
                myprintf("Passing draws :-|\n");
                // Find a valid non-pass move.
                const auto nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr && !nopass->first_visit()) {
                    const auto nopass_eval = nopass->get_raw_eval(color);
                    if (nopass_eval > 0.5f) {
                        myprintf("Avoiding pass because there could be a winning alternative.\n");
                        bestmove = nopass->get_move();
                        besteval = nopass_eval;
                    }
                }
                if (bestmove == FastBoard::PASS) {
                    myprintf("No seemingly better alternative to passing.\n");
                }
            }
        } else if (m_rootstate.get_last_move() == FastBoard::PASS) {
            // Opponents last move was passing.
            // We didn't consider passing. Should we have and
            // end the game immediately?

            if (!m_rootstate.is_move_legal(color, FastBoard::PASS)) {
                myprintf("Passing is forbidden, I'll play on.\n");
            // do we lose by passing?
            } else if (relative_score < 0.0f) {
                myprintf("Passing loses, I'll play on.\n");
            } else if (relative_score > 0.0f) {
                myprintf("Passing wins, I'll pass out.\n");
                bestmove = FastBoard::PASS;
            } else {
                myprintf("Passing draws, make it depend on evaluation.\n");
                if (besteval < 0.5f) {
                    bestmove = FastBoard::PASS;
                }
            }
        }
    }

    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
        if (should_resign(passflag, besteval)) {
            myprintf("Eval (%.2f%%) looks bad. Resigning.\n",
                     100.0f * besteval);
            bestmove = FastBoard::RESIGN;
        }
    }

    return bestmove;
}

std::string UCTSearch::get_pv(FastState & state, UCTNode& parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    // now can just acquire_reader, but may not be worth doing it..
    if (parent.expandable()) {
        // Not fully expanded. This means someone could expand
        // the node while we want to traverse the children.
        // Avoid the race conditions and don't go through the rabbit hole
        // of trying to print things from this node.
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.get_to_move(), m_run);
    if (best_child.first_visit()) {
        return std::string();
    }
    auto best_move = best_child.get_move();
    auto res = state.move_to_text(best_move);

    state.play_move(best_move);

    auto next = get_pv(state, best_child);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    return res;
}

std::string UCTSearch::get_analysis(int playouts) {
    FastState tempstate = m_rootstate;
    int color = tempstate.board.get_to_move();

    auto pvstring = get_pv(tempstate, *m_root);
    float winrate = 100.0f * m_root->get_raw_eval(color);
    return str(boost::format("Playouts: %d, Visits: %d, Positions: %d, Win: %5.2f%%, PV: %s")
        % playouts % (int)m_root->get_visits() % m_positions.load() % winrate % pvstring.c_str());
}

bool UCTSearch::is_running() const {
    return m_run && UCTNodePointer::get_tree_size() < cfg_max_tree_size;
}

int UCTSearch::est_playouts_left(int elapsed_centis, int time_for_move) const {
    auto playouts = m_playouts.load();
    const auto playouts_left =
        std::max(0, std::min(m_maxplayouts - playouts,
                             m_maxvisits - (int)m_root->get_visits()));

    // Wait for at least 1 second and 100 playouts
    // so we get a reliable playout_rate.
    if (elapsed_centis < 100 || playouts < 100) {
        return playouts_left;
    }
    const auto playout_rate = 1.0f * playouts / elapsed_centis;
    const auto time_left = std::max(0, time_for_move - elapsed_centis);
    return std::min(playouts_left,
                    static_cast<int>(std::ceil(playout_rate * time_left)));
}

size_t UCTSearch::prune_noncontenders(int color, int elapsed_centis, int time_for_move, bool prune) {
    auto lcb_max = 0.0f;
    auto Nfirst = 0;

    m_root->acquire_reader();
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            const auto visits = (int)node->get_visits();
            if (visits > 0) {
                lcb_max = std::max(lcb_max, node->get_eval_lcb(color));
            }
            Nfirst = std::max(Nfirst, visits);
        }
    }
    const auto min_required_visits =
        Nfirst - est_playouts_left(elapsed_centis, time_for_move);
    auto pruned_nodes = size_t{0};
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            const auto visits = node->get_visits();
            const auto has_enough_visits =
                visits >= min_required_visits;
            // Avoid pruning moves that could have the best lower confidence
            // bound.
            const auto high_winrate = visits > 0 ?
                node->get_raw_eval(color) >= lcb_max : false;
            const auto prune_this_node = !(has_enough_visits || high_winrate);

            if (prune) {
                node->set_active(!prune_this_node);
            }
            if (prune_this_node) {
                ++pruned_nodes;
            }
        }
    }
    m_root->release_reader();
    assert(pruned_nodes < m_root->get_children().size());
    return pruned_nodes;
}

bool UCTSearch::have_alternate_moves(int elapsed_centis, int time_for_move) {
    if (cfg_timemanage == TimeManagement::OFF) {
        return true;
    }
    auto my_color = m_rootstate.get_to_move();
    // For self play use. Disables pruning of non-contenders to not bias the training data.
    auto prune = cfg_timemanage != TimeManagement::NO_PRUNING;

    if (m_root->get_children().size() == 0) { return true; }
    m_root->acquire_reader();
    auto pruned = prune_noncontenders(my_color, elapsed_centis, time_for_move, prune);
    auto size = m_root->get_children().size();
    m_root->release_reader();
    if (pruned < size - 1) {
        return true;
    }
    // If we cannot save up time anyway, use all of it. This
    // behavior can be overruled by setting "fast" time management,
    // which will cause Leela to quickly respond to obvious/forced moves.
    // That comes at the cost of some playing strength as she now cannot
    // think ahead about her next moves in the remaining time.
    auto tc = m_rootstate.get_timecontrol();
    if (!tc.can_accumulate_time(my_color)
        || m_maxplayouts < UCTSearch::UNLIMITED_PLAYOUTS) {
        if (cfg_timemanage != TimeManagement::FAST) {
            return true;
        }
    }
    // In a timed search we will essentially always exit because
    // the remaining time is too short to let another move win, so
    // avoid spamming this message every move. We'll print it if we
    // save at least half a second.
    if (time_for_move - elapsed_centis > 50) {
        myprintf("%.1fs left, stopping early.\n",
                    (time_for_move - elapsed_centis) / 100.0f);
    }
    return false;
}

bool UCTSearch::stop_thinking(int elapsed_centis, int time_for_move) const {
    return m_playouts >= m_maxplayouts
           || m_root->get_visits() >= m_maxvisits
           || elapsed_centis >= time_for_move;
}

void UCTSearch::search(int gnum, int i) {
    if (is_running() || !m_root_prepared) {
        acquire_reader();
        auto rootstate = std::make_unique<GameState>(m_rootstate);
        auto root = m_root.get();
        auto pending_counter = m_pending_counter;
        ++(*pending_counter);
        release_reader();
        play_simulation(std::move(rootstate), root, pending_counter, gnum, i);
        return;
    }
    std::unique_lock<std::mutex> lk(m_mutex);
    if (m_root_prepared) m_run = false;
    if (!m_run) m_cv.wait(lk, [this]() { return m_run.load(); });
}

void UCTSearch::increment_playouts() {
    m_playouts++;
}

int UCTSearch::think(int color, passflag_t passflag) {
    // Start counting time for us
    m_gtpstate.start_clock(color);

    // set up timing info
    Time start;

    update_root();
    // set side to move
    m_rootstate.board.set_to_move(color);

    auto time_for_move =
        m_rootstate.get_timecontrol().max_time_for_move(
            m_rootstate.board.get_boardsize(),
            color, m_rootstate.get_movenum());

    myprintf("Thinking at most %.1f seconds...\n", time_for_move/100.0f);
    auto keeprunning = true;
    auto last_update = 0;
    auto last_output = 0;
    do {
        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);
        std::this_thread::sleep_for(std::chrono::milliseconds(
            std::min(std::min(cfg_analyze_tags.interval_centis() - (elapsed_centis - last_output),
                250 - (elapsed_centis - last_update)), time_for_move - elapsed_centis) * 10));
        Time elapsed0;
        elapsed_centis = Time::timediff_centis(start, elapsed0);

        if (cfg_analyze_tags.interval_centis() &&
            elapsed_centis - last_output > cfg_analyze_tags.interval_centis()) {
            last_output = elapsed_centis;
            output_analysis(m_rootstate, *m_root);
        }

        // output some stats every few seconds
        // check if we should still search
        if (!cfg_quiet && elapsed_centis - last_update > 250) {
            last_update = elapsed_centis;
            myprintf("%s\n", get_analysis(m_playouts.load()).c_str());
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(elapsed_centis, time_for_move);
        keeprunning &= have_alternate_moves(elapsed_centis, time_for_move);
    } while (keeprunning);

    m_run = is_running() && !stop_thinking(0, 1);
    // if --noponder, set m_run = false in GTP.cpp

    m_root->acquire_reader();
    // reactivate all pruned root children

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centis() && last_output == 0) {
        output_analysis(m_rootstate, *m_root);
    }

    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }
    m_root->release_reader();

    m_gtpstate.stop_clock(color);
    if (!m_root->has_children()) {
        return FastBoard::PASS;
    }

    // Display search info.
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);
    Training::record(m_network, m_rootstate, *m_root);

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    myprintf("sizeof(UCTNode) is %d\n", sizeof(UCTNode));
    myprintf("sizeof(UCTNodePointer) is %d\n", sizeof(UCTNodePointer));
    if (elapsed_centis+1 > 0) {
        myprintf("%7.2f visits, %u nodes, %u inflated, %d playouts, %.0f n/s, %.0f pos/s\n\n",
                 m_root->get_visits(),
            UCTNodePointer::m_nodes.load(), UCTNodePointer::m_inflated_nodes.load(),
                 m_playouts.load(),
                 (m_playouts * 100.0) / (elapsed_centis+1),
                 (m_positions * 100.0) / (elapsed_centis+1));

        m_network.dump_stats();
#ifdef ACCUM_DEBUG
        myprintf("failed simulations: %u\n", failed_simulations.load());
        myprintf("max leaf vl multiplicity: %u\n", max_leaf_vl.load());
        myprintf("max vl multiplicity: %u\n", max_vl.load());
        myprintf("max pending backups: %u\n", max_pending_backups.load());
        myprintf("max pending with multiplicities: %u\n", max_pending_w_mult.load());
        myprintf("pending backups: %u\n", pending_backups.load());
        myprintf("max pending netresults: %u\n", max_pending_netresults.load());
        myprintf("min pending netresults: %u\n", min_pending_netresults.load());
        myprintf("pending netresults: %u\n", pending_netresults.load());
        //myprintf("%s", m_debug_string.c_str());
#endif
    }
    int bestmove = get_best_move(passflag);

    // Save the explanation.
    m_think_output =
        str(boost::format("move %d, %c => %s\n%s")
        % m_rootstate.get_movenum()
        % (color == FastBoard::BLACK ? 'B' : 'W')
        % m_rootstate.move_to_text(bestmove).c_str()
        % get_analysis(m_root->get_visits()).c_str());

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return bestmove;
}

// Brief output from last think() call.
std::string UCTSearch::explain_last_think() const {
    return m_think_output;
}

void UCTSearch::ponder() {
    auto disable_reuse = cfg_analyze_tags.has_move_restrictions();
    if (disable_reuse) {
        m_last_rootstate.reset(nullptr);
    }

    update_root();

    Time start;
    auto keeprunning = true;
    auto last_output = 0;
    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (cfg_analyze_tags.interval_centis()) {
            Time elapsed;
            int elapsed_centis = Time::timediff_centis(start, elapsed);
            if (elapsed_centis - last_output > cfg_analyze_tags.interval_centis()) {
                last_output = elapsed_centis;
                output_analysis(m_rootstate, *m_root);
            }
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(0, 1);
    } while (!Utils::input_pending() && keeprunning);

    // stop the search
    m_run = keeprunning;
    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centis() && last_output == 0) {
        output_analysis(m_rootstate, *m_root);
    }

    // Display search info.
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);

    myprintf("\n%7.2f visits, %u nodes, %u inflated\n\n", m_root->get_visits(), 
        UCTNodePointer::m_nodes.load(), UCTNodePointer::m_inflated_nodes.load());
    m_network.dump_stats();
#ifdef ACCUM_DEBUG
    myprintf("failed simulations: %u\n", failed_simulations.load());
    myprintf("max leaf vl multiplicity: %u\n", max_leaf_vl.load());
    myprintf("max vl multiplicity: %u\n", max_vl.load());
    myprintf("max pending backups: %u\n", max_pending_backups.load());
    myprintf("max pending with multiplicities: %u\n", max_pending_w_mult.load());
    myprintf("pending backups: %u\n", pending_backups.load());
    myprintf("max pending netresults: %u\n", max_pending_netresults.load());
    myprintf("min pending netresults: %u\n", min_pending_netresults.load());
    myprintf("pending netresults: %u\n", pending_netresults.load());
#endif
    // Copy the root state. Use to check for tree re-use in future calls.
    if (!disable_reuse) {
        m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    }
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts),
                                      decltype(m_maxplayouts)>::value,
                  "Inconsistent types for playout amount.");
    m_maxplayouts = std::min(playouts, UNLIMITED_PLAYOUTS);
}

void UCTSearch::set_visit_limit(int visits) {
    static_assert(std::is_convertible<decltype(visits),
                                      decltype(m_maxvisits)>::value,
                  "Inconsistent types for visits amount.");
    // Limit to type max / 2 to prevent overflow when multithreading.
    m_maxvisits = std::min(visits, UNLIMITED_PLAYOUTS);
}

