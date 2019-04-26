/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Junhee Yoo and contributors

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
*/

#include <iostream>

#include "GTP.h"
#include "DistributedNetwork.h"

using boost::asio::ip::tcp;

template <typename... T> static void netprintf(const char * fmt, T... params) {
    if (cfg_nn_client_verbose) {
        Utils::myprintf(fmt, params...);
    }
}
std::vector<float> DistributedClientNetwork::get_output_from_socket(const std::vector<float> & input_data,
                                                                    boost::asio::ip::tcp::socket & socket) {

    std::vector<char> input_data_ch(input_data.size()); // input_data (18*361)
    assert(input_data_ch.size() == INPUT_CHANNELS * NUM_INTERSECTIONS);
    std::copy(begin(input_data), end(input_data), begin(input_data_ch));

    std::vector<float> output_data_f(NUM_INTERSECTIONS + 2);
    try {
        boost::system::error_code error;
        boost::asio::write(socket, boost::asio::buffer(input_data_ch), error);
        if (error)
            throw boost::system::system_error(error); // Some other error.

        boost::asio::read(socket, boost::asio::buffer(output_data_f), error);
        if (error)
            throw boost::system::system_error(error); // Some other error.
    } catch (std::exception& e)
    {
        netprintf("Socket killed by exception : %s\n", e.what());
        throw;
    }
    return output_data_f;
}

void DistributedClientNetwork::initialize(int playouts, const std::vector<std::string> & serverlist, std::uint64_t hash) {
    m_servers = std::vector<Server>(serverlist.size());
    for(auto i=0U; i<serverlist.size(); i++) {
        m_servers[i].m_server_name = serverlist[i];
    }

    Network::initialize(playouts, "");

    // if this didn't create enough threads, the background thread will retry creating more and more
    // if it never creates enough threads, local capability (be it CPU or GPU) will be used
    init_servers(hash);

    // create a background thread which tries to create new connectins if some are dead.
    // thread stays active forever, hence if somebody wants to have capability of destroying
    // hets in the middle of a run, this thread should also be safely killed...
    std::thread t(
        [this, hash]() {
            while (m_running.load()) {
                std::this_thread::sleep_for(
                    std::chrono::seconds(1)
                );
                if (m_active_socket_count.load() < static_cast<size_t>(cfg_num_threads)) {
                    init_servers(hash);
                }
            }
        }
    );
    m_fork_thread = std::move(t);
}

void DistributedClientNetwork::init_servers(std::uint64_t hash) {
    const auto num_threads = (cfg_num_threads - m_active_socket_count.load() + m_servers.size() - 1) / m_servers.size();
    for (auto & s : m_servers) {
        auto x = s.m_server_name;
        std::vector<std::string> x2;
        boost::split(x2, x, boost::is_any_of(":"));
        if (x2.size() != 2) {
            printf("Error in --nn-client argument parsing : Expecting [server]:[port] syntax\n");
            printf("(got %s\n", x.c_str());
            throw std::runtime_error("Malformed --nn-client argument ");
        }

        auto addr = x2[0];
        auto port = x2[1];

        tcp::resolver resolver(m_io_service);

        // these are deprecated in latest boost but still a quite recent Ubuntu distribution
        // doesn't support the alternative newer interface.
        decltype(resolver)::iterator endpoints;
        decltype(resolver)::query query(addr, port);
        try {
            endpoints = resolver.resolve(query);
        } catch (...) {
            netprintf("Cannot resolve server address %s port %s\n", addr.c_str(), port.c_str());
            // cannot resolve server - probably server dead
            break;
        }


        for (auto i=size_t{0}; i<num_threads; i++) {
            auto socket = std::make_shared<tcp::socket>(m_io_service);

            try {
                auto connect_task = [this, &addr, &port, &socket, &endpoints, hash] () {
                    boost::asio::connect(*socket, endpoints);
                    std::array<std::uint64_t,1> my_hash {hash};
                    std::array<std::uint64_t,1> remote_hash {0};
    
                    boost::system::error_code error;
                    boost::asio::write(*socket, boost::asio::buffer(my_hash), error);
                    if (error)
                        throw boost::system::system_error(error); // Some other error.
    
                    boost::asio::read(*socket, boost::asio::buffer(remote_hash), error);
                    if (error)
                        throw boost::system::system_error(error); // Some other error.
    
                    if(my_hash[0] != remote_hash[0]) {
                        netprintf(
                            "NN client dropped to server %s port %s (hash mismatch, remote=%llx, local=%llx)\n",
                                addr.c_str(), port.c_str(), remote_hash[0], my_hash[0]);
                        throw std::exception();
                    }
                };
        
                auto f = std::async(std::launch::async, connect_task);
                auto res = f.wait_for(std::chrono::milliseconds(500));
                if (res == std::future_status::timeout) {
                    socket->shutdown(tcp::socket::shutdown_send);
                    socket->shutdown(tcp::socket::shutdown_receive);
                    socket->close();
                    f.get();
                    throw std::exception();
                }
                f.get();
            } catch (...) {
                // doesn't work. Probably remote side ran out of threads.
                // drop socket.
                netprintf("NN client dropped to server %s port %s (thread %d)\n", addr.c_str(), port.c_str(), i);
                continue;
            }

            m_active_socket_count++;
            s.m_active_socket_count++;
            auto w = std::bind([this, &s](std::shared_ptr<tcp::socket> socket) { worker_thread(std::move(*socket), s); }, socket );
            std::thread t(w);
            t.detach();

            netprintf("NN client connected to server %s port %s (thread %d)\n", addr.c_str(), port.c_str(), i);
        }
    }

    m_socket_initialized = true;
}

void DistributedClientNetwork::initialize(int playouts, const std::string & weightsfile) {
    m_local_initialized = true;
    Network::initialize(playouts, weightsfile);
}

DistributedClientNetwork::~DistributedClientNetwork() {
    m_running = false;
    m_fork_thread.join();
    
    for (auto & s : m_servers) {
        s.m_cv.notify_all();
    }
    while (m_active_socket_count.load() > 0) {}
}

std::pair<std::vector<float>,float> DistributedClientNetwork::get_output_internal(
                                      const std::vector<float> & input_data,
                                      bool selfcheck) {


    if (selfcheck) {
        assert(m_local_initialized);
        return Network::get_output_internal(input_data, true);
    }

    if (!m_socket_initialized) {
        assert(m_local_initialized);
        return Network::get_output_internal(input_data, selfcheck);
    }

    const auto batch_size = cfg_batch_size > 0 ? cfg_batch_size : 1U;
    const auto ptr = m_server_ptr++;
    const auto server_num = (ptr/batch_size) % m_servers.size();
    auto & server = m_servers[server_num];

    if (server.m_active_socket_count.load() == 0) {
        if (m_active_socket_count.load() == 0) {
            std::this_thread::sleep_for(
                std::chrono::seconds(1)
            );
        }
        return get_output_internal(input_data, selfcheck);
    } else {
        // if we are oversubscribing the server...
        if (server.m_active_eval_count.load() >= server.m_active_socket_count.load()) {
            // if load ratio is above average, skip
            if (m_active_eval_count.load() * server.m_active_socket_count.load()
                 < m_active_socket_count.load() * server.m_active_eval_count.load()) 
            {
                return get_output_internal(input_data, selfcheck);
            }
        }
    }
    std::pair<std::vector<float>,float> ret;

    auto entry = std::make_shared<ForwardQueueEntry>(input_data, ret);
    std::unique_lock<std::mutex> lk(entry->mutex);

    {
        std::unique_lock<std::mutex> lk2(server.m_forward_mutex);
        server.m_forward_queue.push_back(entry);
    }

    m_active_eval_count++;
    server.m_active_eval_count++;
    server.m_cv.notify_one();
    entry->cv.wait_for(lk, std::chrono::milliseconds(500));

    if (!entry->out_ready) {
        if (entry->socket != nullptr) {
            // force-close socket
            try {
                entry->socket->shutdown(tcp::socket::shutdown_send);
                entry->socket->shutdown(tcp::socket::shutdown_receive);
                entry->socket->close();
            } catch (...) {}
        } else {
            // no socket picked this entry yet - clean up, don't compute
            entry->out_ready = true; // this signals the socket that this should be dropped
        }

        lk.unlock();

        m_active_eval_count--;
        server.m_active_eval_count--;
        return get_output_internal(input_data, selfcheck);
    }

    m_active_eval_count--;
    server.m_active_eval_count--;

    return ret;
}


void DistributedClientNetwork::worker_thread(boost::asio::ip::tcp::socket && socket, Server & server) {
    while (true) {
        std::unique_lock<std::mutex> lk(server.m_forward_mutex);
        if (server.m_forward_queue.empty()) {
            server.m_cv.wait(lk, [this, &server]() { return !server.m_forward_queue.empty(); });
        }
     
        auto entry = server.m_forward_queue.front();
        server.m_forward_queue.pop_front();
        lk.unlock();

    	std::unique_lock<std::mutex> lk2(entry->mutex);
        if (entry->out_ready) {
            continue;
        }
        entry->socket = &socket;
        lk2.unlock();

        try {
            auto output_data_f = get_output_from_socket(entry->in, socket);

            lk2.lock();
            entry->out.first = std::vector<float>(NUM_INTERSECTIONS + 1);
            std::copy(begin(output_data_f), begin(output_data_f) + NUM_INTERSECTIONS, begin(entry->out.first));
            entry->out.second = output_data_f[NUM_INTERSECTIONS + 1];
            entry->out_ready = true;
            lk2.unlock();
            entry->cv.notify_one();
        } catch (...) {
            lk2.lock();
            entry->socket = nullptr;
            lk2.unlock();
            m_active_socket_count--;
            server.m_active_socket_count--;
            return;
        }
    }
}


NetServer::NetServer(Network & net) : m_net(net)
{
}

void NetServer::listen(int portnum, std::uint64_t hash) {
    try {
        std::atomic<int> num_threads{0};

        tcp::acceptor acceptor(m_io_service, tcp::endpoint(tcp::v4(), portnum));
        Utils::myprintf("NN server listening on port %d\n", portnum);

        for (;;)
        {
            tcp::socket socket(m_io_service);
            acceptor.accept(socket);

            int v = num_threads++;
            if (v >= cfg_num_threads) {
                --num_threads;
                Utils::myprintf("Dropping connection from %s due to too many threads\n",
                     socket.remote_endpoint().address().to_string().c_str()
                );
                socket.shutdown(tcp::socket::shutdown_send);
                socket.shutdown(tcp::socket::shutdown_receive);
                socket.close();
                continue;
            }

            Utils::myprintf("NN server connection established from %s (thread %d, max %d)\n",
                     socket.remote_endpoint().address().to_string().c_str(), v, cfg_num_threads
            );

            std::thread t(
                std::bind(
                    [&num_threads, this, hash](tcp::socket & socket) {

                        auto remote_endpoint = socket.remote_endpoint().address().to_string();

                        std::array<std::uint64_t, 1> my_hash{hash};
                        std::array<std::uint64_t, 1> remote_hash {0};
                        boost::system::error_code error;

                        boost::asio::read(socket, boost::asio::buffer(remote_hash), error);
                        if (error)
                            throw boost::system::system_error(error); // Some other error.

                        boost::asio::write(socket, boost::asio::buffer(my_hash), error);
                        if (error)
                            throw boost::system::system_error(error); // Some other error.


                        while (true) {
                            std::array<char,  INPUT_CHANNELS * NUM_INTERSECTIONS> buf;

                            boost::system::error_code error;
                            boost::asio::read(socket, boost::asio::buffer(buf), error);
                            if (error == boost::asio::error::eof)
                                break; // Connection closed cleanly by peer.
                            else if (error) {
                                Utils::myprintf("Socket read failed with message : %s\n",
                                                error.message().c_str()
                                );
                                break;
                            }

                            std::vector<float> input_data(INPUT_CHANNELS * NUM_INTERSECTIONS);
                            std::copy(begin(buf), end(buf), begin(input_data));

                            auto result = m_net.get_output_internal(input_data, false);

                            std::array<float, NUM_INTERSECTIONS+2> obuf;
                            std::copy(begin(result.first), end(result.first), begin(obuf));
                            obuf[NUM_INTERSECTIONS+1] = result.second;
                            boost::asio::write(socket, boost::asio::buffer(obuf), error);
                            if (error == boost::asio::error::eof)
                                break; // Connection closed cleanly by peer.
                            else if (error) {
                                Utils::myprintf("Socket write failed with message : %s\n",
                                                error.message().c_str()
                                );
                                break;
                            }
                        }

                        Utils::myprintf("NN server connection closed from %s\n", remote_endpoint.c_str());
                        num_threads--;
                    },
                    std::move(socket)
                )
            );
            t.detach();
        }
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}
