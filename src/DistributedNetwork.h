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

#ifndef DISTRIBUTED_NETWORK_H_INCLUDED
#define DISTRIBUTED_NETWORK_H_INCLUDED

#include "config.h"

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <vector>
#include <thread>

#include "Utils.h"
#include "SMP.h"

#include "Network.h"

class DistributedClientNetwork : public Network
{
private:
    boost::asio::io_service m_io_service;
    std::atomic<size_t> m_active_socket_count{0};
    std::atomic<size_t> m_active_eval_count{0};
    std::thread m_fork_thread;
    std::vector<std::string> m_serverlist;
    bool m_local_initialized = false;
    bool m_socket_initialized = false;

    std::atomic<bool> m_running{true};

    class ForwardQueueEntry {
    public:
        std::mutex mutex;
        std::condition_variable cv;
        const std::vector<float>& in;
        std::pair<std::vector<float>,float> & out;
        bool out_ready = false;
        boost::asio::ip::tcp::socket * socket = nullptr;
        ForwardQueueEntry(const std::vector<float>& input,
                        std::pair<std::vector<float>,float>& output) 
            : in(input), out(output)
        {}
    };

    class Server {
    public:
        std::string m_server_name;
        std::mutex m_forward_mutex;
        std::condition_variable m_cv;
        std::list<std::shared_ptr<ForwardQueueEntry>> m_forward_queue;
        std::atomic<int> m_active_socket_count{0};
        std::atomic<int> m_active_eval_count{0};
    };
    std::vector<Server> m_servers;
    std::atomic<int> m_server_ptr;

    std::vector<float> get_output_from_socket(const std::vector<float> & input_data,
                                              boost::asio::ip::tcp::socket & socket);

    void worker_thread(boost::asio::ip::tcp::socket && socket, Server & server);
public:
    void initialize(int playouts, const std::vector<std::string> & serverlist, std::uint64_t hash);
    void initialize(int playouts, const std::string & weightsfile);
    void init_servers(std::uint64_t hash);

    virtual ~DistributedClientNetwork();

protected:
    virtual std::pair<std::vector<float>,float> get_output_internal(const std::vector<float> & input_data,
                                                                    bool selfcheck = false);
};


class NetServer
{
    static constexpr auto INPUT_CHANNELS = Network::INPUT_CHANNELS;
private:
    boost::asio::io_service m_io_service;
    Network & m_net;
public:
    NetServer(Network & net);
    void listen(int portnum, std::uint64_t hash);
};

#endif
