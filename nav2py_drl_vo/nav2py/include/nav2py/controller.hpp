/*
 *  License: MIT
 *  Author(s): Shrijit Singh <shrijitsingh99@gmail.com>
 */

#ifndef NAV2PY__CONTROLLER_HPP_
#define NAV2PY__CONTROLLER_HPP_

#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

double recv_double(int sockfd)
{
    uint64_t retval; //double

    size_t total_bytes = 0;
    do
    {
        size_t bytes_received = recv(sockfd, &retval, (sizeof retval) - total_bytes, 0);
        if (bytes_received <= 0)
            throw std::runtime_error("socket closed");
        total_bytes += bytes_received;
    } while (total_bytes < (sizeof retval));

    retval = be64toh(retval);
    double *d = (double *)&retval;
    return *d;
}

namespace nav2py
{
    class Controller : public nav2_core::Controller
    {

    private:
        int socket_ = -1;

    protected:
        bool nav2py_bootstrap(const std::string cmd)
        {
            // determine port from client
            uint16_t port;
            uint16_t magic = 48073; // magic number

            auto logger = rclcpp::get_logger("TemplateController");

            RCLCPP_DEBUG(logger, "starting process %s", cmd.c_str());

            FILE *pipe = popen(cmd.c_str(), "r");

            if (!pipe)
                throw std::runtime_error("popen() failed");

            constexpr size_t buffer_size = (sizeof port) + (sizeof magic); // port followed by magic nubmer
            char buffer[buffer_size + 1];
            size_t n = 0;
            try
            {
                while ((n < buffer_size) && (fgets(buffer + n, (sizeof buffer) - n, pipe) != NULL))
                    n += strlen(buffer + n);

                if (n < buffer_size)
                    throw std::runtime_error("bootstrap failed");

                uint16_t n_magic;
                std::memcpy(&n_magic, buffer + (sizeof port), sizeof n_magic);

                if (ntohs(n_magic) != magic)
                    throw std::runtime_error("magic number mismatch");

                uint16_t n_port;
                std::memcpy(&n_port, buffer, sizeof n_port);
                port = ntohs(n_port);
            }
            catch (...)
            {
                pclose(pipe);
                throw;
            }

            // open socket client
            int sockfd = socket(AF_INET, SOCK_STREAM, 0);
            if (sockfd == -1)
                throw std::runtime_error("socket() failed");

            sockaddr_in serverAddress;
            serverAddress.sin_family = AF_INET;
            serverAddress.sin_port = htons(port);
            serverAddress.sin_addr.s_addr = INADDR_ANY;

            if (connect(sockfd, (struct sockaddr *)&serverAddress, sizeof(serverAddress)))
                throw std::runtime_error("connect() failed");

            socket_ = sockfd;
            RCLCPP_DEBUG(logger, "socket connected");

            return 0;
        }

        void nav2py_cleanup()
        {
            close(socket_);
            socket_ = -1;
        }
        
        const std::string SEP_BYTE = "\x01";
        const std::string END_BYTE = "\x03";

        void nav2py_send(std::string name, std::vector<std::string> messages){
            send(socket_, name.c_str(), name.size(), 0);
            for(const auto& message : messages){
                send(socket_, SEP_BYTE.c_str(), SEP_BYTE.size(), 0);
                send(socket_, message.c_str(), message.size(), 0);
            }
            send(socket_, END_BYTE.c_str(), END_BYTE.size(), 0);
        }

        geometry_msgs::msg::Twist wait_for_cmd_vel()
        {
            geometry_msgs::msg::Twist cmd_vel;
            cmd_vel.linear.x = recv_double(socket_);
            cmd_vel.angular.z = recv_double(socket_);
            return cmd_vel;
        };
    };
}

#endif // NAV2PY__CONTROLLER_HPP_