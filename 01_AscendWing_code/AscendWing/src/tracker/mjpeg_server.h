#pragma once
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdio>
#include <fcntl.h>

class MjpegServer {
public:
    MjpegServer(int port = 8080) : port_(port), running_(false) {}

    void start() { running_ = true; thread_ = std::thread(&MjpegServer::serve, this); }
    void update(const std::vector<uint8_t>& jpg) { std::lock_guard<std::mutex> lock(mtx_); buffer_ = jpg; }
    void stop() {
        running_ = false;
        shutdown(server_fd_, SHUT_RDWR);  // unblock accept()
        if (thread_.joinable()) thread_.join();
        close(server_fd_);
    }

private:
    int port_, server_fd_ = -1;
    bool running_;
    std::thread thread_;
    std::mutex mtx_;
    std::vector<uint8_t> buffer_;

    static const char* html() {
        return "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
               "<html><body style='margin:0;background:#000;text-align:center'>"
               "<img src='/stream' style='width:100%;max-width:960px'></body></html>";
    }

    void serve() {
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) { perror("socket"); return; }
        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in a = {}; a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons(port_);
        bind(server_fd_, (struct sockaddr*)&a, sizeof(a));
        listen(server_fd_, 2);
        fprintf(stderr, "[MJPG] http://0.0.0.0:%d/\n", port_);

        struct timeval tv; tv.tv_sec = 1; tv.tv_usec = 0;

        while (running_) {
            fd_set fds; FD_ZERO(&fds); FD_SET(server_fd_, &fds);
            if (select(server_fd_ + 1, &fds, nullptr, nullptr, &tv) <= 0) continue;

            int c = accept(server_fd_, nullptr, nullptr);
            if (c < 0) continue;

            char r[1024]; recv(c, r, sizeof(r) - 1, 0); r[sizeof(r) - 1] = 0;

            if (strncmp(r, "GET /stream", 11) == 0) {
                const char* h = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
                send(c, h, strlen(h), MSG_NOSIGNAL);
                while (running_) {
                    std::vector<uint8_t> jpg;
                    { std::lock_guard<std::mutex> lock(mtx_); jpg = buffer_; }
                    if (!jpg.empty()) {
                        char hdr[256]; int n = snprintf(hdr, sizeof(hdr),
                            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", jpg.size());
                        if (send(c, hdr, n, MSG_NOSIGNAL) < 0) break;
                        if (send(c, jpg.data(), jpg.size(), MSG_NOSIGNAL) < 0) break;
                        if (send(c, "\r\n", 2, MSG_NOSIGNAL) < 0) break;
                    }
                    usleep(30000);
                }
            } else {
                const char* p = html();
                send(c, p, strlen(p), MSG_NOSIGNAL);
            }
            close(c);
        }
    }
};
