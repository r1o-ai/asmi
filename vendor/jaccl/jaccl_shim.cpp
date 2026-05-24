// Copyright 2026 r1o project. JACCL C shim for Rust FFI.
// Wraps MeshGroup via the void* Group API with timeout + probes.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include <json.hpp>

#include "jaccl/jaccl.h"
#include "jaccl/jaccl_shim.h"
#include "jaccl/rdma.h"
#include "jaccl/mesh.h"

using json = nlohmann::json;

namespace {

// Duplicate of parse_devices_json from jaccl.cpp (lives in anon namespace there)
std::vector<std::vector<std::vector<std::string>>>
parse_devices(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error(
            std::string("[jaccl-shim] Cannot open devices file: ") + path);
    }
    json devices = json::parse(f);
    if (!devices.is_array()) {
        throw std::runtime_error("[jaccl-shim] Device file must be a JSON array");
    }
    std::vector<std::vector<std::vector<std::string>>> result(devices.size());
    for (size_t rank = 0; rank < devices.size(); rank++) {
        auto conn = devices[rank];
        result[rank].resize(conn.size());
        for (size_t dst = 0; dst < conn.size(); dst++) {
            auto names = conn[dst];
            if (names.is_string()) {
                result[rank][dst].push_back(names);
            } else if (names.is_array()) {
                for (auto& n : names) {
                    result[rank][dst].push_back(n);
                }
            }
            // null entries are fine — they map to empty vectors
        }
    }
    return result;
}

// Stored as shared_ptr to prevent accidental double-free.
// `poisoned` flag allows cancel_pending to signal detached spin-loop threads
// that the group is dead and they should stop polling. This is the fallback
// approach since MeshGroup::connections_ is private — we can't destroy CQs
// directly from outside.
struct GroupHandle {
    std::shared_ptr<jaccl::Group> group;
    std::shared_ptr<std::atomic<bool>> poisoned =
        std::make_shared<std::atomic<bool>>(false);
};

} // namespace

extern "C" {

bool jaccl_is_available(void) {
    try {
        return jaccl::is_available();
    } catch (...) {
        return false;
    }
}

int jaccl_pd_budget_probe(const char* device_name) {
    try {
        auto& ibv = jaccl::ibv();
        if (!ibv.is_available()) return -1;

        int num_devices = 0;
        ibv_device** devices = ibv.get_device_list(&num_devices);
        if (!devices) return -1;

        ibv_context* ctx = nullptr;
        for (int i = 0; i < num_devices; i++) {
            if (std::strcmp(ibv.get_device_name(devices[i]), device_name) == 0) {
                ctx = ibv.open_device(devices[i]);
                break;
            }
        }
        ibv.free_device_list(devices);
        if (!ctx) return -1;

        // Try to allocate a PD — if it succeeds, we still have budget
        ibv_pd* pd = ibv.alloc_pd(ctx);
        if (!pd) {
            ibv.close_device(ctx);
            return 0; // PD exhausted
        }
        ibv.dealloc_pd(pd);
        ibv.close_device(ctx);
        return 1; // At least 1 PD available (we don't know exact count)
    } catch (...) {
        return -1;
    }
}

int jaccl_pd_probe_any_active(void) {
    try {
        auto& ibv = jaccl::ibv();
        if (!ibv.is_available()) return -1;

        int num_devices = 0;
        ibv_device** devices = ibv.get_device_list(&num_devices);
        if (!devices || num_devices == 0) return -1;

        int result = -1;
        for (int i = 0; i < num_devices; i++) {
            ibv_context* ctx = ibv.open_device(devices[i]);
            if (!ctx) continue;

            ibv_pd* pd = ibv.alloc_pd(ctx);
            if (pd) {
                ibv.dealloc_pd(pd);
                ibv.close_device(ctx);
                ibv.free_device_list(devices);
                return 1; // at least one device has PD budget
            }
            ibv.close_device(ctx);
            result = 0; // opened a device but PD exhausted — keep trying others
        }
        ibv.free_device_list(devices);
        return result;
    } catch (...) {
        return -1;
    }
}

jaccl_group_t jaccl_init_mesh_auto(
    int rank,
    int world_size,
    const char* coordinator_ip,
    int coordinator_port,
    int timeout_ms
) {
    try {
        auto& ibv = jaccl::ibv();
        if (!ibv.is_available()) return nullptr;

        // Discover first available RDMA device
        int num_devices = 0;
        ibv_device** devices = ibv.get_device_list(&num_devices);
        if (!devices || num_devices == 0) return nullptr;

        std::string active_device;
        for (int i = 0; i < num_devices; i++) {
            ibv_context* ctx = ibv.open_device(devices[i]);
            if (!ctx) continue;
            // Check if PD can be allocated (device is usable)
            ibv_pd* pd = ibv.alloc_pd(ctx);
            if (pd) {
                active_device = ibv.get_device_name(devices[i]);
                ibv.dealloc_pd(pd);
                ibv.close_device(ctx);
                break;
            }
            ibv.close_device(ctx);
        }
        ibv.free_device_list(devices);

        if (active_device.empty()) return nullptr;

        // Build device_names on the heap — shared ownership so the async
        // lambda outlives this stack frame on timeout (Bug 1 fix, Phase 1).
        auto device_names = std::make_shared<std::vector<std::string>>(world_size);
        for (int i = 0; i < world_size; i++) {
            (*device_names)[i] = (i == rank) ? "" : active_device;
        }

        // Build coordinator string on the heap (same reason)
        auto coord_str = std::make_shared<std::string>(
            std::string(coordinator_ip) + ":" + std::to_string(coordinator_port));

        // Phase 2: std::thread::detach replaces std::async.
        // std::async's future dtor blocks on timeout → function hangs.
        // With detach, timeout returns nullptr immediately. The detached
        // thread owns all data via shared_ptr — when MeshGroup ctor
        // finishes (success or failure), shared_ptrs drop → RAII cleans
        // up PDs. No PD leak.
        auto group_out = std::make_shared<std::shared_ptr<jaccl::Group>>(nullptr);
        auto done = std::make_shared<std::atomic<bool>>(false);
        auto mtx = std::make_shared<std::mutex>();
        auto cv = std::make_shared<std::condition_variable>();

        std::thread([rank, device_names, coord_str, group_out, done, mtx, cv]() {
            try {
                auto g = std::make_shared<jaccl::MeshGroup>(
                    rank, *device_names, *coord_str);
                {
                    std::lock_guard<std::mutex> lock(*mtx);
                    *group_out = g;
                    done->store(true);
                }
                cv->notify_one();
            } catch (const std::exception& e) {
                std::cerr << "[jaccl-shim] init_auto thread failed: "
                          << e.what() << std::endl;
                done->store(true);
                cv->notify_one();
            } catch (...) {
                done->store(true);
                cv->notify_one();
            }
        }).detach();

        {
            std::unique_lock<std::mutex> lock(*mtx);
            if (!cv->wait_for(lock, std::chrono::milliseconds(timeout_ms),
                              [&] { return done->load(); })) {
                // Timeout — thread still running but will clean up via RAII
                // when shared_ptrs drop. PDs recovered.
                std::cerr << "[jaccl-shim] init_auto timed out after "
                          << timeout_ms << "ms" << std::endl;
                return nullptr;
            }
        }

        if (!*group_out) return nullptr;

        auto* handle = new GroupHandle;
        handle->group = *group_out;
        return static_cast<jaccl_group_t>(handle);
    } catch (const std::exception& e) {
        std::cerr << "[jaccl-shim] init_auto failed: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

jaccl_group_t jaccl_init_mesh(
    int rank,
    int world_size,
    const char* coordinator_ip,
    int coordinator_port,
    const char* devices_json_path,
    int timeout_ms
) {
    try {
        // Build coordinator string on the heap — shared ownership so the
        // async lambda outlives this stack frame on timeout (Bug 1 fix, Phase 1).
        auto coord_str = std::make_shared<std::string>(
            std::string(coordinator_ip) + ":" + std::to_string(coordinator_port));

        // Parse devices JSON
        auto devices = parse_devices(devices_json_path);

        // Build Config on the heap (same reason — [&] capture is UB on timeout)
        auto cfg = std::make_shared<jaccl::Config>();
        cfg->set_rank(rank)
           .set_coordinator(*coord_str)
           .set_devices(std::move(devices));

        // Phase 2: std::thread::detach replaces std::async (same pattern
        // as init_auto). Detached thread owns cfg via shared_ptr — RAII
        // cleans up PDs on timeout.
        auto group_out = std::make_shared<std::shared_ptr<jaccl::Group>>(nullptr);
        auto done = std::make_shared<std::atomic<bool>>(false);
        auto mtx = std::make_shared<std::mutex>();
        auto cv = std::make_shared<std::condition_variable>();

        std::thread([cfg, group_out, done, mtx, cv]() {
            try {
                auto g = jaccl::init(*cfg, /*strict=*/true);
                {
                    std::lock_guard<std::mutex> lock(*mtx);
                    *group_out = std::move(g);
                    done->store(true);
                }
                cv->notify_one();
            } catch (const std::exception& e) {
                std::cerr << "[jaccl-shim] init thread failed: "
                          << e.what() << std::endl;
                done->store(true);
                cv->notify_one();
            } catch (...) {
                done->store(true);
                cv->notify_one();
            }
        }).detach();

        {
            std::unique_lock<std::mutex> lock(*mtx);
            if (!cv->wait_for(lock, std::chrono::milliseconds(timeout_ms),
                              [&] { return done->load(); })) {
                std::cerr << "[jaccl-shim] init timed out after "
                          << timeout_ms << "ms" << std::endl;
                return nullptr;
            }
        }

        if (!*group_out) return nullptr;

        auto* handle = new GroupHandle{*group_out};
        return static_cast<jaccl_group_t>(handle);
    } catch (const std::exception& e) {
        std::cerr << "[jaccl-shim] init failed: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[jaccl-shim] init failed: unknown error" << std::endl;
        return nullptr;
    }
}

int jaccl_group_rank(jaccl_group_t g) {
    try {
        auto* h = static_cast<GroupHandle*>(g);
        return h->group->rank();
    } catch (...) {
        return -1;
    }
}

int jaccl_group_size(jaccl_group_t g) {
    try {
        auto* h = static_cast<GroupHandle*>(g);
        return h->group->size();
    } catch (...) {
        return -1;
    }
}

int jaccl_group_probe(jaccl_group_t g) {
    try {
        auto* h = static_cast<GroupHandle*>(g);
        if (h->poisoned->load()) return -1;

        int rank = h->group->rank();
        int size = h->group->size();
        if (size < 2) return -1;

        int peer = (rank == 0) ? 1 : 0;

        // Phase 1b: detach+condvar pattern replaces std::async.
        // std::async's future dtor blocks, making the 2s timeout dead code.
        auto group = h->group;  // shared_ptr copy — safe to outlive caller
        auto done = std::make_shared<std::atomic<int>>(0);  // 0=pending, 1=ok, -2=error
        auto mtx = std::make_shared<std::mutex>();
        auto cv = std::make_shared<std::condition_variable>();
        auto poisoned = h->poisoned;  // shared_ptr copy

        std::thread([group, peer, rank, done, mtx, cv, poisoned]() {
            try {
                // Probe data — heap-allocated to avoid stack UB
                uint8_t probe_byte = 0xAB;
                uint8_t recv_byte = 0;
                if (rank == 0) {
                    group->send(&probe_byte, 1, peer);
                    if (poisoned->load()) { done->store(-2); cv->notify_one(); return; }
                    group->recv(&recv_byte, 1, peer);
                } else {
                    group->recv(&recv_byte, 1, peer);
                    if (poisoned->load()) { done->store(-2); cv->notify_one(); return; }
                    group->send(&probe_byte, 1, peer);
                }
                done->store(1);
            } catch (...) {
                done->store(-2);
            }
            cv->notify_one();
        }).detach();

        std::unique_lock<std::mutex> lock(*mtx);
        if (!cv->wait_for(lock, std::chrono::seconds(2),
                          [&] { return done->load() != 0; })) {
            return -1; // genuine timeout — detached thread will finish eventually
        }
        return done->load() == 1 ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

int jaccl_group_send(
    jaccl_group_t g, const void* buf, size_t len,
    int dst, int timeout_ms
) {
    try {
        auto* h = static_cast<GroupHandle*>(g);
        if (h->poisoned->load()) return -2;

        // Phase 1b: detach+condvar replaces std::async (future dtor blocks =
        // dead timeout). Copy input buffer to heap — caller may free after
        // timeout return.
        auto data = std::make_shared<std::vector<uint8_t>>(
            static_cast<const uint8_t*>(buf),
            static_cast<const uint8_t*>(buf) + len);
        auto group = h->group;  // shared_ptr copy — safe to outlive caller
        auto done = std::make_shared<std::atomic<int>>(0);  // 0=pending, 1=ok, -2=error
        auto mtx = std::make_shared<std::mutex>();
        auto cv = std::make_shared<std::condition_variable>();

        std::thread([group, data, dst, done, mtx, cv]() {
            try {
                group->send(data->data(), data->size(), dst);
                done->store(1);
            } catch (...) {
                done->store(-2);
            }
            cv->notify_one();
        }).detach();

        std::unique_lock<std::mutex> lock(*mtx);
        if (!cv->wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [&] { return done->load() != 0; })) {
            return -1; // genuine timeout — detached thread will finish eventually
        }
        return done->load() == 1 ? 0 : -2;
    } catch (const std::exception& e) {
        std::cerr << "[jaccl-shim] send error: " << e.what() << std::endl;
        return -2;
    } catch (...) {
        return -2;
    }
}

int jaccl_group_recv(
    jaccl_group_t g, void* buf, size_t len,
    int src, int timeout_ms
) {
    try {
        auto* h = static_cast<GroupHandle*>(g);
        if (h->poisoned->load()) return -2;

        // Phase 1b: detach+condvar replaces std::async (future dtor blocks =
        // dead timeout). Receive into a heap buffer, memcpy back on success.
        auto recv_buf = std::make_shared<std::vector<uint8_t>>(len);
        auto group = h->group;  // shared_ptr copy — safe to outlive caller
        auto done = std::make_shared<std::atomic<int>>(0);  // 0=pending, 1=ok, -2=error
        auto mtx = std::make_shared<std::mutex>();
        auto cv = std::make_shared<std::condition_variable>();

        std::thread([group, recv_buf, src, done, mtx, cv]() {
            try {
                group->recv(recv_buf->data(), recv_buf->size(), src);
                done->store(1);
            } catch (...) {
                done->store(-2);
            }
            cv->notify_one();
        }).detach();

        std::unique_lock<std::mutex> lock(*mtx);
        if (!cv->wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [&] { return done->load() != 0; })) {
            return -1; // genuine timeout — detached thread will finish eventually
        }
        if (done->load() == 1) {
            std::memcpy(buf, recv_buf->data(), len);
            return 0;
        }
        return -2;
    } catch (const std::exception& e) {
        std::cerr << "[jaccl-shim] recv error: " << e.what() << std::endl;
        return -2;
    } catch (...) {
        return -2;
    }
}

int jaccl_group_cancel_pending(jaccl_group_t g) {
    // Phase 1b: Poison the group to break spin-loops in detached threads.
    //
    // MeshGroup::connections_ is private, so we cannot destroy CQs directly.
    // Instead, we set a poison flag checked by probe's inner thread (between
    // send/recv ops). For send/recv, the RDMA poll loop in mesh_impl.h is
    // internal to MeshGroup and doesn't check flags — those detached threads
    // may spin indefinitely after cable-pull. The poison flag at least
    // prevents NEW operations from being dispatched to a dead group.
    //
    // Full fix (destroy CQs to break poll loops) requires making
    // MeshGroup::connections_ accessible or adding a cancel() method to
    // the upstream JACCL Group interface. Documented as future work.
    if (!g) return -1;
    auto* h = static_cast<GroupHandle*>(g);
    h->poisoned->store(true);
    return 0;
}

void jaccl_group_free(jaccl_group_t g) {
    if (g) {
        auto* h = static_cast<GroupHandle*>(g);
        h->poisoned->store(true);  // poison before free — best effort
        delete h;
    }
}

} // extern "C"
