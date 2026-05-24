// Copyright 2026 r1o project. JACCL C shim for Rust FFI.
// Wraps MeshGroup via the void* Group API with timeout + probes.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>

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

// Stored as shared_ptr to prevent accidental double-free
struct GroupHandle {
    std::shared_ptr<jaccl::Group> group;
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

        // Construct MeshGroup directly (bypasses Config + devices JSON).
        // Capture by value (shared_ptr copies) — NOT [&] — so the lambda
        // owns the data even if this function returns on timeout.
        auto fut = std::async(std::launch::async,
            [rank, device_names, coord_str]() -> std::shared_ptr<jaccl::Group> {
                return std::make_shared<jaccl::MeshGroup>(
                    rank, *device_names, *coord_str);
            });

        auto status = fut.wait_for(std::chrono::milliseconds(timeout_ms));
        if (status == std::future_status::timeout) {
            std::cerr << "[jaccl-shim] init_auto timed out after "
                      << timeout_ms << "ms" << std::endl;
            return nullptr;
        }

        auto group = fut.get();
        if (!group) return nullptr;

        auto* handle = new GroupHandle;
        handle->group = group;
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

        // Init with timeout via async.
        // Capture by value (shared_ptr copies) — NOT [&].
        auto fut = std::async(std::launch::async, [cfg]() {
            return jaccl::init(*cfg, /*strict=*/true);
        });

        auto status = fut.wait_for(std::chrono::milliseconds(timeout_ms));
        if (status == std::future_status::timeout) {
            std::cerr << "[jaccl-shim] init timed out after "
                      << timeout_ms << "ms" << std::endl;
            return nullptr;
        }

        auto group = fut.get();
        if (!group) return nullptr;

        auto* handle = new GroupHandle{std::move(group)};
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
        int rank = h->group->rank();
        int size = h->group->size();
        if (size < 2) return -1;

        // Send 1 byte to peer and recv 1 byte back
        uint8_t probe_byte = 0xAB;
        uint8_t recv_byte = 0;
        int peer = (rank == 0) ? 1 : 0;

        // Use async with 2-second timeout
        auto fut = std::async(std::launch::async, [&]() {
            if (rank == 0) {
                h->group->send(&probe_byte, 1, peer);
                h->group->recv(&recv_byte, 1, peer);
            } else {
                h->group->recv(&recv_byte, 1, peer);
                h->group->send(&probe_byte, 1, peer);
            }
        });

        auto status = fut.wait_for(std::chrono::seconds(2));
        if (status == std::future_status::timeout) {
            return -1; // QP stale
        }
        fut.get(); // propagate exceptions
        return 0;  // alive
    } catch (...) {
        return -1; // QP stale or error
    }
}

int jaccl_group_send(
    jaccl_group_t g, const void* buf, size_t len,
    int dst, int timeout_ms
) {
    try {
        auto* h = static_cast<GroupHandle*>(g);
        auto fut = std::async(std::launch::async, [&]() {
            h->group->send(buf, len, dst);
        });
        auto status = fut.wait_for(std::chrono::milliseconds(timeout_ms));
        if (status == std::future_status::timeout) {
            return -1; // timeout
        }
        fut.get(); // propagate exceptions
        return 0;
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
        auto fut = std::async(std::launch::async, [&]() {
            h->group->recv(buf, len, src);
        });
        auto status = fut.wait_for(std::chrono::milliseconds(timeout_ms));
        if (status == std::future_status::timeout) {
            return -1; // timeout
        }
        fut.get(); // propagate exceptions
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[jaccl-shim] recv error: " << e.what() << std::endl;
        return -2;
    } catch (...) {
        return -2;
    }
}

void jaccl_group_free(jaccl_group_t g) {
    if (g) {
        auto* h = static_cast<GroupHandle*>(g);
        delete h;
    }
}

} // extern "C"
