#ifndef JACCL_SHIM_H
#define JACCL_SHIM_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* jaccl_group_t;

/* -- Availability + PD health -- */
bool jaccl_is_available(void);
int  jaccl_pd_budget_probe(const char* device_name);
     /* Returns remaining PD slots, or -1 on error.
        Allocates+immediately-deallocates one PD to test. */

/* -- Group lifecycle -- */
jaccl_group_t jaccl_init_mesh(
    int rank,                  /* 0 = coordinator/source, 1 = target */
    int world_size,            /* always 2 for point-to-point */
    const char* coordinator_ip,
    int coordinator_port,
    const char* devices_json_path,
    int timeout_ms             /* wallclock timeout for QP handshake */
);
/* Returns NULL on failure. Caller must check. */

int  jaccl_group_rank(jaccl_group_t g);
int  jaccl_group_size(jaccl_group_t g);

/* -- QP liveness probe -- */
int  jaccl_group_probe(jaccl_group_t g);
     /* Sends+receives 1 byte to/from peer. Returns 0 if alive,
        -1 if QP is stale (cable reseated). Caller should re-init. */

/* -- Point-to-point transfer -- */
int  jaccl_group_send(jaccl_group_t g, const void* buf, size_t len,
                      int dst, int timeout_ms);
int  jaccl_group_recv(jaccl_group_t g, void* buf, size_t len,
                      int src, int timeout_ms);
     /* Returns 0 on success, -1 on timeout, -2 on RDMA error. */

/* -- Teardown (call ONLY at process exit or confirmed cable reseat) -- */
void jaccl_group_free(jaccl_group_t g);

#ifdef __cplusplus
}
#endif

#endif /* JACCL_SHIM_H */
