#include <stdio.h>
#include <infiniband/verbs.h>

int main() {
    int n = 0;
    struct ibv_device **devs = ibv_get_device_list(&n);
    if (!devs) { printf("{\"error\":\"no devices\"}\n"); return 1; }

    printf("{\"devices\":[");
    for (int i = 0; i < n; i++) {
        struct ibv_context *ctx = ibv_open_device(devs[i]);
        if (!ctx) continue;

        struct ibv_device_attr attr;
        ibv_query_device(ctx, &attr);

        struct ibv_port_attr pattr;
        const char *state = "UNKNOWN";
        if (ibv_query_port(ctx, 1, &pattr) == 0) {
            switch(pattr.state) {
                case IBV_PORT_ACTIVE: state = "ACTIVE"; break;
                case IBV_PORT_DOWN: state = "DOWN"; break;
                case IBV_PORT_INIT: state = "INIT"; break;
                default: state = "OTHER"; break;
            }
        }

        // Only probe PD on ACTIVE ports — DOWN ports aren't exhausted,
        // they just have no cable so alloc may fail for unrelated reasons.
        const char *pd_status = "inactive";
        if (pattr.state == IBV_PORT_ACTIVE) {
            struct ibv_pd *pd = ibv_alloc_pd(ctx);
            if (pd) {
                pd_status = "ok";
                ibv_dealloc_pd(pd);
            } else {
                pd_status = "exhausted";
            }
        }

        if (i > 0) printf(",");
        printf("{\"name\":\"%s\",\"port_state\":\"%s\",\"max_pd\":%d,\"max_qp\":%d,\"max_mr\":%d,\"pd_status\":\"%s\"}",
               ibv_get_device_name(devs[i]), state, attr.max_pd, attr.max_qp, attr.max_mr,
               pd_status);

        ibv_close_device(ctx);
    }
    printf("]}\n");
    ibv_free_device_list(devs);
    return 0;
}
