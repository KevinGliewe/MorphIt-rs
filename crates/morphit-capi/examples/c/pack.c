/*
 * Minimal MorphIt C example: pack one mesh and save the spheres.
 *
 *   pack <mesh.obj|stl> [preset] [num_spheres] [iterations] [seed] [out.json]
 */
#include <stdio.h>
#include <stdlib.h>

#include "morphit.h"

#define CHECK(call)                                                            \
    do {                                                                       \
        morphit_status st_ = (call);                                           \
        if (st_ < 0) {                                                         \
            fprintf(stderr, "%s failed (%d): %s\n", #call, (int)st_,           \
                    morphit_last_error());                                     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int on_progress(const morphit_step_info *info, void *user_data) {
    const int every = *(const int *)user_data;
    if (info->iteration % every == 0 || info->done || info->density_control_fired) {
        printf("iter %5llu  loss %12.6f  spheres %zu%s\n",
               (unsigned long long)info->iteration, info->total_loss, info->num_spheres,
               info->density_control_fired ? "  (density control)" : "");
    }
    return 0; /* nonzero would cancel */
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mesh> [preset] [num_spheres] [iterations] [seed] [out.json]\n", argv[0]);
        return 2;
    }
    const char *mesh_path = argv[1];
    const char *preset = argc > 2 ? argv[2] : "MorphIt-B";
    long long spheres = argc > 3 ? atoll(argv[3]) : 20;
    long long iterations = argc > 4 ? atoll(argv[4]) : 300;
    long long seed = argc > 5 ? atoll(argv[5]) : 42;
    const char *out_path = argc > 6 ? argv[6] : "spheres.json";

    printf("MorphIt %s\n", morphit_version());

    morphit_mesh *mesh = NULL;
    morphit_config *config = NULL;
    morphit_session *session = NULL;

    CHECK(morphit_mesh_load(mesh_path, &mesh));
    morphit_mesh_info info;
    CHECK(morphit_mesh_get_info(mesh, &info));
    printf("mesh: %zu faces, volume %.6g\n", info.num_faces, info.volume);

    CHECK(morphit_config_new(preset, &config));
    CHECK(morphit_config_set_i64(config, "model.num_spheres", spheres));
    CHECK(morphit_config_set_i64(config, "training.iterations", iterations));
    CHECK(morphit_config_set_i64(config, "random_seed", seed));

    CHECK(morphit_session_new(mesh, config, &session));
    /* The session keeps what it needs; both may be released now. */
    morphit_mesh_free(mesh);
    morphit_config_free(config);

    char device[128];
    size_t needed = 0;
    CHECK(morphit_session_device(session, device, sizeof device, &needed));
    printf("device: %s\n", device);

    int every = 50;
    CHECK(morphit_run(session, on_progress, &every)); /* runs and finalizes */

    size_t count = 0;
    CHECK(morphit_get_spheres(session, NULL, NULL, NULL, 0, &count));
    double *centers = (double *)malloc(3 * count * sizeof(double));
    double *radii = (double *)malloc(count * sizeof(double));
    CHECK(morphit_get_spheres(session, centers, radii, NULL, count, &count));
    printf("%zu spheres, first ones:\n", count);
    for (size_t i = 0; i < count && i < 5; ++i) {
        printf("  c = (% .4f, % .4f, % .4f)  r = %.4f\n", centers[3 * i], centers[3 * i + 1],
               centers[3 * i + 2], radii[i]);
    }
    free(centers);
    free(radii);

    CHECK(morphit_result_save(session, out_path));
    printf("wrote %s\n", out_path);
    CHECK(morphit_session_free(session));
    return 0;
}
