/*
 * MorphIt C threading example.
 *
 * 1. Packs one shared mesh with four sessions running concurrently on four
 *    threads, while the main thread polls their progress.
 * 2. Starts a long run on a worker thread, reads spheres from the main thread
 *    while it runs, then cancels it.
 *
 *   threads <mesh.obj|stl>
 */
#include <stdio.h>
#include <stdlib.h>

#include "morphit.h"

#ifdef _WIN32
#include <windows.h>
typedef HANDLE thread_t;
typedef DWORD thread_ret;
#define THREAD_CALL WINAPI
static void thread_spawn(thread_t *t, thread_ret(THREAD_CALL *fn)(void *), void *arg) {
    *t = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)fn, arg, 0, NULL);
}
static void thread_join(thread_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}
static void sleep_ms(int ms) { Sleep((DWORD)ms); }
#else
#include <pthread.h>
#include <time.h>
typedef pthread_t thread_t;
typedef void *thread_ret;
#define THREAD_CALL
static void thread_spawn(thread_t *t, thread_ret (*fn)(void *), void *arg) { pthread_create(t, NULL, fn, arg); }
static void thread_join(thread_t t) { pthread_join(t, NULL); }
static void sleep_ms(int ms) {
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
#endif

#define CHECK(call)                                                            \
    do {                                                                       \
        morphit_status st_ = (call);                                           \
        if (st_ < 0) {                                                         \
            fprintf(stderr, "%s failed (%d): %s\n", #call, (int)st_,           \
                    morphit_last_error());                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

typedef struct {
    morphit_session *session;
    morphit_status result;
} job_t;

static thread_ret THREAD_CALL run_job(void *arg) {
    job_t *job = (job_t *)arg;
    job->result = morphit_run(job->session, NULL, NULL);
    return 0;
}

static morphit_session *make_session(const morphit_mesh *mesh, long long spheres, long long iterations,
                                     long long seed) {
    morphit_config *config = NULL;
    morphit_session *session = NULL;
    CHECK(morphit_config_new("MorphIt-B", &config));
    CHECK(morphit_config_set_i64(config, "model.num_spheres", spheres));
    CHECK(morphit_config_set_i64(config, "training.iterations", iterations));
    CHECK(morphit_config_set_i64(config, "random_seed", seed));
    CHECK(morphit_session_new(mesh, config, &session));
    morphit_config_free(config);
    return session;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <mesh>\n", argv[0]);
        return 2;
    }
    morphit_mesh *mesh = NULL;
    CHECK(morphit_mesh_load(argv[1], &mesh));

    /* --- 1. four concurrent sessions sharing one mesh ------------------- */
    enum { N = 4 };
    job_t jobs[N];
    thread_t threads[N];
    for (int i = 0; i < N; ++i) {
        jobs[i].session = make_session(mesh, 16 + 4 * i, 300, 100 + i);
        thread_spawn(&threads[i], run_job, &jobs[i]);
    }
    for (;;) {
        int running = 0;
        printf("progress:");
        for (int i = 0; i < N; ++i) {
            morphit_state_info st;
            CHECK(morphit_state(jobs[i].session, &st));
            running += st.running || st.state == MORPHIT_STATE_RUNNING;
            printf("  [%d] %3llu/%llu", i, (unsigned long long)st.iteration,
                   (unsigned long long)st.total_iterations);
        }
        printf("\n");
        if (!running) break;
        sleep_ms(100);
    }
    for (int i = 0; i < N; ++i) {
        thread_join(threads[i]);
        size_t count = 0;
        CHECK(morphit_sphere_count(jobs[i].session, &count));
        printf("session %d: status %d, %zu spheres\n", i, (int)jobs[i].result, count);
        CHECK(morphit_session_free(jobs[i].session));
    }

    /* --- 2. poll a long run from another thread, then cancel it --------- */
    job_t job = {make_session(mesh, 24, 1000000, 7), MORPHIT_OK};
    thread_t worker;
    thread_spawn(&worker, run_job, &job);
    double centers[3 * 24], radii[24];
    for (int k = 0; k < 5; ++k) {
        sleep_ms(50);
        size_t count = 0;
        morphit_state_info st;
        CHECK(morphit_state(job.session, &st));
        CHECK(morphit_get_spheres(job.session, centers, radii, NULL, 24, &count)); /* consistent snapshot */
        printf("iteration %llu: sphere 0 at (%.4f, %.4f, %.4f) r=%.4f\n", (unsigned long long)st.iteration,
               centers[0], centers[1], centers[2], radii[0]);
        /* Mutating calls are refused while the run is active. */
        if (morphit_step(job.session, NULL) != MORPHIT_ERR_BUSY) {
            fprintf(stderr, "expected MORPHIT_ERR_BUSY\n");
            return 1;
        }
    }
    CHECK(morphit_cancel(job.session));
    thread_join(worker);
    printf("long run returned %d (MORPHIT_ERR_CANCELLED = %d): %s\n", (int)job.result,
           (int)MORPHIT_ERR_CANCELLED, job.result == MORPHIT_ERR_CANCELLED ? "ok" : "unexpected");
    size_t pruned = 0;
    CHECK(morphit_finalize(job.session, &pruned));
    CHECK(morphit_session_free(job.session));
    morphit_mesh_free(mesh);
    return job.result == MORPHIT_ERR_CANCELLED ? 0 : 1;
}
