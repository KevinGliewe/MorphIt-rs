// Tests of the C++ wrapper (morphit.hpp). Plain checks, no framework:
//   test_morphit <repo root>
// Exits nonzero on the first failure.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "morphit.hpp"

#define EXPECT(cond)                                                                   \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

template <class F>
static morphit_status throws_status(F f) {
    try {
        f();
    } catch (const morphit::Error& e) {
        return e.status();
    }
    return MORPHIT_OK;
}

static morphit::Config small_config(std::int64_t spheres, std::int64_t iterations) {
    morphit::Config c("MorphIt-B");
    c.set("model.num_spheres", spheres)
        .set("training.iterations", iterations)
        .set("random_seed", std::int64_t{7})
        .set("model.num_inside_samples", std::int64_t{400})
        .set("model.num_surface_samples", std::int64_t{400})
        .set("model.device", "cpu");
    return c;
}

static morphit::Mesh unit_box() {
    std::vector<morphit::Vec3> v = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    std::vector<morphit::Triangle> t = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
                                        {2, 3, 7}, {2, 7, 6}, {0, 4, 7}, {0, 7, 3}, {1, 2, 6}, {1, 6, 5}};
    return morphit::Mesh::from_arrays(v, t);
}

static void test_errors() {
    EXPECT(throws_status([] { morphit::Mesh::load("definitely-missing.obj"); }) == MORPHIT_ERR_IO);
    EXPECT(throws_status([] { morphit::Config("MorphIt-X"); }) == MORPHIT_ERR_CONFIG);
    EXPECT(throws_status([] { morphit::Config().set("training.nope", 1.0); }) == MORPHIT_ERR_CONFIG);
    try {
        morphit::Config().set("training.nope", 1.0);
    } catch (const morphit::Error& e) {
        EXPECT(std::string(e.what()).find("nope") != std::string::npos);
    }
}

static void test_mesh_and_config() {
    auto box = unit_box();
    EXPECT(std::fabs(box.info().volume - 1.0) < 1e-12);
    auto inside = box.contains({{0.3, 0.4, 0.45}, {2, 0.4, 0.45}});
    EXPECT(inside.size() == 2 && inside[0] && !inside[1]);
    auto copy = box;  // shared
    EXPECT(copy.get() == box.get());

    auto c = small_config(5, 10);
    EXPECT(c.get_i64("model.num_spheres") == 5);
    auto d = c.clone();
    d.set("model.num_spheres", std::int64_t{9});
    EXPECT(c.get_i64("model.num_spheres") == 5 && d.get_i64("model.num_spheres") == 9);
    EXPECT(c.to_json().find("\"num_spheres\": 5") != std::string::npos);
}

static void test_session(const morphit::Mesh& mesh) {
    auto config = small_config(6, 40);
    morphit::Session a(mesh, config);
    int calls = 0;
    auto outcome = a.run([&](const morphit::StepInfo&) { return ++calls > 0; });
    EXPECT(outcome != morphit::RunOutcome::Cancelled);
    EXPECT(calls == 40);
    EXPECT(a.state().state == MORPHIT_STATE_FINALIZED);

    // Stepping gives the same result as running.
    morphit::Session b(mesh, config);
    std::size_t steps = 0;
    while (b.step()) ++steps;
    EXPECT(steps == 40);
    b.finalize();
    EXPECT(a.centers() == b.centers() && a.radii() == b.radii());
    EXPECT(a.result_json() == b.result_json());

    // A callback returning false cancels; the session can continue.
    morphit::Session c(mesh, config);
    EXPECT(c.run([](const morphit::StepInfo& s) { return s.iteration < 3; }) == morphit::RunOutcome::Cancelled);
    EXPECT(c.state().iteration == 4);
    EXPECT(c.run() != morphit::RunOutcome::Cancelled);

    // An exception in the callback stops the run and propagates.
    morphit::Session d(mesh, config);
    bool caught = false;
    try {
        d.run([](const morphit::StepInfo&) -> bool { throw std::runtime_error("boom"); });
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "boom";
    }
    EXPECT(caught);

    // cancel() from another thread; reads work during the run.
    auto long_config = small_config(6, 100000);
    morphit::Session e(mesh, long_config);
    std::atomic<bool> started{false};
    morphit::RunOutcome result = morphit::RunOutcome::Completed;
    std::thread t([&] {
        result = e.run([&](const morphit::StepInfo&) {
            started = true;
            return true;
        });
    });
    while (!started) std::this_thread::yield();
    EXPECT(e.radii().size() == 6);
    EXPECT(e.state().running != 0);
    EXPECT(throws_status([&] { e.step(); }) == MORPHIT_ERR_BUSY);
    e.cancel();
    t.join();
    EXPECT(result == morphit::RunOutcome::Cancelled);

    // Export and metrics.
    auto q = a.evaluate();
    EXPECT(q.actual_n == a.radii().size() && q.r_in > 0);
    auto urdf = morphit::object_urdf(a.centers(), a.radii());
    EXPECT(urdf.text.find("<robot name=\"object\"") != std::string::npos);
    auto mjcf = morphit::object_mjcf(a.centers(), a.radii());
    EXPECT(mjcf.text.find("<mujoco") != std::string::npos);
    auto q2 = morphit::evaluate_packing(a.mesh(), a.centers(), a.radii());
    EXPECT(q2.actual_n == q.actual_n);
}

static void test_robot(const std::string& repo) {
    auto robot = morphit::RobotPackage::from_folder(repo + "/web/examples/kinova_description");
    auto report = robot.inspect();
    auto items = report.pack_items();
    EXPECT(items.size() >= 2);
    EXPECT(report.json().find("\"collisions\"") != std::string::npos);
    morphit::PackParams p;
    p.num_spheres = 4;
    p.iterations = 10;
    p.seed = 0;
    for (std::size_t i = 0; i < 2; ++i) {
        auto s = robot.pack_link(report, i, p, "cpu");
        s.run();
        robot.set_link_result(report, i, s);
    }
    auto assembled = robot.assemble(report, "#3399ff", 0.5);
    EXPECT(assembled.urdf.find("<sphere") != std::string::npos);
    EXPECT(assembled.stats.mesh_collisions_replaced == 2);
    p.num_spheres = 0;
    EXPECT(throws_status([&] { robot.pack_link(report, 0, p); }) == MORPHIT_ERR_INVALID_ARG);
}

int main(int argc, char** argv) {
    const std::string repo = argc > 1 ? argv[1] : ".";
    test_errors();
    test_mesh_and_config();
    test_session(morphit::Mesh::load(repo + "/crates/morphit/tests/fixtures/link0.obj"));
    test_robot(repo);
    std::printf("C++ wrapper: all checks passed\n");
    return 0;
}
