// Minimal MorphIt C++ example: pack one mesh, save the spheres, write a URDF.
//
//   pack_cpp <mesh> [preset] [num_spheres] [iterations] [seed] [out.json]

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "morphit.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <mesh> [preset] [num_spheres] [iterations] [seed] [out.json]\n";
        return 2;
    }
    try {
        const std::string preset = argc > 2 ? argv[2] : "MorphIt-B";
        const long long spheres = argc > 3 ? std::atoll(argv[3]) : 20;
        const long long iterations = argc > 4 ? std::atoll(argv[4]) : 300;
        const long long seed = argc > 5 ? std::atoll(argv[5]) : 42;
        const std::string out = argc > 6 ? argv[6] : "spheres.json";

        std::cout << "MorphIt " << morphit::version() << "\n";
        auto mesh = morphit::Mesh::load(argv[1]);
        std::cout << "mesh: " << mesh.info().num_faces << " faces, volume " << mesh.info().volume << "\n";

        morphit::Config config(preset);
        config.set("model.num_spheres", static_cast<std::int64_t>(spheres))
            .set("training.iterations", static_cast<std::int64_t>(iterations))
            .set("random_seed", static_cast<std::int64_t>(seed));

        morphit::Session session(mesh, config);
        std::cout << "device: " << session.device() << "\n";
        auto outcome = session.run([&](const morphit::StepInfo& s) {
            if (s.iteration % 50 == 0 || s.done) {
                std::printf("iter %5llu  loss %12.6f  spheres %zu\n", static_cast<unsigned long long>(s.iteration),
                            s.total_loss, s.num_spheres);
            }
            return true;  // false would stop the run
        });
        std::cout << (outcome == morphit::RunOutcome::Converged ? "converged" : "completed") << "\n";

        auto centers = session.centers();
        auto radii = session.radii();
        std::cout << centers.size() << " spheres, first: c = (" << centers[0][0] << ", " << centers[0][1] << ", "
                  << centers[0][2] << ")  r = " << radii[0] << "\n";
        session.save(out);

        auto q = session.evaluate();
        std::printf("coverage %.3f  outside %.3f  mean surface distance %.2f mm\n", q.r_in, q.r_out, q.d_avg_mm);

        morphit::ObjectOptions opts;
        opts.name = "object";
        auto urdf = morphit::object_urdf(centers, radii, opts);
        std::ofstream(out + ".urdf") << urdf.text;
        std::cout << "wrote " << out << " and " << out << ".urdf\n";
    } catch (const morphit::Error& e) {
        std::cerr << "error " << e.status() << ": " << e.what() << "\n";
        return 1;
    }
    return 0;
}
