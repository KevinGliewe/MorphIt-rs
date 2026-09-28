// MorphIt C++ robot example: pack every collision mesh of a URDF package and
// write the spherical URDF.
//
//   robot_cpp <package folder> [num_spheres] [iterations] [out.urdf]

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "morphit.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <package folder> [num_spheres] [iterations] [out.urdf]\n";
        return 2;
    }
    try {
        auto robot = morphit::RobotPackage::from_folder(argv[1]);
        auto report = robot.inspect();
        auto items = report.pack_items();
        std::cout << items.size() << " collision meshes to pack\n";

        morphit::PackParams params;
        params.num_spheres = argc > 2 ? static_cast<std::size_t>(std::atoll(argv[2])) : 10;
        params.iterations = argc > 3 ? static_cast<std::size_t>(std::atoll(argv[3])) : 100;
        params.seed = 0;
        for (std::size_t i = 0; i < items.size(); ++i) {
            auto session = robot.pack_link(report, i, params);
            session.run();
            robot.set_link_result(report, i, session);
            std::cout << "  " << items[i].link << "[" << items[i].collision_index << "]: " << session.radii().size()
                      << " spheres\n";
        }

        auto assembled = robot.assemble(report, "#3399ff", 0.6);
        const std::string out = argc > 4 ? argv[4] : "robot.spherical.urdf";
        std::ofstream(out) << assembled.urdf;
        std::cout << "replaced " << assembled.stats.mesh_collisions_replaced << " mesh collisions; wrote " << out
                  << "\n";
    } catch (const morphit::Error& e) {
        std::cerr << "error " << e.status() << ": " << e.what() << "\n";
        return 1;
    }
    return 0;
}
