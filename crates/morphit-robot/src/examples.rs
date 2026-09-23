//! The bundled example library (`web/examples`): object meshes (with
//! thumbnails and pre-baked URDFs) and robot packages (with pre-baked
//! spherical URDFs), in the same order and with the same labels as the
//! Python API's registries.

/// A registered example object.
pub struct ExampleObject {
    pub name: &'static str,
    pub label: &'static str,
    pub filename: &'static str,
    pub default: bool,
}

const fn obj(name: &'static str, label: &'static str, filename: &'static str) -> ExampleObject {
    ExampleObject { name, label, filename, default: false }
}

pub const EXAMPLE_OBJECTS: &[ExampleObject] = &[
    ExampleObject {
        name: "bunny",
        label: "bunny.obj — Stanford Bunny",
        filename: "bunny.obj",
        default: true,
    },
    obj("link0", "link0.obj — Franka FR3 base link", "link0.obj"),
    obj("teapot", "teapot.obj — Porcelain teapot", "teapot.obj"),
    obj("dog_bowl", "dog_bowl.obj — Ceramic dog bowl", "dog_bowl.obj"),
    obj("boot", "boot.obj — Suede ankle boot", "boot.obj"),
    obj("mug", "mug.obj — Coffee mug", "mug.obj"),
    obj("bowl", "bowl.obj — Ceramic bowl", "bowl.obj"),
    obj("panda_toy", "panda_toy.obj — Panda figurine", "panda_toy.obj"),
    obj("planter", "planter.obj — Plant pot", "planter.obj"),
    obj("foobler", "foobler.obj — Treat-dispenser ball", "foobler.obj"),
    obj("frother", "frother.obj — Milk frother", "frother.obj"),
    obj("cooker", "cooker.obj — Pressure cooker", "cooker.obj"),
    obj("rhino", "rhino.obj — Rhino figurine", "rhino.obj"),
    obj("lion", "lion.obj — Lion figurine", "lion.obj"),
    obj("school_bus", "school_bus.obj — Toy school bus", "school_bus.obj"),
    obj("speaker", "speaker.obj — Portable speaker", "speaker.obj"),
    obj("toaster", "toaster.obj — 2-slice toaster", "toaster.obj"),
    obj("nesquik", "nesquik.obj — Nesquik canister", "nesquik.obj"),
    obj("coffee_jar", "coffee_jar.obj — Coffee jar", "coffee_jar.obj"),
    obj("pitcher", "pitcher.obj — Porcelain pitcher", "pitcher.obj"),
];

/// A registered example robot package.
pub struct ExampleRobot {
    pub name: &'static str,
    pub label: &'static str,
    /// Package folder under `web/examples`.
    pub folder: &'static str,
    /// URDF basename inside the folder.
    pub urdf: &'static str,
    /// Pre-baked spherical URDF under `web/examples`.
    pub spherical_urdf: &'static str,
    pub default: bool,
}

const fn robot(
    name: &'static str,
    label: &'static str,
    folder: &'static str,
    urdf: &'static str,
    spherical_urdf: &'static str,
) -> ExampleRobot {
    ExampleRobot { name, label, folder, urdf, spherical_urdf, default: false }
}

pub const EXAMPLE_ROBOTS: &[ExampleRobot] = &[
    ExampleRobot {
        name: "kinova",
        label: "Kinova m1n4s200",
        folder: "kinova_description",
        urdf: "m1n4s200_standalone.urdf",
        spherical_urdf: "kinova.spherical.urdf",
        default: true,
    },
    robot("panda", "Franka Emika Panda", "franka_panda", "panda.urdf", "panda.spherical.urdf"),
    robot("ur5", "Universal Robots UR5", "ur5", "ur5_gripper.urdf", "ur5.spherical.urdf"),
    robot("kuka_iiwa", "KUKA LBR iiwa", "kuka_iiwa", "model.urdf", "kuka_iiwa.spherical.urdf"),
    robot("yumi", "ABB YuMi (IRB 14000)", "yumi", "yumi.urdf", "yumi.spherical.urdf"),
    robot("spot", "Boston Dynamics Spot (quadruped)", "spot", "spot.urdf", "spot.spherical.urdf"),
    robot("fetch", "Fetch (mobile manipulator)", "fetch", "fetch.urdf", "fetch.spherical.urdf"),
    robot("valkyrie", "NASA Valkyrie (humanoid)", "valkyrie", "valkyrie_A.urdf", "valkyrie.spherical.urdf"),
    robot(
        "fanuc_m20ia",
        "Fanuc M-20iA (industrial)",
        "fanuc_m20ia",
        "m20ia10l.urdf",
        "fanuc_m20ia.spherical.urdf",
    ),
    robot(
        "abb_irb2400",
        "ABB IRB 2400 (industrial)",
        "abb_irb2400",
        "irb2400.urdf",
        "abb_irb2400.spherical.urdf",
    ),
];
