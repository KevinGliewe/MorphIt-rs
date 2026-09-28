/*
 * MorphIt C++ API: a header-only C++17 wrapper of the C API (morphit.h).
 *
 *   #include "morphit.hpp"
 *
 *   auto mesh = morphit::Mesh::load("bunny.obj");
 *   morphit::Config config("MorphIt-B");
 *   config.set("model.num_spheres", 64).set("random_seed", 42);
 *   morphit::Session session(mesh, config);
 *   session.run([](const morphit::StepInfo& s) { return true; });  // false stops
 *   auto centers = session.centers();                                // std::vector<std::array<double, 3>>
 *   session.save("bunny.json");
 *
 * Errors throw morphit::Error (with the C status code). Handles are RAII:
 * Mesh is copyable (meshes are immutable and shared); Config, Session,
 * RobotPackage and InspectionReport are move-only. The threading rules are
 * the C API's: every object may be used from any thread; on a Session the
 * read functions and cancel() may be called while run() is active elsewhere.
 *
 * In CMake: find_package(morphit CONFIG) and link morphit::morphit_cpp.
 */
#ifndef MORPHIT_HPP
#define MORPHIT_HPP

#include "morphit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace morphit {

/// A failed call: `status()` is the C status code, `what()` the message.
class Error : public std::runtime_error {
public:
    Error(morphit_status status, const std::string& message) : std::runtime_error(message), status_(status) {}
    morphit_status status() const noexcept { return status_; }

private:
    morphit_status status_;
};

using StepInfo = morphit_step_info;
using StateInfo = morphit_state_info;
using MeshInfo = morphit_mesh_info;
using QualityMetrics = morphit_quality_metrics;
using AssembleStats = morphit_assemble_stats;
using Vec3 = std::array<double, 3>;
using Triangle = std::array<std::uint32_t, 3>;

namespace detail {

inline morphit_status check(morphit_status s) {
    if (s != MORPHIT_OK && s != MORPHIT_DONE) {
        throw Error(s, morphit_last_error());
    }
    return s;
}

/// The C two-call pattern for string outputs.
template <class F>
std::string read_string(F&& f) {
    std::size_t needed = 0;
    check(f(static_cast<char*>(nullptr), std::size_t{0}, &needed));
    std::string s(needed, '\0');
    check(f(s.data(), s.size(), &needed));
    if (!s.empty()) s.pop_back();  // the NUL terminator
    return s;
}

inline std::vector<double> flatten(const std::vector<Vec3>& v) {
    std::vector<double> out;
    out.reserve(v.size() * 3);
    for (const auto& p : v) out.insert(out.end(), p.begin(), p.end());
    return out;
}

inline std::vector<Vec3> unflatten(const std::vector<double>& flat) {
    std::vector<Vec3> out(flat.size() / 3);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = {flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]};
    return out;
}

}  // namespace detail

/// Library version, e.g. "0.1.0".
inline std::string version() { return morphit_version(); }

/// Size the worker pool before the first pack (default: all cores).
inline void set_num_threads(std::size_t n) { detail::check(morphit_set_num_threads(n)); }

/// GPU adapter names; index i is the N of device "gpu:N".
inline std::vector<std::string> devices() {
    std::size_t n = 0;
    detail::check(morphit_device_count(&n));
    std::vector<std::string> out;
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(detail::read_string([i](char* b, std::size_t c, std::size_t* need) {
            return morphit_device_name(i, b, c, need);
        }));
    }
    return out;
}

/// Forward the library's log messages (level 1 = error ... 5 = trace) to
/// `callback`, from any thread. Process-wide; call once.
inline void set_log_callback(std::function<void(int, std::string_view)> callback, int max_level = 3) {
    static std::function<void(int, std::string_view)> stored;
    stored = std::move(callback);
    detail::check(morphit_set_log_callback(
        [](int level, const char* message, void*) {
            try {
                if (stored) stored(level, message);
            } catch (...) {
            }
        },
        nullptr, max_level));
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

/// An immutable triangle mesh. Copies share the same data.
class Mesh {
public:
    /// Load an OBJ, STL, PLY or DAE file.
    static Mesh load(const std::string& path) {
        morphit_mesh* m = nullptr;
        detail::check(morphit_mesh_load(path.c_str(), &m));
        return Mesh(m);
    }

    /// Load from file contents; `ext` is "obj", "stl", "ply" or "dae".
    static Mesh from_bytes(const std::vector<std::uint8_t>& data, const std::string& ext, const std::string& name = {}) {
        morphit_mesh* m = nullptr;
        detail::check(morphit_mesh_from_bytes(data.data(), data.size(), ext.c_str(), name.empty() ? nullptr : name.c_str(), &m));
        return Mesh(m);
    }

    /// Build from vertices and counter-clockwise triangles (zero-based indices).
    static Mesh from_arrays(const std::vector<Vec3>& vertices, const std::vector<Triangle>& triangles) {
        std::vector<double> xyz = detail::flatten(vertices);
        std::vector<std::uint32_t> tri;
        tri.reserve(triangles.size() * 3);
        for (const auto& t : triangles) tri.insert(tri.end(), t.begin(), t.end());
        morphit_mesh* m = nullptr;
        detail::check(morphit_mesh_from_arrays(xyz.data(), vertices.size(), tri.data(), triangles.size(), &m));
        return Mesh(m);
    }

    MeshInfo info() const {
        MeshInfo i{};
        detail::check(morphit_mesh_get_info(get(), &i));
        return i;
    }

    /// Which points lie inside the mesh.
    std::vector<bool> contains(const std::vector<Vec3>& points) const {
        std::vector<double> xyz = detail::flatten(points);
        std::vector<std::uint8_t> inside(points.size());
        detail::check(morphit_mesh_contains(get(), xyz.data(), points.size(), inside.data()));
        return std::vector<bool>(inside.begin(), inside.end());
    }

    /// The mesh as packing prepares it (convex hull of each body, then the
    /// union of overlapping bodies).
    Mesh prepared(bool union_overlapping_bodies = true, bool convex_hull = false) const {
        morphit_mesh* m = nullptr;
        detail::check(morphit_mesh_prepare(get(), union_overlapping_bodies, convex_hull, &m));
        return Mesh(m);
    }

    /// What `prepared` does, as JSON.
    std::string prep_report_json(bool union_overlapping_bodies = true, bool convex_hull = false) const {
        return detail::read_string([&](char* b, std::size_t c, std::size_t* n) {
            return morphit_mesh_prep_report_json(get(), union_overlapping_bodies, convex_hull, b, c, n);
        });
    }

    /// Write .obj or .stl (by extension).
    void save(const std::string& path) const { detail::check(morphit_mesh_save(get(), path.c_str())); }

    const morphit_mesh* get() const noexcept { return mesh_.get(); }

private:
    explicit Mesh(morphit_mesh* m) : mesh_(m, morphit_mesh_free) {}
    std::shared_ptr<morphit_mesh> mesh_;
    friend class Session;
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Optimizer configuration: a preset plus dotted-key overrides.
class Config {
public:
    /// A preset: MorphIt-V, -S, -B (default), -Obj or -Obj-mass.
    explicit Config(const std::string& preset = "MorphIt-B") {
        morphit_config* c = nullptr;
        detail::check(morphit_config_new(preset.c_str(), &c));
        config_.reset(c);
    }

    /// A config from nested JSON (as under "config" in a result file).
    static Config from_json(const std::string& json) {
        morphit_config* c = nullptr;
        detail::check(morphit_config_from_json(json.c_str(), &c));
        return Config(c);
    }

    Config clone() const {
        morphit_config* c = nullptr;
        detail::check(morphit_config_clone(get(), &c));
        return Config(c);
    }

    Config& set(const std::string& key, double v) {
        detail::check(morphit_config_set_f64(get(), key.c_str(), v));
        return *this;
    }
    Config& set(const std::string& key, std::int64_t v) {
        detail::check(morphit_config_set_i64(get(), key.c_str(), v));
        return *this;
    }
    Config& set(const std::string& key, int v) { return set(key, static_cast<std::int64_t>(v)); }
    Config& set(const std::string& key, bool v) {
        detail::check(morphit_config_set_bool(get(), key.c_str(), v));
        return *this;
    }
    Config& set(const std::string& key, const std::string& v) {
        detail::check(morphit_config_set_str(get(), key.c_str(), v.c_str()));
        return *this;
    }
    Config& set(const std::string& key, const char* v) { return set(key, std::string(v)); }

    /// Apply `{"dotted.key": value, ...}`; all or nothing.
    Config& set_json(const std::string& json_updates) {
        detail::check(morphit_config_set_json(get(), json_updates.c_str()));
        return *this;
    }

    double get_f64(const std::string& key) const {
        double v = 0;
        detail::check(morphit_config_get_f64(get(), key.c_str(), &v));
        return v;
    }
    std::int64_t get_i64(const std::string& key) const {
        std::int64_t v = 0;
        detail::check(morphit_config_get_i64(get(), key.c_str(), &v));
        return v;
    }

    std::string to_json() const {
        return detail::read_string([&](char* b, std::size_t c, std::size_t* n) { return morphit_config_to_json(get(), b, c, n); });
    }

    morphit_config* get() const noexcept { return config_.get(); }

private:
    explicit Config(morphit_config* c) : config_(c) {}
    struct Free {
        void operator()(morphit_config* c) const noexcept { morphit_config_free(c); }
    };
    std::unique_ptr<morphit_config, Free> config_;
};

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

struct QualityOptions {
    QualityOptions() { detail::check(morphit_quality_options_default(&c)); }
    morphit_quality_options c;
};

/// How `Session::run` ended.
enum class RunOutcome { Completed, Converged, Cancelled };

/// A packing run. Move-only.
class Session {
public:
    /// Prepare the mesh, draw the samples and place the initial spheres.
    Session(const Mesh& mesh, const Config& config) {
        morphit_session* s = nullptr;
        detail::check(morphit_session_new(mesh.get(), config.get(), &s));
        session_.reset(s);
    }

    /// One iteration; `std::nullopt` when no iterations are left.
    std::optional<StepInfo> step() {
        StepInfo info{};
        if (detail::check(morphit_step(get(), &info)) == MORPHIT_DONE) return std::nullopt;
        return info;
    }

    /// Step until done, then finalize. `on_step` returning false stops the
    /// run (the session can be run again later); an exception it throws stops
    /// the run and is rethrown here. `cancel()` from another thread stops it too.
    RunOutcome run(std::function<bool(const StepInfo&)> on_step = {}) {
        struct Ctx {
            std::function<bool(const StepInfo&)>* f;
            std::exception_ptr error;
            bool converged = false;
        } ctx{&on_step, nullptr, false};
        auto trampoline = [](const morphit_step_info* info, void* user) -> int {
            auto* c = static_cast<Ctx*>(user);
            c->converged = info->converged != 0;
            if (!*c->f) return 0;
            try {
                return (*c->f)(*info) ? 0 : 1;
            } catch (...) {
                c->error = std::current_exception();
                return 1;
            }
        };
        morphit_status s = morphit_run(get(), trampoline, &ctx);
        if (ctx.error) std::rethrow_exception(ctx.error);
        if (s == MORPHIT_ERR_CANCELLED) return RunOutcome::Cancelled;
        detail::check(s);
        return ctx.converged ? RunOutcome::Converged : RunOutcome::Completed;
    }

    /// Stop a running `run()` after the current iteration (any thread).
    void cancel() { detail::check(morphit_cancel(get())); }

    /// Remove spheres whose centers ended outside the mesh; returns how many.
    std::size_t finalize() {
        std::size_t pruned = 0;
        detail::check(morphit_finalize(get(), &pruned));
        return pruned;
    }

    StateInfo state() const {
        StateInfo st{};
        detail::check(morphit_state(get(), &st));
        return st;
    }

    std::vector<Vec3> centers() const { return detail::unflatten(read(morphit_get_centers)); }
    std::vector<double> radii() const { return read(morphit_get_radii); }
    std::vector<double> masses() const { return read(morphit_get_masses); }

    /// The last iteration's info (throws before the first step).
    StepInfo last_step() const {
        StepInfo info{};
        detail::check(morphit_get_last_step(get(), &info));
        return info;
    }

    /// The current spheres in the Python MorphIt JSON schema.
    std::string result_json() const { return str(morphit_result_json); }
    void save(const std::string& path) const { detail::check(morphit_result_save(get(), path.c_str())); }
    std::string config_json() const { return str(morphit_session_config_json); }
    std::string device() const { return str(morphit_session_device); }
    std::string mesh_prep_json() const { return str(morphit_session_mesh_prep_json); }
    /// The per-iteration history (waits for a running iteration).
    std::string history_json() const { return str(morphit_history_json); }

    /// The mesh as packed (after mesh preparation).
    Mesh mesh() const {
        morphit_mesh* m = nullptr;
        detail::check(morphit_session_mesh(get(), &m));
        return Mesh(m);
    }

    /// Quality of the current spheres against the prepared mesh.
    QualityMetrics evaluate(const QualityOptions& options = {}) const {
        QualityMetrics q{};
        detail::check(morphit_session_evaluate(get(), &options.c, &q));
        return q;
    }

    morphit_session* get() const noexcept { return session_.get(); }

private:
    explicit Session(morphit_session* s) : session_(s) {}
    friend class RobotPackage;

    template <class F>
    std::vector<double> read(F f) const {
        std::size_t n = 0;
        detail::check(f(get(), nullptr, 0, &n));
        std::vector<double> v(n);
        detail::check(f(get(), v.data(), v.size(), &n));
        v.resize(n);
        return v;
    }
    template <class F>
    std::string str(F f) const {
        return detail::read_string([&](char* b, std::size_t c, std::size_t* n) { return f(get(), b, c, n); });
    }

    struct Free {
        void operator()(morphit_session* s) const noexcept {
            // A run on another thread keeps the session busy: stop it first.
            morphit_cancel(s);
            while (morphit_session_free(s) == MORPHIT_ERR_BUSY) std::this_thread::yield();
        }
    };
    std::unique_ptr<morphit_session, Free> session_;
};

// ---------------------------------------------------------------------------
// Export and metrics
// ---------------------------------------------------------------------------

/// Options of `object_urdf` / `object_mjcf`.
struct ObjectOptions {
    ObjectOptions() {
        morphit_object_options d;
        detail::check(morphit_object_options_default(&d));
        rgba = {d.rgba[0], d.rgba[1], d.rgba[2], d.rgba[3]};
        total_mass = d.total_mass;
        anchored = d.anchored != 0;
        decimals = d.decimals;
    }
    std::string name = "object";
    std::array<double, 4> rgba{};
    double total_mass = 1.0;
    bool anchored = false;
    int decimals = 6;
};

/// A generated model and the point its sphere positions are relative to.
struct ObjectModel {
    std::string text;
    Vec3 centroid{};
};

namespace detail {
template <class F>
ObjectModel object_model(F f, const std::vector<Vec3>& centers, const std::vector<double>& radii, const ObjectOptions& o) {
    if (centers.size() != radii.size()) throw Error(MORPHIT_ERR_INVALID_ARG, "centers and radii differ in length");
    std::vector<double> c = flatten(centers);
    morphit_object_options opts{o.name.c_str(), {o.rgba[0], o.rgba[1], o.rgba[2], o.rgba[3]}, o.total_mass, o.anchored, o.decimals};
    ObjectModel m;
    m.text = read_string([&](char* b, std::size_t cap, std::size_t* n) {
        return f(c.data(), radii.data(), radii.size(), &opts, b, cap, n, m.centroid.data());
    });
    return m;
}
}  // namespace detail

/// The spheres as a URDF (one link per sphere on fixed joints).
inline ObjectModel object_urdf(const std::vector<Vec3>& centers, const std::vector<double>& radii, const ObjectOptions& options = {}) {
    return detail::object_model(morphit_object_urdf, centers, radii, options);
}

/// The spheres as MJCF for MuJoCo.
inline ObjectModel object_mjcf(const std::vector<Vec3>& centers, const std::vector<double>& radii, const ObjectOptions& options = {}) {
    return detail::object_model(morphit_object_mjcf, centers, radii, options);
}

/// Quality of spheres against a mesh (pass the prepared mesh; `masses` may be
/// empty to derive them from the density).
inline QualityMetrics evaluate_packing(const Mesh& mesh, const std::vector<Vec3>& centers, const std::vector<double>& radii,
                                       const std::vector<double>& masses = {}, const QualityOptions& options = {}) {
    if (centers.size() != radii.size() || (!masses.empty() && masses.size() != radii.size()))
        throw Error(MORPHIT_ERR_INVALID_ARG, "centers, radii and masses differ in length");
    std::vector<double> c = detail::flatten(centers);
    QualityMetrics q{};
    detail::check(morphit_evaluate_packing(mesh.get(), c.data(), radii.data(), masses.empty() ? nullptr : masses.data(), radii.size(), &options.c, &q));
    return q;
}

// ---------------------------------------------------------------------------
// Robots
// ---------------------------------------------------------------------------

/// One collision to replace with spheres.
struct PackItem {
    std::string link;
    std::size_t collision_index;
};

/// The inspection of one URDF. Move-only; immutable.
class InspectionReport {
public:
    /// The full report as JSON (collisions, actions, warnings, ...).
    std::string json() const {
        return detail::read_string([&](char* b, std::size_t c, std::size_t* n) { return morphit_report_json(get(), b, c, n); });
    }

    /// The collisions to pack, in order; their index is the `item` of
    /// `RobotPackage::pack_link`.
    std::vector<PackItem> pack_items() const {
        std::size_t n = 0;
        detail::check(morphit_report_pack_count(get(), &n));
        std::vector<PackItem> out;
        for (std::size_t i = 0; i < n; ++i) {
            PackItem it{{}, 0};
            it.link = detail::read_string([&](char* b, std::size_t c, std::size_t* need) {
                return morphit_report_pack_item(get(), i, b, c, need, &it.collision_index);
            });
            out.push_back(std::move(it));
        }
        return out;
    }

    const morphit_robot_report* get() const noexcept { return report_.get(); }

private:
    explicit InspectionReport(morphit_robot_report* r) : report_(r) {}
    friend class RobotPackage;
    struct Free {
        void operator()(morphit_robot_report* r) const noexcept { morphit_report_free(r); }
    };
    std::unique_ptr<morphit_robot_report, Free> report_;
};

/// Parameters of `RobotPackage::pack_link` (the web API's defaults).
struct PackParams {
    std::string variant = "MorphIt-B";
    std::size_t num_spheres = 20;
    std::size_t iterations = 200;
    std::optional<std::uint64_t> seed;
    bool union_overlapping_bodies = true;
    bool convex_hull = false;
    /// The web UI's advanced overrides as a JSON object, or empty.
    std::string advanced_json;
};

/// A URDF package (URDF plus meshes) held in memory. Move-only.
class RobotPackage {
public:
    RobotPackage() {
        morphit_robot* r = nullptr;
        detail::check(morphit_robot_new(&r));
        robot_.reset(r);
    }

    static RobotPackage from_folder(const std::string& path) {
        morphit_robot* r = nullptr;
        detail::check(morphit_robot_from_folder(path.c_str(), &r));
        return RobotPackage(r);
    }

    static RobotPackage from_zip(const std::vector<std::uint8_t>& data) {
        morphit_robot* r = nullptr;
        detail::check(morphit_robot_from_zip(data.data(), data.size(), &r));
        return RobotPackage(r);
    }

    void add_file(const std::string& path, const std::vector<std::uint8_t>& data) {
        detail::check(morphit_robot_add_file(get(), path.c_str(), data.data(), data.size()));
    }

    /// Inspect a URDF (by file name; empty: the only one).
    InspectionReport inspect(const std::string& urdf = {}) const {
        morphit_robot_report* rep = nullptr;
        detail::check(morphit_robot_inspect(get(), urdf.empty() ? nullptr : urdf.c_str(), &rep));
        return InspectionReport(rep);
    }

    /// A session packing pack item `item`; run it, then `set_link_result`.
    Session pack_link(const InspectionReport& report, std::size_t item, const PackParams& p = {}, const std::string& device = "auto") const {
        morphit_pack_params c{p.variant.c_str(), p.num_spheres, p.iterations, p.seed ? static_cast<std::int64_t>(*p.seed) : -1,
                              p.union_overlapping_bodies, p.convex_hull, p.advanced_json.empty() ? nullptr : p.advanced_json.c_str()};
        morphit_session* s = nullptr;
        detail::check(morphit_robot_pack_link(get(), report.get(), item, &c, device.c_str(), &s));
        return Session(s);
    }

    /// Record the session's spheres as the result of pack item `item`.
    void set_link_result(const InspectionReport& report, std::size_t item, const Session& session) {
        detail::check(morphit_robot_set_link_result(get(), report.get(), item, session.get()));
    }

    /// Record the spheres of `link[collision_index]` from a result JSON.
    void set_link_result_json(const std::string& link, std::size_t collision_index, const std::string& result_json) {
        detail::check(morphit_robot_set_link_result_json(get(), link.c_str(), collision_index, result_json.c_str()));
    }

    void clear_link_results() { detail::check(morphit_robot_clear_link_results(get())); }

    struct Assembled {
        std::string urdf;
        AssembleStats stats;
    };

    /// The URDF with the packed collisions replaced by sphere links.
    Assembled assemble(const InspectionReport& report, const std::string& base_color = {}, double color_variation = 0.0) const {
        Assembled a{{}, {}};
        a.urdf = detail::read_string([&](char* b, std::size_t c, std::size_t* n) {
            return morphit_robot_assemble(get(), report.get(), base_color.empty() ? nullptr : base_color.c_str(), color_variation, b, c, n, &a.stats);
        });
        return a;
    }

    morphit_robot* get() const noexcept { return robot_.get(); }

private:
    explicit RobotPackage(morphit_robot* r) : robot_(r) {}
    struct Free {
        void operator()(morphit_robot* r) const noexcept { morphit_robot_free(r); }
    };
    std::unique_ptr<morphit_robot, Free> robot_;
};

}  // namespace morphit

#endif  // MORPHIT_HPP
