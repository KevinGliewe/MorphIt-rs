// Check the npm package the way users get it: `npm pack`, install the tarball
// into a fresh project, then from Node.js pack a mesh, export a URDF, run the
// robot pipeline, and type-check a TypeScript file against the bundled types.
//   node tools/ci/test_npm_package.mjs [package dir]   (default target/npm/morphit-rs)
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const pkgDir = path.resolve(process.argv[2] ?? path.join(repo, "target/npm/morphit-rs"));
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const run = (cmd, args, cwd) => {
  try {
    return execFileSync(cmd, args, { cwd, stdio: ["ignore", "pipe", "inherit"], shell: process.platform === "win32" }).toString();
  } catch (e) {
    process.stderr.write(e.stdout?.toString() ?? "");
    throw new Error(`${path.basename(cmd)} ${args.join(" ")} failed (exit ${e.status})`);
  }
};

const work = fs.mkdtempSync(path.join(os.tmpdir(), "morphit-npm-"));
try {
  const tarball = run(npm, ["pack", pkgDir, "--pack-destination", work, "--silent"], work).trim().split(/\r?\n/).pop();
  fs.writeFileSync(path.join(work, "package.json"), JSON.stringify({ name: "morphit-npm-test", private: true, type: "module" }));
  run(npm, ["install", "--silent", "--no-audit", "--no-fund", path.join(work, tarball), "typescript@5"], work);

  const link0 = JSON.stringify(path.join(repo, "crates/morphit/tests/fixtures/link0.obj"));
  const kinova = JSON.stringify(path.join(repo, "web/examples/kinova_description"));
  fs.writeFileSync(
    path.join(work, "test.mjs"),
    `
import fs from "node:fs";
import path from "node:path";
import init, { Config, Mesh, Session, RobotPackage, objectUrdf, version } from "morphit-rs";
await init();
const assert = (c, m) => { if (!c) { console.error("check failed: " + m); process.exit(1); } };

const mesh = Mesh.fromBytes(fs.readFileSync(${link0}), "obj", "link0.obj");
const config = Config.fromPreset("MorphIt-B");
config.set("model.num_spheres", 8);
config.set("training.iterations", 30);
config.set("random_seed", 1);
config.set("model.device", "cpu");
const session = new Session(config, mesh);
while (!session.isDone) await session.stepMany({ maxSteps: 10 });
session.finalize();
const result = JSON.parse(session.resultJson());
assert(result.radii.length === session.radii().length && result.radii.length > 0, "result");
const urdf = objectUrdf(session.centers(), session.radii(), { name: "link0" }).text;
assert(urdf.includes('<robot name="link0"'), "urdf");

// Robot pipeline from files in memory.
const root = ${kinova};
const files = {};
const walk = (d) => { for (const e of fs.readdirSync(d, { withFileTypes: true })) {
  const p = path.join(d, e.name);
  if (e.isDirectory()) walk(p); else files[path.relative(root, p).split(path.sep).join("/")] = fs.readFileSync(p);
} };
walk(root);
const pkg = RobotPackage.fromFiles(files);
const report = pkg.inspect();
const items = report.collisions.filter((c) => c.action === "pack").slice(0, 2);
for (const item of items) {
  const s = pkg.packLink(item, { numSpheres: 4, iterations: 10, seed: 0 }, "cpu");
  while (!s.isDone) await s.stepMany({ maxSteps: 10 });
  s.finalize();
  pkg.setLinkResult(item.link_name, item.collision_index, s.resultJson());
}
const assembled = pkg.assemble(undefined, "#3399ff", 0.5);
assert(assembled.urdf.includes("<sphere") && assembled.meshCollisionsReplaced === 2, "assemble");
console.log("morphit-rs " + version() + " works in Node: " + result.radii.length + " spheres, robot assembled");
`,
  );
  process.stdout.write(run(process.execPath, ["test.mjs"], work));

  // The types: a small TypeScript program must type-check against the package.
  fs.writeFileSync(
    path.join(work, "check.ts"),
    `
import init, { Config, Mesh, Session, RobotPackage, objectUrdf, type StepInfo, type InspectionReport } from "morphit-rs";
export async function pack(bytes: Uint8Array): Promise<string> {
  await init();
  const session = new Session(Config.fromPreset("MorphIt-B"), Mesh.fromBytes(bytes, "obj"));
  const step: StepInfo | undefined = await session.stepMany({ budgetMs: 10 });
  const loss: number | undefined = step?.totalLoss;
  const report: InspectionReport = new RobotPackage().inspect();
  void loss; void report;
  return objectUrdf(session.centers(), session.radii(), { name: "x", anchored: true }).text;
}
`,
  );
  fs.writeFileSync(
    path.join(work, "tsconfig.json"),
    JSON.stringify({ compilerOptions: { strict: true, noEmit: true, module: "nodenext", moduleResolution: "nodenext", target: "es2022", lib: ["es2022", "dom", "esnext.disposable"], skipLibCheck: false }, files: ["check.ts"] }),
  );
  run(process.execPath, [path.join(work, "node_modules/typescript/bin/tsc"), "-p", work], work);
  console.log("TypeScript types: ok");
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
