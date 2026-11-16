import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import initNode, { Renegade } from "../index.node.js";
import initWeb, { Renegade as WebRenegade } from "../index.web.js";

test("Node model supports mixed features, weighting, neighbors, and statistics", async () => {
  await initNode();
  const model = new Renegade([{ type: "numeric", range: 10 }, { type: "categorical" }]);
  assert.equal(model.globalOutputMean(), null);
  assert.throws(() => model.predict([0, "a"]), /no training data/);
  model.add([0, "a"], 2).add([0, "a"], 6, 3).add([10, "b"], 10);
  assert.equal(model.size, 3);
  assert.equal(model.globalOutputMean(), 6);
  assert.equal(model.predictK([0, "a"], 3), 5);
  const neighbors = model.queryK([0, "a"], 3);
  assert.deepEqual(neighbors.items, [
    { distance: 0, output: 2, weight: 1 },
    { distance: 0, output: 6, weight: 3 },
    { distance: 1, output: 10, weight: 1 },
  ]);
  assert.equal(neighbors.weightedMean(), 5);
  assert.equal(neighbors.dispersion().mean, 5);
  assert.equal(neighbors.gaussianDispersion(1).mean, 5);
  assert.equal(neighbors.classVotes().length, 3);
  assert.equal(neighbors.extrapolate().k, 3);
  assert.throws(() => neighbors.gaussianDispersion(0), /positive/);
  neighbors.free();
  assert.ok(Number.isFinite(model.predict([0, "a"])));
  assert.equal(model.predictExtrapolated([0, "a"]).k, model.getOptimalK());
  assert.equal(model.diagnostics().numEntries, 3);
  assert.equal(typeof model.predictWithPrior([0, "a"], 4), "number");
  model.forceRetrain();
  assert.equal(model.diagnostics().optimalK, null);
  model.free();
});

test("validates schema, data and query parameters", () => {
  assert.throws(() => new Renegade([]), /non-empty/);
  assert.throws(() => new Renegade([{ type: "numeric", range: 0 }]), /positive/);
  const model = new Renegade([{ type: "numeric", range: 1 }, { type: "categorical" }]);
  assert.throws(() => model.add([1], 0), /schema/);
  assert.throws(() => model.add([NaN, "a"], 0), /finite/);
  assert.throws(() => model.add([1, {}], 0), /categorical/);
  assert.throws(() => model.add([1, "a"], Infinity), /finite/);
  assert.throws(() => model.add([1, "a"], 0, -1), /positive/);
  assert.equal(model.size, 0);
  model.add([1, "a"], 3);
  assert.throws(() => model.queryK([1, "a"], 0), /positive/);
  assert.throws(() => model.queryK([1, "a"], 1.5), /integer/);
  assert.throws(() => model.predictWithPrior([1, "a"], NaN), /finite/);
  model.free();
});

test("training and incremental inserts remain queryable", () => {
  const model = new Renegade([{ type: "numeric", range: 20 }]);
  for (let i = 0; i < 20; i++) model.add([i], i * 2);
  assert.ok(Number.isFinite(model.predict([8.5])));
  assert.ok(model.diagnostics().optimalK > 0);
  model.add([21], 42);
  assert.equal(model.size, 21);
  assert.equal(model.predictK([21], 1), 42);
  model.free();
});

test("web entry initializes from bytes and performs predictions", async () => {
  const bytes = await readFile(new URL("../dist/web/renegade_bg.wasm", import.meta.url));
  await initWeb({ module_or_path: bytes });
  const model = new WebRenegade([{ type: "categorical" }]);
  model.add(["x"], 10).add(["y"], 20);
  assert.equal(model.predictK(["y"], 1), 20);
  const neighbors = model.queryK(["unseen"], 2);
  assert.deepEqual(neighbors.items.map((n) => n.distance), [1, 1]);
  neighbors.free();
  model.free();
});
