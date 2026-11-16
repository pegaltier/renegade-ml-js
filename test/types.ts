import init, { Renegade, type Dispersion, type Feature } from "renegade-ml-js";

await init();
const features: Feature[] = [{ type: "numeric", range: 10 }, { type: "categorical" }];
const model = new Renegade(features);
model.add([2, "region"], 0.5);
const prediction: number = model.predict([3, "region"]);
const neighbors = model.query([3, "region"]);
const dispersion: Dispersion | null = neighbors.dispersion();
const votes: [number, number][] = neighbors.classVotes();
// @ts-expect-error numeric features require a range
new Renegade([{ type: "numeric" }]);
// @ts-expect-error output must be numeric
model.add([1, "region"], "class");
neighbors.free();
model.free();
void [prediction, dispersion, votes];
