# renegade-js

WebAssembly bindings for [Renegade](https://github.com/sanity/renegade), a
zero-configuration KNN-based supervised learner. The Rust dependency is pinned
to `renegade-ml` 0.6.0. Training and prediction run locally; no data is sent to
a server. This package is not the Renegade project's official JavaScript SDK.

## Usage

```js
import init, { Renegade } from "renegade-js";

// Required in browsers; harmless in Node.js.
await init();
const model = new Renegade([
  { type: "numeric", range: 500 },
  { type: "categorical" },
]);
model.add([40, "eu"], 0.8);
model.add([100, "us"], 0.4, 0.5); // optional instance weight
console.log(model.predict([50, "eu"]));
const neighbors = model.query([50, "eu"]);
console.log(neighbors.items, neighbors.classVotes(), neighbors.dispersion());
neighbors.free();
model.free();
```

In Node.js, the ESM entry initializes synchronously and calling `init()` is
optional. Browser ESM/bundlers use the web entry and must await `init()` before
constructing a model. Serve the `.wasm` file with your application; pass a URL,
bytes, or a compiled module to `init` if the default relative URL is unsuitable.
The package contains TypeScript declarations; no TypeScript runtime is needed.

Each point is an array matching the feature schema. Numeric values and outputs
must be finite numbers. A numeric range is a positive, caller-chosen scale:
the baseline distance is `min(abs(a-b)/range, 1)`. Categorical values are
strings or finite numbers; equal values have distance zero and different ones
distance one. Strings and numbers are distinct categories. Categories are
encoded in encounter order, including unseen query categories. Like the Rust
library, the metric learner uses numeric encodings for categorical features,
which can impose an artificial ordering when the learned metric is active.

`predict` uses Renegade's auto-selected K and kernel; `predictK` uses a fixed K
and inverse-distance weighting. `query` and `queryK` return a `Neighbors` object
with `items`, `weightedMean()`, `classVotes()`, `dispersion()`,
`gaussianWeightedMean(bandwidth)`, `gaussianDispersion(bandwidth)`, and
`extrapolate()`. `predictExtrapolated`, `predictWithPrior`, `getOptimalK`,
`forceRetrain`, `diagnostics`, `globalOutputMean` and `globalOutputVariance`
are also available. Diagnostics reflect the current training state and do not
force training. A query's dispersion describes that neighbor set, which may
differ from the set or kernel chosen by `predict`.

Calling prediction or query methods on an empty model, passing an invalid point,
weight, bandwidth or K throws an error. Empty global statistics return `null`.
Call `free()` on models and neighbor sets to release their WebAssembly memory.
The current wrapper does not expose custom Rust distance functions or `retain`.

## Build

Install Rust with the `wasm32-unknown-unknown` target, Node.js 20+ and
`wasm-pack`, then run `npm ci`, `npm run build`, `npm test`, and
`npm run test:types`. The type check uses the TypeScript 7 native preview.
`npm pack` builds the Node and browser artifacts before creating the
distributable package.

## License

LGPL-3.0-or-later, matching Renegade. See [LICENSE](LICENSE) and
[Renegade's license information](https://github.com/sanity/renegade#license).
The package includes the wrapper source and Cargo lockfile; the exact upstream
source is available from [crates.io](https://crates.io/crates/renegade-ml/0.6.0).
Use the build instructions above to rebuild the WebAssembly with modifications.
