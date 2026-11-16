function finite(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number`);
  }
}

function positive(value, name) {
  finite(value, name);
  if (value <= 0) throw new RangeError(`${name} must be positive`);
}

function count(value) {
  if (!Number.isInteger(value) || value <= 0 || value > 0xffffffff) {
    throw new RangeError("k must be a positive 32-bit integer");
  }
}

export function createAPI(wasm) {
  class Neighbors {
    #inner;
    constructor(inner) { this.#inner = inner; }
    free() { this.#inner.free(); }
    get items() { return this.#inner.items(); }
    weightedMean() { return this.#inner.weighted_mean(); }
    gaussianWeightedMean(bandwidth) {
      positive(bandwidth, "bandwidth");
      return this.#inner.gaussian_weighted_mean(bandwidth);
    }
    dispersion() { return this.#inner.dispersion(); }
    gaussianDispersion(bandwidth) {
      positive(bandwidth, "bandwidth");
      return this.#inner.gaussian_dispersion(bandwidth);
    }
    classVotes() { return this.#inner.class_votes(); }
    extrapolate() { return this.#inner.extrapolate(); }
  }

  class Renegade {
    #inner;
    #features;
    #categories;

    constructor(features) {
      if (!Array.isArray(features) || features.length === 0) {
        throw new TypeError("features must be a non-empty array");
      }
      this.#features = features.map((feature) => {
        if (feature?.type === "categorical") return { type: "categorical" };
        if (feature?.type === "numeric") {
          positive(feature.range, "numeric feature range");
          return { type: "numeric", range: feature.range };
        }
        throw new TypeError("each feature must be numeric or categorical");
      });
      this.#categories = this.#features.map(() => new Map());
      this.#inner = new wasm.Model(Float64Array.from(this.#features, (f) => f.range ?? 0));
    }

    #point(values) {
      if (!Array.isArray(values) || values.length !== this.#features.length) {
        throw new TypeError("point must be an array matching the feature schema");
      }
      for (let i = 0; i < values.length; i++) {
        const value = values[i];
        if (this.#features[i].type === "numeric") finite(value, `feature ${i}`);
        else if (typeof value !== "string" && (typeof value !== "number" || !Number.isFinite(value))) {
          throw new TypeError(`categorical feature ${i} must be a string or finite number`);
        }
      }
      return Float64Array.from(values, (value, i) => {
        if (this.#features[i].type === "numeric") return value;
        const categories = this.#categories[i];
        if (!categories.has(value)) categories.set(value, categories.size);
        return categories.get(value);
      });
    }

    #requireData() {
      if (this.#inner.size() === 0) throw new Error("model has no training data");
    }

    free() { this.#inner.free(); }
    get size() { return this.#inner.size(); }
    add(point, output, weight = 1) {
      finite(output, "output");
      positive(weight, "weight");
      this.#inner.add(this.#point(point), output, weight);
      return this;
    }
    forceRetrain() { this.#inner.force_retrain(); }
    getOptimalK() { this.#requireData(); return this.#inner.optimal_k(); }
    predict(point) { this.#requireData(); return this.#inner.predict(this.#point(point)); }
    predictK(point, k) {
      this.#requireData(); count(k);
      return this.#inner.predict_k(this.#point(point), k);
    }
    predictWithPrior(point, prior) {
      this.#requireData(); finite(prior, "prior");
      return this.#inner.predict_with_prior(this.#point(point), prior);
    }
    predictExtrapolated(point) {
      this.#requireData();
      return this.#inner.predict_extrapolated(this.#point(point));
    }
    query(point) {
      this.#requireData();
      return new Neighbors(this.#inner.query(this.#point(point)));
    }
    queryK(point, k) {
      this.#requireData(); count(k);
      return new Neighbors(this.#inner.query_k(this.#point(point), k));
    }
    globalOutputMean() { return this.#inner.global_output_mean() ?? null; }
    globalOutputVariance() { return this.#inner.global_output_variance() ?? null; }
    diagnostics() { return this.#inner.diagnostics(); }
  }
  return { Renegade, Neighbors };
}
