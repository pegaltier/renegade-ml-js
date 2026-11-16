use renegade_ml::{DataPoint, Renegade};
use serde::Serialize;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[derive(Clone)]
struct Point {
    values: Vec<f64>,
    ranges: Arc<Vec<f64>>,
}

impl DataPoint for Point {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        self.values
            .iter()
            .zip(&other.values)
            .zip(self.ranges.iter())
            .map(|((&a, &b), &range)| {
                if range == 0.0 {
                    if a == b {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    ((a - b).abs() / range).min(1.0)
                }
            })
            .collect()
    }

    fn feature_values(&self) -> Vec<f64> {
        self.values.clone()
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Neighbor {
    distance: f64,
    output: f64,
    weight: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Dispersion {
    mean: f64,
    variance: f64,
    std_dev: f64,
    effective_n: f64,
    weight_sum: f64,
    standard_error: f64,
}

impl From<renegade_ml::Dispersion> for Dispersion {
    fn from(d: renegade_ml::Dispersion) -> Self {
        Self {
            mean: d.mean,
            variance: d.variance,
            std_dev: d.std_dev,
            effective_n: d.effective_n,
            weight_sum: d.weight_sum,
            standard_error: d.standard_error(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Extrapolation {
    value: f64,
    r_squared: f64,
    k: usize,
}

impl From<renegade_ml::ExtrapolatedPrediction> for Extrapolation {
    fn from(p: renegade_ml::ExtrapolatedPrediction) -> Self {
        Self {
            value: p.value,
            r_squared: p.r_squared,
            k: p.k,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FeatureDiagnostics {
    index: usize,
    weight: f64,
    effect_curve: Vec<(f64, f64)>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputStats {
    min: f64,
    max: f64,
    mean: f64,
    num_distinct: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Diagnostics {
    num_entries: usize,
    optimal_k: Option<usize>,
    metric_active: bool,
    kernel_bandwidth: Option<f64>,
    trained_at: usize,
    entries_since_training: usize,
    is_classification: bool,
    feature_metrics: Option<Vec<FeatureDiagnostics>>,
    output_stats: OutputStats,
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub struct Model {
    inner: Renegade<Point>,
    ranges: Arc<Vec<f64>>,
}

impl Model {
    fn point(&self, values: Vec<f64>) -> Point {
        Point {
            values,
            ranges: self.ranges.clone(),
        }
    }
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(ranges: Vec<f64>) -> Self {
        Self {
            inner: Renegade::new(),
            ranges: Arc::new(ranges),
        }
    }

    pub fn add(&mut self, values: Vec<f64>, output: f64, weight: f64) {
        self.inner.add_weighted(self.point(values), output, weight);
    }

    pub fn size(&self) -> usize {
        self.inner.len()
    }

    pub fn force_retrain(&mut self) {
        self.inner.force_retrain();
    }

    pub fn optimal_k(&mut self) -> usize {
        self.inner.get_optimal_k()
    }

    pub fn predict(&mut self, values: Vec<f64>) -> f64 {
        self.inner.predict(&self.point(values))
    }

    pub fn predict_k(&self, values: Vec<f64>, k: usize) -> f64 {
        self.inner.predict_k(&self.point(values), k)
    }

    pub fn predict_with_prior(&mut self, values: Vec<f64>, prior: f64) -> f64 {
        self.inner.predict_with_prior(&self.point(values), prior)
    }

    pub fn predict_extrapolated(&mut self, values: Vec<f64>) -> Result<JsValue, JsValue> {
        to_js(&Extrapolation::from(
            self.inner.predict_extrapolated(&self.point(values)),
        ))
    }

    pub fn query(&mut self, values: Vec<f64>) -> NeighborSet {
        NeighborSet {
            inner: self.inner.query(&self.point(values)),
        }
    }

    pub fn query_k(&self, values: Vec<f64>, k: usize) -> NeighborSet {
        NeighborSet {
            inner: self.inner.query_k(&self.point(values), k),
        }
    }

    pub fn global_output_mean(&self) -> Option<f64> {
        self.inner.global_output_mean()
    }

    pub fn global_output_variance(&self) -> Option<f64> {
        self.inner.global_output_variance()
    }

    pub fn diagnostics(&self) -> Result<JsValue, JsValue> {
        let d = self.inner.diagnostics();
        to_js(&Diagnostics {
            num_entries: d.num_entries,
            optimal_k: d.optimal_k,
            metric_active: d.metric_active,
            kernel_bandwidth: d.kernel_bandwidth,
            trained_at: d.trained_at,
            entries_since_training: d.entries_since_training,
            is_classification: d.is_classification,
            feature_metrics: d.feature_metrics.map(|features| {
                features
                    .into_iter()
                    .map(|f| FeatureDiagnostics {
                        index: f.index,
                        weight: f.weight,
                        effect_curve: f.effect_curve,
                    })
                    .collect()
            }),
            output_stats: OutputStats {
                min: d.output_stats.min,
                max: d.output_stats.max,
                mean: d.output_stats.mean,
                num_distinct: d.output_stats.num_distinct,
            },
        })
    }
}

#[wasm_bindgen]
pub struct NeighborSet {
    inner: renegade_ml::Neighbors,
}

#[wasm_bindgen]
impl NeighborSet {
    pub fn items(&self) -> Result<JsValue, JsValue> {
        to_js(
            &self
                .inner
                .neighbors
                .iter()
                .map(|n| Neighbor {
                    distance: n.distance,
                    output: n.output,
                    weight: n.weight,
                })
                .collect::<Vec<_>>(),
        )
    }

    pub fn weighted_mean(&self) -> f64 {
        self.inner.weighted_mean()
    }

    pub fn gaussian_weighted_mean(&self, bandwidth: f64) -> f64 {
        self.inner.gaussian_weighted_mean(bandwidth)
    }

    pub fn dispersion(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.dispersion().map(Dispersion::from))
    }

    pub fn gaussian_dispersion(&self, bandwidth: f64) -> Result<JsValue, JsValue> {
        to_js(
            &self
                .inner
                .gaussian_dispersion(bandwidth)
                .map(Dispersion::from),
        )
    }

    pub fn class_votes(&self) -> Result<JsValue, JsValue> {
        to_js(&self.inner.class_votes())
    }

    pub fn extrapolate(&self) -> Result<JsValue, JsValue> {
        to_js(&Extrapolation::from(self.inner.extrapolate()))
    }
}
