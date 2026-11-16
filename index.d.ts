export type Feature = { type: "numeric"; range: number } | { type: "categorical" };
export type Point = readonly (number | string)[];
export interface Neighbor { distance: number; output: number; weight: number }
export interface Dispersion {
  mean: number;
  variance: number;
  stdDev: number;
  effectiveN: number;
  weightSum: number;
  standardError: number;
}
export interface Extrapolation { value: number; rSquared: number; k: number }
export interface FeatureDiagnostics {
  index: number;
  weight: number;
  effectCurve: [number, number][];
}
export interface Diagnostics {
  numEntries: number;
  optimalK: number | null;
  metricActive: boolean;
  kernelBandwidth: number | null;
  trainedAt: number;
  entriesSinceTraining: number;
  isClassification: boolean;
  featureMetrics: FeatureDiagnostics[] | null;
  outputStats: { min: number; max: number; mean: number; numDistinct: number };
}
export class Neighbors {
  private constructor();
  free(): void;
  readonly items: Neighbor[];
  weightedMean(): number;
  gaussianWeightedMean(bandwidth: number): number;
  dispersion(): Dispersion | null;
  gaussianDispersion(bandwidth: number): Dispersion | null;
  classVotes(): [number, number][];
  extrapolate(): Extrapolation;
}
export class Renegade {
  constructor(features: readonly Feature[]);
  free(): void;
  readonly size: number;
  add(point: Point, output: number, weight?: number): this;
  forceRetrain(): void;
  getOptimalK(): number;
  predict(point: Point): number;
  predictK(point: Point, k: number): number;
  predictWithPrior(point: Point, prior: number): number;
  predictExtrapolated(point: Point): Extrapolation;
  query(point: Point): Neighbors;
  queryK(point: Point, k: number): Neighbors;
  globalOutputMean(): number | null;
  globalOutputVariance(): number | null;
  diagnostics(): Diagnostics;
}
/** Initialize WebAssembly in browsers; a no-op in Node.js. */
export default function init(input?: unknown): Promise<unknown>;
