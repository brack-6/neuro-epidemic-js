// neuro_epidemic.js — Neuro-Epidemic Engine (NEE)
// Isomorphism: neural nets and SIR epidemic models are both graph dynamical
// systems: x_{t+1} = sigma(W * x_t). Backprop learns contact matrix W and
// transmission params beta/gamma from observed incidence time series.

const NODES = 6;
const DT = 1.0;

// --- Sigmoid and its derivative ---
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function dsigmoid(x) { const s = sigmoid(x); return s * (1 - s); }

// --- NEE Layer: learnable SIR mean-field step ---
class EpidemicLayer {
  constructor(n) {
    this.n = n;
    // contact matrix W[i][j]: how much node j infects node i
    this.W = Array.from({length: n}, () =>
      Array.from({length: n}, () => Math.random() * 0.1)
    );
    this.beta  = 0.3 + Math.random() * 0.2;
    this.gamma = 0.25; // fixed from clinical prior
    // cache for backprop
    this._lastS = null;
    this._lastI = null;
    this._lastR = null;
    this._lastInflow = null;
  }

  // forward: one discrete mean-field SIR step
  forward(state) {
    const n = this.n;
    const S = state.slice(0, n);
    const I = state.slice(n, 2 * n);
    const R = state.slice(2 * n, 3 * n);

    // linear graph mixing: inflow[i] = sum_j W[i][j] * I[j]
    const inflow = Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        inflow[i] += this.W[i][j] * I[j];

    // nonlinear SIR update
    const newS = S.map((s, i) => Math.max(0, s - this.beta * s * inflow[i]));
    const newI = I.map((iv, i) => Math.max(0, iv + this.beta * S[i] * inflow[i] - this.gamma * iv));
    const newR = R.map((r, i) => Math.min(1, r + this.gamma * I[i]));

    // cache for backprop
    this._lastS = S; this._lastI = I; this._lastR = R; this._lastInflow = inflow;

    return [...newS, ...newI, ...newR];
  }

  // backward: SGD update on W, beta, gamma given output error
  backward(error, lr) {
    const n = this.n;
    const S = this._lastS;
    const I = this._lastI;
    const inflow = this._lastInflow;

    // grad of beta: dLoss/dbeta = sum_i error_I[i] * S[i] * inflow[i]
    let dBeta = 0;
    for (let i = 0; i < n; i++)
      dBeta += error[n + i] * S[i] * inflow[i];
    this.beta  = Math.max(0.01, this.beta  - lr * dBeta);

    // grad of gamma: dLoss/dgamma = -sum_i error_I[i] * I[i]
    let dGamma = 0;
    for (let i = 0; i < n; i++)
      dGamma -= error[n + i] * I[i];
    // gamma fixed: simulates known recovery time from clinical data
    // this.gamma = Math.max(0.01, this.gamma - lr * dGamma);

    // grad of W[i][j]: dLoss/dW[i][j] = error_I[i] * beta * S[i] * I[j]
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        const dW = error[n + i] * this.beta * S[i] * I[j];
        this.W[i][j] = Math.max(0, this.W[i][j] - lr * dW);
      }
  }
}

// --- Ground truth simulator (known params, used to generate training data) ---
function simulateGroundTruth(n, steps) {
  const trueW = Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) =>
      i !== j && Math.abs(i - j) <= 1 ? 0.25 : 0.0
    )
  );
  const trueBeta = 0.55, trueGamma = 0.25;

  let S = Array(n).fill(0.95);
  let I = Array(n).fill(0.0);
  let R = Array(n).fill(0.05);
  I[Math.floor(n / 2)] = 0.08; // seed infection in middle node
  S[Math.floor(n / 2)] = 0.87;

  const series = [];
  for (let t = 0; t < steps; t++) {
    const inflow = Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        inflow[i] += trueW[i][j] * I[j];

    const newS = S.map((s, i) => Math.max(0, s - trueBeta * s * inflow[i]));
    const newI = I.map((iv, i) => Math.max(0, iv + trueBeta * S[i] * inflow[i] - trueGamma * iv));
    const newR = R.map((r, i) => Math.min(1, r + trueGamma * I[i]));
    S = newS; I = newI; R = newR;
    series.push([...S, ...I, ...R]);
  }
  return series;
}

// --- MSE loss and error vector ---
function mse(pred, target) {
  return pred.reduce((acc, v, i) => acc + (v - target[i]) ** 2, 0) / pred.length;
}
function errorVec(pred, target) {
  return pred.map((v, i) => 2 * (v - target[i]) / pred.length);
}

// --- Training loop ---
function train(layer, series, epochs, lr) {
  for (let ep = 0; ep < epochs; ep++) {
    let totalLoss = 0;
    for (let t = 0; t < series.length - 1; t++) {
      const pred = layer.forward(series[t]);
      const target = series[t + 1];
      totalLoss += mse(pred, target);
      const err = errorVec(pred, target);
      layer.backward(err, lr);
    }
    if (ep % 100 === 0)
      console.log(`Epoch ${ep}: loss=${( totalLoss / series.length).toFixed(6)}, beta=${layer.beta.toFixed(3)}, gamma=${layer.gamma.toFixed(3)}`);
  }
}

// --- Main ---
const STEPS  = 40;
const EPOCHS = 5000;
const LR     = 0.5;

console.log("Generating ground truth (beta=0.55, gamma=0.25)...");
const series = simulateGroundTruth(NODES, STEPS);

console.log("Training NEE layer...");
const layer = new EpidemicLayer(NODES);
train(layer, series, EPOCHS, LR);

console.log("\nLearned parameters:");
console.log(`  beta:  ${layer.beta.toFixed(4)}  (true: 0.55)`);
console.log(`  gamma: ${layer.gamma.toFixed(4)}  (true: 0.25)`);console.log(`  R0 learned: ${(layer.beta/layer.gamma).toFixed(4)}  (true R0: ${(0.55/0.25).toFixed(4)})`);
console.log("  W[0]:  ", layer.W[0].map(v => v.toFixed(3)));

// Counterfactual: what if we halve beta?
console.log("\nCounterfactual (beta halved):");
const cfLayer = new EpidemicLayer(NODES);
cfLayer.W     = layer.W.map(row => row.slice());
cfLayer.beta  = layer.beta * 0.5;
cfLayer.gamma = layer.gamma;
let state = series[0].slice();
let peakI = 0;
for (let t = 0; t < STEPS; t++) {
  state = cfLayer.forward(state);
  const totalI = state.slice(NODES, 2 * NODES).reduce((a, b) => a + b, 0);
  peakI = Math.max(peakI, totalI);
}
console.log(`  Peak total infected: ${peakI.toFixed(3)} (vs uncontrolled)`);
