# Design: BLIS Autoscaler Algorithm Discovery Campaign

**Date**: 2026-06-01
**Status**: Draft — pending user approval
**Repo**: `agentic-strategy-evolution` (campaign config)
**Target repo**: `inference-sim` branch `feat/dpp-decider` @ `87e5a7ba`

---

## Problem Statement

The BLIS V2SaturationAnalyzer is a faithful port of the llm-d WVA V2 algorithm. It uses
token-based supply/demand signals with fixed thresholds to drive scale-up and scale-down.
The Phase 2 analyzer trace reveals a concrete cost inefficiency: at rate=22 req/s, the
analyzer scales the cluster to **3 replicas** (2 would have sufficed) and keeps all three
active for **240 s of idle stabilization** before scale-down fires. The 3-replica overshoot
arises because UnlimitedEngine dispatches `delta=+2` on the first actuation — it sees the
full RequiredCapacity gap and fills it in one shot, without accounting for the in-flight
loading delay that will naturally reduce demand.

This campaign directs nous to explore three algorithmic dimensions that could close this
cost gap without sacrificing SLO compliance:

1. **Predictive scaling** — forecast demand 3–5 minutes ahead using a per-tick arrival
   rate estimate, and size scale decisions to the *future* steady-state, not the current
   instantaneous overload snapshot.
2. **Multi-metric combination** — replace the token-only demand signal with a weighted
   combination of queue depth, KV utilization, and observed per-replica TTFT/latency,
   producing a richer signal that can detect over-provisioning sooner.
3. **Hysteresis curves** — make the effective scale-up and scale-down thresholds a
   function of recent scaling history (e.g., post-scale-up, raise the scale-down floor),
   reducing flapping cost without a separate per-direction stabilization window.

---

## Research Question

Within the discovery space of **(a) predictive demand forecasting**, **(b) multi-metric
combination formulas**, and **(c) hysteresis-curve thresholds**, which autoscaler design
achieves the minimum GPU cost (replica-hours) while keeping **p99 TTFT ≤ 500 ms** under
a step-workload `λ: 10 → 25 → 10 req/s` on BLIS (Llama-2-7b-hf, H100, TP=1)?

Sub-questions per dimension:

**(a) Predictive**: Does forecasting demand Δ=3–5 min ahead allow earlier, right-sized
scale-up (avoiding the 3-replica overshoot) and earlier scale-down (reducing idle time)?

**(b) Multi-metric**: Does combining queue depth + KV utilization + observed
`AvgInTokens`/`AvgOutTokens` (populated in 87e5a7ba) into a weighted demand signal reduce
overshoot vs. token-only signals?

**(c) Hysteresis**: Do state-dependent thresholds (post-scale-up, raise the scale-down
floor to prevent thrashing) cut flap cost without a separate stabilization window?

**Baseline to beat**: V2SaturationAnalyzer with `scale_up_threshold=0.85`,
`scale_down_boundary=0.70`, 240 s windows, `loading_delay` mean=240 s — which produces
the 3-replica overshoot and ~420 s of excess GPU-time observed in Phase 2.

---

## Anchors (pre-campaign, do NOT re-derive)

| Symbol | Value | Source |
|--------|-------|--------|
| Model | `meta-llama/Llama-2-7b-hf` | Phase 1 sweep |
| Hardware | H100, TP=1 | Phase 1 sweep |
| `(n, m)` | `(512, 128)` tokens | Phase 1/2 |
| `TotalKvCapacityTokens` | 110,176 tokens | Phase 2 debug run |
| `k1` (at `KvCacheThreshold=0.8`) | 88,141 tokens | Phase 2 trace |
| Single-replica saturation cliff | ~21–22 req/s | Phase 1 sweep |
| `λ_low` | 10 req/s (safe, 1 replica) | Phase 1 sweep |
| `λ_high` | 25 req/s (above cliff) | Phase 1 sweep |
| TTFT SLO `D` | 500 ms p99 | campaign-set |
| `cost_per_hour` | 3.00 $/hr (H100 representative) | campaign-set |
| Autoscaler tick | 30 s (`interval_us=30_000_000`) | WVA production parity |
| Loading delay | mean=240 s, stddev=30 s | Phase 2 plan |
| Simulation horizon | 1800 s (3 × 600 s phases) | campaign-set |
| Scale-up stabilization window | 120 s (reduced from 240 s) | campaign-set |
| Scale-down stabilization window | 240 s | WVA production parity |
| Latency model | `trained-physics` | Phase 2 plan |
| Branch | `feat/dpp-decider` @ `87e5a7ba` | user |

**Step workload**: implemented via two overlapping `ClientSpec` entries with
`LifecycleSpec.Windows`:
- Client A: always active, `rate_fraction=0.4` → λ_low = 10 req/s during both phases
- Client B: active only during high-load window `[600s, 1200s]`, `rate_fraction=0.6`
  → aggregate = 25 req/s during step, 10 req/s otherwise

---

## Signal Availability

The three candidate algorithms need different per-replica signals. Status in 87e5a7ba:

| Signal | Status | Used by |
|--------|--------|---------|
| `KvTokensInUse`, `QueueDepth` | ✅ populated | baseline V2, all candidates |
| `AvgInTokens`, `AvgOutTokens` | ✅ populated (new in 87e5a7ba) | multi-metric, k2 |
| `MaxBatchSize` | ✅ populated | k2 compute-bound |
| `TTFT`, `DispatchRate`, `ITL` | ❌ zero (removed in #1382) | multi-metric latency signals |
| Per-tick arrival rate λ̂ | ❌ not in `RouterState` | predictive |
| `mean_active_replicas`, cost | ❌ not in metrics output | cost measurement |

**Code that nous may need to add**:

1. **Cost metric** — `mean_active_replicas` (time-weighted integral of active replica count
   / horizon) and `total_gpu_cost_usd` (= mean_active_replicas × cost_per_hour × horizon_s
   / 3600) in the metrics output. Add to `sim/metrics.go` and `sim/cluster/cluster.go`.

2. **Arrival rate tracking** — count requests arriving per autoscaler tick window in
   `cluster.go`; expose as `ArrivalRatePerSec float64` in `ModelSignals`. Needed for
   predictive scaling.

3. **Per-replica TTFT tracker** — rolling average TTFT on `InstanceSimulator` using the
   same O(1) accessor pattern as `AvgInputTokens()`/`AvgOutputTokens()` from 87e5a7ba.
   Needed for multi-metric latency signals.

4. **Analyzer selection from YAML** — add `analyzer_type` field to the autoscaler YAML
   config (`autoscaler.analyzer.type: v2-saturation | predictive | multi-metric |
   hysteresis`). Add a factory switch in `cluster.go` (or `bundle.go`) to instantiate the
   right implementation.

---

## Architecture

### Autoscaler pipeline (existing)

```
ScalingTickEvent
  → DefaultCollector.Collect(*RouterState) → []ModelSignals
  → for each model: Analyzer.Analyze(ModelSignals) → AnalyzerResult
  → Engine.Optimize([]AnalyzerResult, GPUInventory) → []ScaleDecision
  → schedule ScaleActuationEvent after HPAScrapeDelay
  → DirectActuator.Apply(ScaleDecision)
```

### New analyzer files (one per candidate)

```
sim/cluster/
  saturation_analyzer.go    ← existing V2SaturationAnalyzer (baseline)
  predictive_analyzer.go    ← (new) PredictiveAnalyzer
  multi_metric_analyzer.go  ← (new) MultiMetricAnalyzer
  hysteresis_analyzer.go    ← (new) HysteresisAnalyzer
```

Each implements:
```go
type Analyzer interface {
    Analyze(metrics ModelSignals) AnalyzerResult
    Name() string
}
```

New analyzers are wired via the `analyzer_type` config field. All extend the existing
`V2SaturationAnalyzerConfig` or use their own config struct registered in the YAML bundle.

### Predictive analyzer sketch

Uses `ModelSignals.ArrivalRatePerSec` (to be added) and a configurable forecast horizon
`PredictionHorizonUs`. At each tick, computes expected demand in `PredictionHorizonUs`
microseconds assuming the current arrival rate and mean service time, then scales to
satisfy the *forecasted* demand rather than the current instantaneous demand.

### Multi-metric analyzer sketch

Replaces the scalar demand estimate with a weighted score:
```
demand_score = w_kv * KvTokensInUse
             + w_q  * QueueDepth * AvgInTokens
             + w_ttft * (observed_ttft / slo_ttft_target)
```
Weights `(w_kv, w_q, w_ttft)` are config parameters. The scale-up signal fires when
`demand_score / supply > ScaleUpThreshold`.

### Hysteresis analyzer sketch

Extends V2SaturationAnalyzer with a state machine tracking the last scale direction.
After a scale-up event, the effective `scale_down_boundary` is raised by `HysteresisGap`
for `HysteresisWindowUs` microseconds, preventing immediate scale-down of the just-added
replica. Converges to the static threshold once the window expires.

---

## Observable Metrics

```
mean_ttft_ms               # primary SLO metric
p99_ttft_ms                # SLO target: ≤ 500 ms
ttft_slo_violation_rate    # fraction of requests with TTFT > D
mean_itl_ms
p99_itl_ms
throughput_rps
mean_active_replicas       # cost proxy; to be added
total_gpu_cost_usd         # = mean_active_replicas × cost_per_hour × horizon_s / 3600; to be added
scale_up_count             # from log: grep "[actuator] scale-up"
scale_down_count           # from log: grep "[actuator] scale-down"
flap_count                 # scale-up followed by scale-down within 2 × scale_up_stabilization_window_us
prefill_server_utilization
decode_server_utilization
```

---

## Controllable Knobs

```
# Algorithm selection
analyzer_type              # v2-saturation | predictive | multi-metric | hysteresis

# V2Saturation baseline knobs
scale_up_threshold         # utilization fraction triggering scale-up
scale_down_boundary        # utilization fraction below which scale-down is safe
kv_cache_threshold         # fraction of KV capacity counted as effective supply

# Predictive knobs
prediction_horizon_us      # forecast window: 180_000_000–300_000_000 (3–5 min)
arrival_ema_alpha          # EMA decay for arrival rate estimate (0.1–0.5)

# Multi-metric knobs
weight_kv                  # weight on KV token demand
weight_queue               # weight on queue token demand
weight_ttft                # weight on TTFT/SLO ratio signal
ttft_slo_target_us         # SLO target fed into multi-metric score

# Hysteresis knobs
hysteresis_gap             # delta added to scale_down_boundary post-scale-up
hysteresis_window_us       # duration of post-scale-up threshold elevation

# Stabilization windows
scale_up_stabilization_window_us
scale_down_stabilization_window_us

# Workload
aggregate_rate             # total λ; split by lifecycle windows
lambda_low                 # 10 req/s
lambda_high                # 25 req/s
step_start_us              # 600_000_000 (600 s)
step_end_us                # 1_200_000_000 (1200 s)
arrival_process            # poisson | gamma (for iter-4 bursty test)
arrival_cv                 # CV > 1 for iter-4 adversarial test
```

---

## Iteration Plan

### Iter-1 — Cost metric + baseline characterization

Add `mean_active_replicas` and `total_gpu_cost_usd` to BLIS output. Add `ArrivalRatePerSec`
to `ModelSignals`. Run V2SaturationAnalyzer under the step workload with the Phase 2 config
(scale_up=0.85, scale_down=0.70, windows=240s). Quantify the 3-replica overshoot cost
precisely: how many GPU-hours are wasted? What is the p99 TTFT profile across phases?

**Deliverable**: Baseline cost-SLO point. Characterization of the overshoot pattern (how
many excess replica-seconds, how long until scale-down fires).

### Iter-2 — Best-promising algorithm prototype

Based on iter-1 findings, nous selects the most promising dimension:
- If overshoot dominates (replicas = 3 when 2 suffice) → try **predictive** (right-size
  the scale decision using forecasted demand)
- If idle time after load drop dominates → try **hysteresis** (raise scale-down floor
  post-scale-up, enabling faster scale-down)
- If demand signal noise causes jitter → try **multi-metric** (add TTFT signal to dampen
  premature scale-up under transient spikes)

Implement the chosen analyzer in a new file. Compare cost-SLO vs. baseline.

**Deliverable**: First candidate algorithm. Cost improvement or explanation of why the
dimension didn't help.

### Iter-3 — Refine + explore second dimension

Tune the iter-2 algorithm (prediction horizon, metric weights, or hysteresis curve shape).
If iter-2 is conclusively better, explore a second dimension from the discovery space.
Produce a Pareto curve (cost vs. p99_ttft) for each candidate.

**Deliverable**: Refined candidate(s). Pareto curves showing cost-SLO trade-offs.

### Iter-4 — Adversarial validation

Test the best iter-3 algorithm under a bursty workload (`arrival.process: gamma`, CV=1.5,
near-saturation rate=20 req/s). Validates that the discovered algorithm does not regress
on stability (flap count, SLO violation rate) relative to V2Saturation under non-Poisson
arrivals.

**Deliverable**: Stability comparison. Recommendation of the best algorithm configuration.

---

## Campaign YAML Location

```
examples/blis-autoscaling-campaign.yaml
```

## Success Criteria

1. At least one candidate algorithm beats the V2Saturation baseline on total GPU cost by
   ≥10% at matched p99 TTFT ≤ 500 ms.
2. The winning algorithm does not increase `flap_count` relative to the baseline.
3. Under the adversarial iter-4 workload, p99 TTFT SLO violation rate is ≤ the baseline
   violation rate (no stability regression).
4. The winning configuration is expressible as a BLIS YAML policy bundle that can be
   shipped as an example.
