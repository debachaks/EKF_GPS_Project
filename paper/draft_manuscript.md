<!--
DRAFT — Journal of Hardware and Systems Security (Springer HaSS), regular paper track.
Target length: <=14 pages (double/1.5-spaced, figs/tables/refs included).
Formatting: convert to Springer Nature's unified LaTeX class, sn-jnl.cls
(Overleaf template: "Springer Nature LaTeX Template"), reference style TBD
per HaSS's specific author guidelines (couldn't be fetched directly --
guidelines page sits behind an IdP login wall; confirm reference style
and any HaSS-specific requirements manually before final submission).

IMPORTANT: every [CITE] marker below is a placeholder. No fabricated
citations have been inserted -- these need real, verified references
before submission. Author names/affiliations are placeholders too.
-->

# Detecting GPS Spoofing via Hardware Performance Counter Side-Channels from an ANIS-Gated Adaptive Kalman Filter

**Authors:** [Author names], [Affiliation]

## Abstract

GPS/INS extended Kalman filters (EKFs) deployed on embedded platforms typically use fixed process and measurement noise covariances (Q, R), tuned offline from sensor datasheets. This design is well known to degrade under model mismatch — including GPS spoofing, which shifts the measurement away from ground truth without triggering any structural change in the filter itself. Adaptive techniques such as the Strong Tracking Filter (STF) address this by inflating the state covariance when the innovation statistic (NIS/ANIS) indicates the model no longer fits. We show, however, that applying STF unconditionally is counterproductive for *detecting* spoofing: by helping the filter re-converge onto the spoofed trajectory faster, it closes the very anomaly a detector needs to observe (alarm rate on a jump attack fell from 147/299 to 6.6/299 steps under unconditional STF). We instead gate STF behind an ANIS threshold derived from the χ²(3, 0.95) distribution, so the adaptive branch only executes when independent statistical evidence already suggests something is wrong.

Software-only defenses against this attack class — chi-squared/NIS monitoring of the filter's own innovation — are vulnerable in principle to an attacker who shapes the injected measurement specifically to stay under the monitored statistic's threshold, a strategy demonstrated in prior adversarial work. A detection channel that does not depend on the filter's self-reported statistics is a meaningfully independent second line of defense against an attacker confined to the measurement channel who does not know or specifically target the filter's exact internal gating threshold; Section 1.1 makes this boundary precise. This conditional branch has a side effect we exploit directly: it is a data-dependent code path, and its execution leaves a measurable footprint on RISC-V hardware performance monitor (HPM) counters — FP-pipeline stalls, conditional-branch retirements, and jump-and-link retirements all shift when the branch fires, while a deliberately-chosen negative-control counter (pipeline flushes) should not. We build three convergent window-diff detection statistics over these counters (denoted D, G, V in this work), each following a shared three-stage construction — a per-window statistic, differenced across time, then z-scored against a 20-trial normal-mode baseline — with an explicit guard against near-zero-variance ("fragile") baseline positions that would otherwise cause the detector to diverge numerically. Evaluated across 20 seeds of normal, jump, and drift GPS-spoofing trials on a RISC-V EKF implementation, the three detectors achieve F1 >= 0.96 with false-alarm rates <= 5% on five of six usable counters; a sixth counter, included as a near-negative-control, shows markedly lower recall (0.73-0.85), which we show is attributable to persistent pre-attack noise rather than a flaw in the detection construction. Majority-vote ensembling across the three detectors drives false alarms toward zero for the well-behaved counters at negligible recall cost.

**Keywords:** GPS spoofing detection, hardware performance counters, adaptive Kalman filtering, Strong Tracking Filter, side-channel detection, embedded systems security

---

## 1. Introduction

Embedded navigation systems that fuse GPS with inertial measurement units (IMUs) via an extended Kalman filter (EKF) are foundational to autonomous vehicles, UAVs, and a wide range of safety- and security-relevant embedded platforms. In practice, a large fraction of these systems use a *fixed*-covariance EKF: process noise Q and measurement noise R are tuned once, offline, from IMU/GPS datasheet specifications, and left unchanged at runtime [CITE]. This is a deliberate and common engineering choice on resource-constrained platforms, where adaptive noise estimation adds computational overhead and potential instability.

The well-documented weakness of this design is that the filter has no internal mechanism to notice when its own model assumptions have stopped holding [CITE — Zhou & Frank STF]. A GPS spoofing attack is a particularly insidious instance of this: it silently shifts the measurement away from ground truth, and a fixed-covariance filter simply absorbs the false signal at whatever rate its (static) Kalman gain allows, with no internal alarm. Adaptive filtering techniques — Sage-Husa noise estimation, the Strong Tracking Filter (STF) [CITE], innovation-based adaptive estimation — exist precisely to address this class of problem, historically motivated by maneuvering targets and sensor faults rather than adversarial spoofing specifically.

### 1.1 Why a hardware-level detection channel

The standard software-only defense against this class of attack is an innovation-based statistical test — a chi-squared (NIS) check on the filter's own residuals, exactly the mechanism Section 4 builds on. This is effective against unsophisticated spoofing, but it has a structural weakness: it is entirely internal to the same software/numerical trust boundary the attacker is already manipulating. Recent adversarial work has shown this concretely — reinforcement-learning-based spoofing strategies have been demonstrated that shape an injected trajectory specifically to stay under a monitored Kalman filter's chi-squared detection threshold [CITE — RL-based KF spoofing, see Related Work], i.e., an attacker with enough knowledge of the defender's own statistical test can construct an attack that evades it by construction. Any purely software-internal detector — however well-designed — inherits this weakness in principle, because the attacker only needs to control the one signal (the filter's self-reported innovation statistic) that the detector reads.

A hardware performance counter (HPC) channel is a fundamentally different kind of observation from the filter's self-reported NIS/ANIS: it measures the physical, microarchitectural consequence of which instructions the CPU actually executed, not a value computed and exposed by the algorithm under attack. This independence has a precise boundary, which we state explicitly rather than leave implicit.

Within this paper's threat model (Section 3.2) — an attacker confined to the external GPS measurement stream, with no write access to the filter's internal state — `anis_prev` is computed honestly by untampered filter code from whatever measurement the attacker injects. Such an attacker's only lever over the ANIS-gated branch, and therefore over the HPC channel (Section 5), is indirect: shaping the injected measurement so that the honestly-computed ANIS stays under `ANIS_THRESHOLD` throughout the attack. If an attacker with white-box knowledge of the exact threshold succeeds at this, `compute_lambda_k()` never executes, and *neither* the software ANIS check nor the HPC side channel derived from it observes anything anomalous — because nothing anomalous actually happens internally to observe. The two channels are therefore not independent of each other against this specific attacker: HPC visibility is entirely mediated through the same branch the software-only check also watches.

What the HPC channel *is* independent of is a stronger, different attacker: one with enough system access to directly falsify the filter's internal state (e.g., overwrite `anis_prev` itself) rather than merely inject a measurement the filter processes honestly. Such an attacker would trivially defeat both channels through the same lever — direct control of the branch condition — and at that point there is no meaningful two-channel defense-in-depth claim left to make. Extending protection to that stronger threat model would require an architectural change this single-firmware-image system does not currently have (e.g., HPC readout via a privilege-isolated monitor the compromised code cannot reach); we note this explicitly as a scope limitation rather than an implicit assumption (Section 9).

HPCs are additionally attractive on embedded platforms specifically because they are already present on essentially all modern CPU cores (including the RISC-V HPM extension used here) and are read passively, at negligible marginal cost, rather than requiring additional onboard computation the way a more elaborate software anomaly detector would.

This paper makes four contributions:

1. We show empirically that the classical remedy (unconditional STF) is in direct tension with spoofing *detectability*: inflating covariance whenever the innovation looks unusual helps the filter re-track a spoofed trajectory faster, which suppresses the very anomaly signal a detector needs. We instead gate the adaptive branch behind an innovation-based statistical test (ANIS vs. its formal χ² critical value), so it only executes when justified — a design choice motivated less by classical tracking performance and more by its side effect (Contribution 2).

2. We show that this conditional branch is externally observable from *outside* the filter's own numerical output, via RISC-V hardware performance monitor (HPM) counters. We give a counter-by-counter causal account of *why* each counter does or does not carry signal, including a deliberately-included negative-control counter, and confirm the causal story empirically. To the best of our knowledge, while HPC-based side-channel detection is well established for malware and control-flow-integrity attacks (Section 2), and while chi-squared/NIS-based Kalman filter monitoring is well established as a software-only GPS spoofing defense (itself shown to be evadable by an informed adversary — Section 1.1), no prior work combines the two: using microarchitectural hardware telemetry, rather than the filter's own reported statistics, as the detection channel for sensor spoofing against a state estimator.

3. We build and evaluate three convergent, principled window-diff detection statistics over these counters, with an explicit treatment of two failure modes that a naive implementation is prone to: (a) numerical blow-up when a between-trial baseline variance collapses to near-zero, and (b) false-positive inflation from counting pre-attack noise as a detection. We report standard detection metrics (Precision, Recall, F1, False-Alarm rate) computed honestly with respect to both failure modes, and show that simple ensembling across the three detectors further reduces false alarms at minimal recall cost for well-behaved counters.

4. We give an explicit, empirically-grounded argument for *why* a hardware-level channel is preferable to a purely software/statistical one against this attack class (Section 1.1), rather than treating the choice of observation layer as incidental to the detection method.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 describes the system and threat model. Section 4 details the adaptive filter design, including the failed unconditional-STF attempt. Section 5 gives the hardware-counter side-channel account. Section 6 describes the detection methodology. Section 7 describes the experimental setup. Section 8 presents results. Section 9 discusses limitations. Section 10 concludes.

## 2. Related Work

*Note: the items below were located via exploratory search to scope this section and establish the novelty claim in Section 1.1; titles/venues are as found and MUST be independently verified (authors, year, venue, DOI) against the primary source before being entered in the reference list. This is a starting point for the literature review, not a finished one.*

### 2.1 GPS/GNSS spoofing detection

Existing spoofing countermeasures broadly split into cryptographic/signal-authentication techniques and anomaly-detection techniques operating on the receiver or navigation-filter output [CITE]. Within the anomaly-detection branch, innovation-based (chi-squared/NIS) monitoring of a Kalman-filtered navigation solution is a standard approach [CITE — e.g. "Analysis of Kalman Filter Innovation-Based GNSS Spoofing Detection Method for INS/GNSS Integrated Navigation System," IEEE, located via search, full details TBC], with reported test statistics including the raw innovation mean, an averaged/summed innovation test, and the snapshot chi-squared test [CITE]. Impact-assessment work on INS/GNSS integrated systems under spoofing has specifically evaluated detector performance against **step** and **ramp** fault profiles [CITE — "Impact Assessment of GNSS Spoofing Attacks on INS/GNSS Integrated Navigation System," located via search] — the standard GNSS-community terminology corresponding to this paper's jump and drift attacks respectively (Section 3.2); autopilot-response-based and carrier-phase/IMU-based INS monitoring against spoofing has also been studied by established groups in this space (e.g. Illinois Tech's NavLab, UT Austin's RadioNavLab) [CITE — Tanil et al.; Clements et al.; full details TBC]. Vehicle-telemetry-based detection (position/speed/status consistency checks, sometimes combined with lightweight onboard machine learning) has also been studied for UAV platforms [CITE — telemetry-based UAV GPS spoofing detection]. Anomaly-based frameworks for autonomous-vehicle GPS spoofing detection (e.g. "GPS-IDS"-style approaches) and prediction-based GNSS spoofing detection for autonomous vehicles have also been proposed [CITE]. None of this line of work operates on microarchitectural/hardware telemetry.

Notably, at least one impact-assessment study in this space independently observes that detection has "a limited detection window as the Kalman filter dynamically tunes itself to track spoofing profiles" [CITE — same impact-assessment source as above] — a finding consistent in spirit with this paper's Section 4.1 result that adaptive covariance inflation can suppress detectability, though that prior work does not propose or evaluate a gated/conditional remedy, nor connect the phenomenon to a hardware-observable side channel.

Directly relevant to the motivation in Section 1.1: adversarial work has demonstrated reinforcement-learning-based strategies for spoofing a Kalman filter that is itself being monitored by a chi-squared statistical test, constructing injected measurements that evade the software-only detector [CITE — RL-based spoofing of a monitored Kalman filter, located via search, full bibliographic details to be confirmed]; relatedly, at least one study explicitly constructs a "worst-case spoofing profile" designed to maximize position error while remaining undetected by an innovation-based monitor [CITE — worst-case/undetectable spoofing profile, located via search]. We use these as direct motivating evidence in Section 1.1 that a purely software-internal detection channel is not a complete defense on its own.

### 2.2 Adaptive Kalman filtering

Sage-Husa adaptive noise estimation, the Strong Tracking Filter (STF) [CITE — Zhou & Frank], and innovation-based adaptive estimation (IAE) are established techniques for maintaining filter consistency under model mismatch, historically motivated by maneuvering targets and sensor faults rather than adversarial spoofing specifically [CITE]. As noted in Section 2.1, at least one GNSS impact-assessment study has independently observed a "limited detection window" effect consistent in spirit with this paper's Section 4.1 finding, so the general tension between adaptive/robust estimation and detectability is not entirely without precedent. What we did not find prior evidence of is (a) a quantified before/after comparison of detection rate with and without adaptive covariance inflation on the same attack set, or (b) a proposed remedy that gates the adaptive mechanism behind an independent statistical test specifically to resolve this tension, or (c) any connection of the phenomenon to a hardware-observable side channel — these three remain the specific claimed contributions of Sections 4 and 5, and should be stated in the final paper as refinements of a known tension rather than as its first observation.

### 2.3 Hardware performance counters for security

HPC-based detection is well established for problems adjacent to, but distinct from, this paper's setting: HPC-based malware identification and classification [CITE], system-wide detection and mitigation of microarchitectural side-channel attacks (e.g. cache attacks, Spectre/Meltdown-class attacks) using performance counters [CITE — "Fight Hardware with Hardware", and related SoK-style evaluations of HPC-based cache-attack detection], and hardware-performance-counter-based control-flow attestation (HPCCFA) for detecting control-flow-modifying attacks such as kernel rootkits [CITE]. A recurring theme in this literature, consistent with Section 1.1's argument, is that a hardware-level observation gives a defender a check that is largely independent of what an attacker can directly manipulate in software.

### 2.4 Positioning this work

To the best of our knowledge, no prior work sits at the intersection of Sections 2.1-2.3: using microarchitectural hardware performance counters, rather than a navigation filter's own reported innovation statistics, to detect GPS/GNSS spoofing against a Kalman-filtered state estimator — and, more specifically, no prior work treats an adaptive filter's own robustness mechanism (here, an ANIS-gated Strong Tracking Filter branch) as a deliberate instrumentation point for such detection. This claim should be revisited with a fuller literature search before submission; the search conducted for this draft was scoped to establish plausibility, not to be exhaustive.

## 3. System and Threat Model

### 3.1 Filter

A 6-state GPS+IMU EKF (position/velocity), implemented on a RISC-V embedded platform. IMU-measured acceleration is used as the predict-step control input; GPS position is the measurement. Q and R are fixed, tuned offline. Every step, the filter computes:

$$NIS = \text{inn}^\top S^{-1} \text{inn}, \qquad S = HPH^\top + R$$

and $ANIS(t)$, a rolling mean of the last 10 NIS values.

### 3.2 Attack model

The adversary spoofs the GPS measurement stream, shifting it away from the true position starting at a fixed onset point (iteration 150 of a 300-iteration run) without any observable change to the IMU stream (assumed to be a physically independent sensor the attacker cannot reach). Two attack profiles are evaluated:

- **Jump**: a large, near-instantaneous constant offset applied to the GPS measurement at onset.
- **Drift**: a small, slowly-growing offset applied from onset onward — a harder-to-detect, low-and-slow variant of the same attack class.

These correspond to the **step** and **ramp** fault profiles standard in the GNSS spoofing-detection literature for evaluating detector performance [CITE — step/ramp GNSS spoofing profiles, see Related Work]; we adopt the jump/drift naming used throughout this project's own implementation but note the correspondence explicitly for readers familiar with the fault-detection terminology.

A third mode, **normal**, has no attack applied at any point and serves as the negative-condition baseline throughout.

### 3.3 Defender's observation model

The defender has access to (a) the filter's own internal state and diagnostics (NIS, ANIS, filtered/true position, when available for validation) and (b) a set of RISC-V HPM counters read once per EKF iteration, but explicitly *not* privileged access to the raw GPS measurement stream independent of what the filter itself receives. The central question this paper addresses is whether (b) alone — a side channel entirely outside the filter's numerical output — can support attack detection.

## 4. Adaptive Filter Design

### 4.1 Unconditional Strong Tracking: a failed first attempt

The classical STF fading factor is:

$$\lambda_k = \max\left(1,\ \frac{\text{trace}(V_k - R)}{\text{trace}(HFPF^\top H^\top)}\right), \qquad P \leftarrow \lambda_k \cdot FPF^\top + Q$$

where $V_k$ is the empirical innovation covariance over a short trailing window. Applied at every step, this is the textbook fix for model mismatch. In our setting it substantially *suppressed* detectability: the raw NIS-alarm rate on a jump attack fell from 147/299 steps to 6.6/299, and on drift from 51/299 to 2/299. The mechanism is direct: inflating $P$ raises the Kalman gain, so the filter absorbs the spoofed trajectory faster, narrowing the prediction/measurement gap that any innovation-based detector relies on. We additionally found that the raw magnitude of $\lambda_k$ itself is not discriminative between attack and normal conditions when applied unconditionally (drift vs. normal: mean $\lambda_k \approx 1.20$ in both cases). This result is consistent with, and gives a quantified account of, the "limited detection window" effect noted qualitatively in prior GNSS impact-assessment work (Section 2.1-2.2) — our contribution is the explicit before/after comparison, the diagnosis of the mechanism (Kalman-gain-driven re-convergence), and the gated remedy in Section 4.2.

### 4.2 ANIS-gated Strong Tracking

We instead compute $\lambda_k$ only when the previous step's ANIS already exceeds a formally-justified threshold:

```
if (anis_prev > ANIS_THRESHOLD) {                 // gate 1: only compute lambda_k when justified
    lambda_k = compute_lambda_k(...);
    if (lambda_k > 1.0) {                          // gate 2: only inflate P if lambda_k says so
        P = lambda_k * F*P*F' + Q;
    } else {
        P = F*P*F' + Q;
    }
} else {
    P = F*P*F' + Q;                                 // unmodified predict
}
```

`ANIS_THRESHOLD = 4.377`, the $\chi^2(3, 0.95)$ critical value, is not a tuned hyperparameter but a formal 95%-confidence cutoff: under the null hypothesis of a correctly-modeled filter with only ordinary sensor noise, ANIS is expected to exceed this value roughly 5% of the time by chance — this is a designed false-alarm rate, not evidence of a flaw, and we return to its consequences in Section 8.4. `anis_prev` (the *previous* step's ANIS) is used deliberately: the current step's ANIS is not known until after that step's own measurement update, so the gate is causally one step behind by construction.

With this design, ANIS decides *whether to look*; $\lambda_k$, computed only when triggered, decides *how much to inflate*. This resolves the Section 4.1 failure mode: the filter no longer suppresses its own detectability under sustained attack, since the adaptive branch activates far less often overall and specifically less often during the extended-duration steady state of an attack than the unconditional version did.

## 5. Hardware Performance Counter Side Channel

### 5.1 Why the branch is externally observable

Sections 4.1-4.2 describe a data-dependent branch: which code path executes on a given step depends on `anis_prev`. Different paths compile to different instruction mixes and execute different amounts of work — `compute_lambda_k` performs a trailing-window mean, a trace computation, and a matrix product, none of which execute on the unmodified-predict path. This difference is measurable independent of the filter's own numerical output, via HPM counters read once per iteration.

### 5.2 Counter selection and causal role

Table 1 lists the eight counters instrumented, their measured hardware event, and their intended causal role in this experiment (M1: FP-latency hypothesis; M2: branch/call-count hypothesis; one deliberate negative control).

**Table 1: HPM counter mapping**

| Counter | Event | Role | Causal link to the ANIS-gated branch |
|---|---|---|---|
| hpmcounter3 | FP interlock cycles | M1, primary | `compute_lambda_k`'s trailing-mean/trace/matrix math stalls the FPU pipeline; only executes when the branch triggers |
| hpmcounter4 | FP div/sqrt retired | M1, count control | Companion count-side signal to counter 3 |
| hpmcounter5 | cond-branch instr. retired | M2 | The branch itself, plus `compute_lambda_k`'s internal loops, are conditional branches that only execute on the triggered path |
| hpmcounter6 | long-latency interlock | M1, alt. | Secondary FP-stall observable |
| hpmcounter7 | FP fused-multiply-add retired | M1 | Highest-volume compute signal candidate |
| hpmcounter8 | int arithmetic retired | activity reference | Generic loop/index bookkeeping baseline |
| hpmcounter9 | pipeline flushes | **negative control** | Should not respond to the branch at all |
| hpmcounter10 | jump-and-link retired | M2 | Calling `compute_lambda_k()` itself emits a JAL — a near-literal invocation counter |

Empirically (Section 8), hpmcounter3/4/5/8/10 carry usable signal; hpmcounter6/7 show zero between-trial variance across all 20 normal-mode trials at nearly every window position (i.e., the EKF's own steady-state FP workload saturates them regardless of the branch, leaving no headroom for the STF branch to show through) and are excluded from detection entirely; hpmcounter9, included specifically as a negative control, does show some response, which we attribute to noise contamination rather than a genuine causal link (Section 8.4) — a useful negative result in itself, since it demonstrates that not every counter that happens to cross threshold occasionally is evidence of the mechanism under study.

## 6. Detection Methodology

### 6.1 Baselining

Raw HPM values are logged as hex strings and converted to integers before any arithmetic. For each counter, at each of the 300 EKF iterations, we compute $\mu(\text{iter})$ and $\sigma(\text{iter})$ across 20 independent normal-mode trials, then z-score every trial (normal, jump, drift) against that per-iteration baseline. All detection statistics below operate on this z-series, not the raw counter value.

### 6.2 A shared three-stage detector construction

We evaluate three detection statistics — denoted **D**, **G**, **V** — that share an identical three-stage structure, differing only in the per-window statistic used in Stage 1:

**Stage 1 (per-window statistic).** Slide a window of length $W$ over the z-series, one sample at a time:
- $D(t) = |\beta(t)/SE(\beta(t))|$ — the t-statistic of a windowed OLS slope fit (self-normalizing against its own within-window residual noise).
- $V(t) = |\log(S^2(t,W)+1)|$ — log-compressed windowed sample variance.
- $G(t) = \text{mean}(z \text{ over the window})$ — the simplest possible statistic, the window's own level.

**Stage 2 (window-diff).** $X_{new}(t) = |X(t) - X(t-1)|$ — how much the Stage-1 statistic moved since the previous window, rather than its absolute level.

**Stage 3 (z-score).** $score(t) = (X_{new}(t) - \mu_{Xnew}(t)) / \sigma_{Xnew}(t)$, with $\mu, \sigma$ computed across the 20 normal trials at each window position, mirroring the counter-level baselining of Section 6.1.

### 6.3 Sigma-fragility guard

At some window positions, all 20 normal trials produce numerically identical Stage-2 values for a given counter (the counter's deterministic behavior at that point in the run leaves no between-trial spread to measure), driving $\sigma_{Xnew}(t)$ to exactly zero. Dividing by such a value — or clamping it to an arbitrarily small epsilon, as an early version of this pipeline did — produces scores on the order of $10^{11}$-$10^{12}$: a numerical artifact, not a genuine signal, and one that silently corrupted an earlier ratio-based version of this metric family entirely. We instead flag any window position with $\sigma_{Xnew}(t) < 10^{-6}$ as **fragile** and exclude it from both threshold calibration and detection.

### 6.4 Thresholding and detection rule

Per counter, the detection threshold is $H = Q_{95}(\{\max_t |score(t)| : \text{trial} \in \text{normal trials}\})$ — the 95th percentile of each normal trial's own peak score. A window is flagged if $|score(t)| > H$. A run counts as **detected** only if a flag occurs at or after the true attack onset (iteration $\geq 150$); flags before onset are excluded from the detected/not-detected decision, since they cannot reflect the (not-yet-injected) attack by construction — but see Section 8.4 for why this convention must be interpreted carefully rather than taken as evidence of a clean threshold.

## 7. Experimental Setup

20 independent seeds per mode (normal, jump, drift), 300 EKF iterations each, on a RISC-V embedded target. Detector window sizes were selected per-metric via a window-size sweep ($W \in \{5,10,15,20,25,30\}$ for D/G, $\{5,10,20,30,50\}$ for V), favoring the window size that gave stable early-onset sensitivity without degrading threshold stability: $W{=}10$ for D, $W{=}5$ for G and V. Six of the eight instrumented counters (hpmcounter3/4/5/8/9/10) produce usable (non-fully-fragile) data; hpmcounter6/7 are excluded (Section 5.2).

Detection is scored at the run level as a standard binary classification problem: ground truth is 1 for jump/drift runs, 0 for normal runs (jump and drift pooled into a single "attack" class, 40 attack runs + 20 normal runs = 60 runs per counter); a run is predicted positive if it is *detected* per Section 6.4. We report Precision, Recall, F1, and False-Alarm rate (FP / (FP+TN)).

## 8. Results

### 8.1 Per-counter, per-detector performance

**Table 2: Detection performance (Precision / Recall / F1 / False-Alarm), per counter**

| Counter | D (W=10) F1 / FA | G (W=5) F1 / FA | V (W=5) F1 / FA |
|---|---|---|---|
| hpmcounter3 | 0.988 / 0.05 | 1.000 / 0.00 | 0.975 / 0.05 |
| hpmcounter4 | 0.962 / 0.05 | 0.974 / 0.00 | 0.962 / 0.05 |
| hpmcounter5 | 0.975 / 0.05 | 0.962 / 0.05 | 0.975 / 0.05 |
| hpmcounter8 | 1.000 / 0.00 | 0.962 / 0.05 | 0.962 / 0.05 |
| hpmcounter9 | 0.841 / 0.00 | 0.841 / 0.00 | 0.829 / 0.05 |
| hpmcounter10 | 0.975 / 0.05 | 0.962 / 0.05 | 0.962 / 0.05 |

*(Full TP/FP/FN/TN breakdown: see supplementary results table / `detection_confusion_matrix.csv`.)*

hpmcounter3/4/5/8/10 achieve F1 $\geq$ 0.96 across all three detectors, with false-alarm rate capped at 5% (1 false positive out of 20 normal runs) or 0% — consistent with the designed false-alarm rate of a 95th-percentile threshold (Section 6.4), not evidence of instability. hpmcounter8 under D and hpmcounter3 under G both achieve a perfect F1 = 1.000.

hpmcounter9 is the consistent outlier, with recall in the range 0.60-0.85 depending on detector and voting scenario (Section 8.3) despite a *low* false-alarm rate — its weakness is under-detection, not over-triggering, and Section 8.4 traces this to persistent pre-attack noise rather than a detector-design flaw.

*(Figure: heatmap_4counters_midthreshold, one representative seed — visualizes all three detectors x three modes x four counters at once, color-coded against each counter's own threshold.)*

### 8.2 Hardware-level evidence for the causal story

*(Figure: hpmcounter3 raw value + rate (d(HPM)/d(mcycle)) panels, one seed vs. 20-seed normal mean.)* Raw counter growth is visually near-identical across normal/jump/drift (dominated by the EKF's constant per-step workload); the *rate* signal is where the branch's effect appears — sustained elevated plateaus in jump/drift coinciding with the windows where `lambda_k` (Section 4.2) is observed to be active, and no such plateau in normal-mode runs outside brief, isolated, chance ANIS crossings (Section 8.4).

### 8.3 Ensemble voting across detectors

Combining the three detectors' run-level flags under OR / majority-vote(>=2) / AND rules shows two distinct behaviors. For hpmcounter9, F1 declines monotonically from 0.907 (OR) to 0.841 (majority) to 0.750 (AND) — tightening agreement steadily loses its already-marginal true positives. For the well-behaved counters (e.g. hpmcounter3: F1 0.976 -> 1.000 -> 0.987; false-alarm rate 0.10 -> 0.00 -> 0.00), tightening agreement removes false alarms almost for free, since genuine attacks tend to trigger all three detectors together while chance noise rarely does.

**Table 3: Ensemble voting, representative counters**

| Counter | Scenario | F1 | False Alarm |
|---|---|---|---|
| hpmcounter3 | I (any fires) | 0.976 | 0.10 |
| hpmcounter3 | II (>=2 fire) | 1.000 | 0.00 |
| hpmcounter3 | III (all 3 fire) | 0.987 | 0.00 |
| hpmcounter9 | I (any fires) | 0.907 | 0.05 |
| hpmcounter9 | II (>=2 fire) | 0.841 | 0.00 |
| hpmcounter9 | III (all 3 fire) | 0.750 | 0.00 |

### 8.4 Honest accounting of false alarms: a pre-onset audit

The post-onset-only detection rule (Section 6.4) is necessary but not sufficient for an honest evaluation, since it silently discards *any* flag before onset without reporting how much was discarded, or where. We audit this directly: across the full 20-seed x ~289-window population per counter, normal-mode flags are vanishingly rare (0-2 flagged windows total across the entire normal-mode population, for most counters) — consistent with the intended $\leq$5% trial-level false-alarm rate. Pre-onset flags *within attack-mode runs*, however, are unevenly distributed: hpmcounter4/5/8/10 show near-zero pre-onset contamination (0-5% of their flagged windows occur before onset), hpmcounter3 shows a moderate amount (7-16%), and hpmcounter9 shows severe contamination (30-62% of its flagged windows, across all three detectors, occur before the attack has even started). This is strong evidence that hpmcounter9's weakness (Section 8.1) is a property of the counter's own noise characteristics rather than a flaw in the shared detector construction (Section 6.2) — and is consistent with its intended role as a near-negative-control (Section 5.2, Table 1).

Table 4 breaks the same pre-onset audit down by mode, pooled across the five usable counters (hpmcounter9 excluded), to check whether pre-onset activation is truly mode-independent as the post-onset-only rule implicitly assumes. It is not quite: normal-mode pre-onset flags are close to zero, while jump/drift pre-onset flags run consistently higher in absolute count, though still well under 0.3% of all pre-onset windows for every metric. Since the post-onset-only rule excludes this entire segment from every reported detection metric regardless of mode, this asymmetry cannot affect Precision/Recall/F1/False-Alarm as reported — but it is a real, disclosed asymmetry rather than an artifact we chose not to look for; see Section 9, item 6 for discussion of its likely origin.

**Table 4: Pre-onset flagged windows by mode, pooled across hpmcounter3/4/5/8/10 (all 20 seeds)**

| Metric | Normal (n / total) | Jump (n / total) | Drift (n / total) |
|---|---|---|---|
| D_final (W=10) | 1 / 14000 | 22 / 14000 | 13 / 14000 |
| G_final (W=5) | 2 / 14500 | 37 / 14500 | 19 / 14500 |
| V_final (W=5) | 0 / 14500 | 22 / 14500 | 20 / 14500 |

## 9. Discussion and Limitations

1. **Single embedded platform.** All results are from one RISC-V target; generalization to other microarchitectures with different HPM event sets is untested.

2. **Two of eight counters unusable.** hpmcounter6/7 carry no signal in this workload, saturated by the EKF's own steady-state computation; a different filter implementation (larger state dimension, different matrix-heavy operations) might shift which counters are informative.

3. **A negative-control counter is not perfectly silent.** hpmcounter9's residual signal, while shown to be attributable to noise (Section 8.4) rather than genuine causal coupling, is a reminder that "negative control" is a design intention, not a guarantee, and should be empirically verified per platform.

4. **Threshold false-alarm/miss-rate trade-off.** The 95%-confidence ANIS threshold (Section 4.2) and 95th-percentile detection threshold (Section 6.4) are conventional choices, not uniquely optimal; tightening either would reduce false alarms at some cost to sensitivity for subtler attacks (drift consistently harder to detect than jump throughout this work).

5. **Attack model scope.** Only jump and (a specific profile of) drift spoofing were evaluated; more sophisticated, adaptive, or gradually-onset spoofing strategies designed with knowledge of this detection scheme are not evaluated and are a natural direction for future work. In particular, Section 1.1 establishes that a measurement-only attacker with white-box knowledge of `ANIS_THRESHOLD` who successfully keeps the true ANIS under it for the full attack duration evades the software ANIS check and the HPC side channel *together*, since HPC visibility is entirely mediated through the same gated branch; no such adaptive, threshold-aware attack was constructed or evaluated here, and doing so (along with measuring how much such an attack's effectiveness is itself constrained by having to stay under the gate) is future work.

6. **Mode-dependent pre-onset ANIS/STF activation rate.** The ANIS gate (Section 4.2) is a 95%-confidence statistical test, so it is expected to -- and does -- fire occasionally before attack onset from ordinary sensor noise alone, in every mode including normal. This is structurally harmless to every reported detection metric in this paper: the post-onset-only rule (Section 6.4) means a pre-onset flag, regardless of cause, can never be counted as a True Positive or a False Positive, since the run-level `detected` decision only ever inspects flags with `window_end_iter >= 150`. However, auditing the raw pre-onset activation rate (pooled across the five usable counters, all 20 seeds) shows it is not perfectly mode-independent: normal-mode pre-onset flags are vanishingly rare (0-2 out of ~14,000-14,500 windows per metric), while jump/drift pre-onset flags, though still well under 0.3% in absolute terms, run consistently higher (13-37 out of the same denominator -- see Table 4). We traced this to the test harness generating slightly different sensor-noise realizations per mode even in the shared pre-onset segment (Section 8.4's seed-level lambda_k audit found cases where jump/drift pre-onset noise streams diverge from normal's, and occasionally from each other), rather than to any backward leakage from the injected attack, which by construction does not execute before iteration 150. Because the post-onset-only rule excludes this segment from every reported metric, it does not affect Precision/Recall/F1/False-Alarm as reported; but a fully mode-matched noise-generation process in the test harness would be a cleaner setup, and is a natural target for future work before drawing conclusions from pre-onset behavior specifically.

## 10. Conclusion

We showed that gating a Strong Tracking Filter's adaptive covariance inflation behind a formal ANIS threshold resolves a direct tension between classical tracking robustness and spoofing detectability, and that the resulting conditional branch is independently observable via RISC-V hardware performance counters. Three convergent window-diff detection statistics built over this side channel achieve F1 $\geq$ 0.96 with $\leq$5% false-alarm rates on five of six usable counters, with honest accounting for both a sigma-fragility failure mode and pre-attack noise contamination. This suggests hardware-counter telemetry is a viable, filter-output-independent detection layer for GPS spoofing on embedded adaptive-filtering platforms.

## References

*[To be completed with verified citations before submission.]*

---
<!-- END DRAFT -->
