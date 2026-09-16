<!--
DRAFT — IEEE ICC 2027, Communication QoS, Reliability, and Modeling (CQRM)
Symposium. Deadline: October 2, 2026. Target length: 6 pages (+2 overlength
pages allowed, $100/page).

STANDALONE PAPER: this paper does NOT cite, reference, or point to the
statistical-detector work (D_final/G_final/V_final construction, the
ANIS-gated STF causal mechanism) as "our companion work," "our prior
work," or "related work" -- that fuller journal treatment (Springer HaSS)
is being submitted AFTER this one, and at this paper's own submission time
it is not a citable prior publication. Section 3 below therefore describes
the system and D/G/V feature construction as this paper's OWN setup,
condensed but self-contained. Keep it that way in any future edit.

ML scope (decided): one-class only (Mahalanobis distance + Local Outlier
Factor) -- no supervised classifiers. Feature: longest consecutive run of
threshold-crossing windows per counter/metric (the persistence feature),
not max(|score|). Candidate counters: hpmcounter3/4/5/8/10 (5); hpmcounter3
excluded from the final fused model (Section 5.2), leaving 4
(hpmcounter4/5/8/10). Evaluation: Leave-One-Seed-Out cross-validation,
standard + stress evaluations.

Figures actually generated and referenced below (real files, already in
the repo):
  - Figure 1: Trajectory/plots/seed1_normal_jump_drift_3d_ecef_measured_column.png
    (jump/drift/normal ECEF trajectory comparison -- defines the attack
    model visually; lives outside paper/, adjust the relative path when
    assembling the submission package)
  - Figure 2: plots_heatmap/ml_persistence_prepost_boxplot_jump.png (compact
    grouped boxplot, ~3.2in x 2.3in -- replaces the earlier 40-row heatmap,
    which was too tall/dominant for a double-column page despite being
    column-width-correct)
    (or the drift one / both side by side -- pick one when finalizing)
  - Figure 3: plots_heatmap/ml_fusion_threshold_sweep.png
One additional figure is left as an explicit TODO placeholder (a bar-chart
summary of Tables 2-3) -- generate only if space allows after the rest of
the draft is locked; the numbers are already fully conveyed by the tables.

All [CITE]/TODO markers are unverified placeholders -- no fabricated
citations. Bibliography keys with real, checkable sources are used
directly (see paper/references.bib); keys prefixed todo_ are generic
placeholders for broad claims not yet backed by a specific verified source.
-->

# Reliable Anomaly Detection for GPS-Dependent Networked Systems via Unsupervised Fusion of Hardware Telemetry

**Authors:** [Author names], [Affiliation]

## Abstract

GPS-derived position estimates underpin the reliability of a wide range of networked and cyber-physical systems, from autonomous vehicles to distributed time synchronization. GPS spoofing silently degrades this reliability by shifting the measurement away from ground truth without any structural change in the receiving system. We build a hardware-telemetry-based anomaly detector for a GPS/IMU extended Kalman filter (EKF) running on an embedded RISC-V platform: three window-based statistical features (D, G, V), computed per hardware performance counter, capture the data-dependent execution footprint of the filter's own adaptive-covariance logic -- a channel independent of the position estimate itself. We show that an unsupervised, one-class fusion model over these features, requiring no labeled attack examples at calibration time, reliably distinguishes spoofed from unspoofed operation across 20 independent trials per condition, evaluated via leave-one-seed-out cross-validation, achieving perfect recall with zero false alarms (F1=1.000) on the conventional evaluation. We further stress-test reliability by scoring each spoofed trial's own pre-attack segment as a held-out negative case, and show that (i) a persistence-aware feature -- rewarding sustained rather than momentary threshold-crossing -- cuts the resulting false-alarm rate by roughly 3x over a naive peak-value feature at no cost to recall, and (ii) characterizing and excluding a single systematically-biased telemetry channel from the fused feature set resolves nearly all remaining false alarms (F1=0.988 under this harder evaluation). Our results argue that reliable fusion of hardware telemetry for anomaly detection depends as much on principled feature and channel design as on the fusion algorithm itself.

**Keywords:** reliability, anomaly detection, GPS spoofing, hardware telemetry, unsupervised learning, sensor fusion

---

## 1. Introduction

Networked and cyber-physical systems increasingly depend on GPS-derived position and timing estimates for correct, reliable operation — autonomous vehicle coordination, distributed sensor networks, and time-synchronized communication infrastructure all inherit their reliability, in part, from the reliability of the underlying positioning solution [CITE]. This dependency introduces a specific reliability threat: GPS spoofing, in which an adversary (or, in benign cases, multipath/interference) shifts the received measurement away from ground truth. A GPS/INS state estimator with fixed process and measurement noise covariances has no internal mechanism to notice when this has happened — it simply absorbs the corrupted measurement at whatever rate its Kalman gain allows, and the system's downstream reliability degrades silently, with no structural indication that anything is wrong [CITE].

Detecting this class of reliability degradation is normally approached at the level of the position estimate itself — innovation-based statistical tests (chi-squared/NIS monitoring) on the state estimator's own residuals [CITE]. This is effective, but it is a single observation layer: it depends entirely on quantities the estimator itself computes and reports, which is also why recent work has shown that spoofing can be engineered specifically to stay under such a monitor's threshold [CITE]. The system we study here (Section 3) instead gates an adaptive covariance-inflation mechanism — a Strong Tracking Filter, used to maintain estimator robustness under model mismatch — behind a formal innovation threshold, so the adaptive branch only executes conditionally. This gating decision is a data-dependent code path, and its execution leaves a measurable footprint on the embedded processor's own hardware performance counters (HPCs) — FP-pipeline stalls, conditional-branch retirements, jump-and-link retirements — independent of the estimator's self-reported output. This gives a defender a second, physically-grounded observation of the same underlying reliability question.

We construct three per-counter statistical features from this telemetry (window-diff detectors, denoted D, G, V; Section 3) and ask a reliability-modeling question directly motivated by deployment realism: **can these per-counter signals be fused into a single anomaly decision by an unsupervised, learned model — one that never requires a labeled attack example — rather than a hand-designed, fixed combination rule?** Labeled spoofing incidents are rarely available in advance for a deployed reliability-monitoring system, whereas normal operating telemetry is abundant; a fusion approach that can exploit only the latter is the more realistic design point.

A natural alternative to a learned fusion is simpler still: threshold each counter's statistical feature independently against its own normal-operation baseline, then combine the resulting per-counter flags with a fixed rule such as "flag if any counter crosses its threshold" or "flag if a majority do." This alternative is brittle in a specific, structural way: it collapses each counter's continuous evidence into a single bit *before* combining, discarding exactly the information — how far past threshold, and whether several counters are moderately elevated together rather than one sharply so — that would let a combiner tell a genuine joint anomaly apart from an isolated quirk in one channel. A fixed threshold-and-vote rule is also restricted to axis-aligned per-counter cutoffs joined by simple Boolean logic, regardless of how the counters' features actually co-vary under normal operation. An unsupervised one-class model instead operates directly on the continuous, multi-counter feature vector (Section 4.1) and produces a smooth distance or density score shaped by the real joint structure of normal-operation data: several counters shifting together, even mildly, can cross its decision boundary the same way one counter shifting sharply can, without either behavior being hard-coded in advance. This is the specific sense in which fusion is *learned* rather than designed, and it is the primary reason we pursue it over the simpler alternative.

We show that an unsupervised, one-class fusion model over these per-counter statistical features achieves reliable detection, evaluated with leave-one-seed-out cross-validation so that every reported result is on data the model never saw during calibration. We additionally show that the choice of *temporal aggregation* within each counter's own feature construction matters as much as the choice of fusion algorithm: aggregating each counter's windowed statistic by its single most extreme value is vulnerable to isolated noise excursions, while aggregating by the *longest sustained run* of threshold-crossing behavior is far more robust, at no cost to detection sensitivity.

This paper makes three contributions:

1. We show that an unsupervised one-class model — evaluated here with two representative approaches, Mahalanobis distance and Local Outlier Factor — can fuse hardware-telemetry-derived features across multiple counters into a single reliable anomaly decision, without ever requiring a labeled anomaly example at calibration time.

2. We show that a persistence-aware feature — the longest run of consecutive time windows exhibiting anomalous statistical behavior, rather than the single most extreme value observed — substantially improves the robustness of this fusion, and give a direct, controlled comparison against the simpler alternative.

3. We give an honest, cross-validated account of detection reliability, including a documented case where one candidate hardware counter carried a systematic bias unrelated to the anomaly of interest, and show how excluding it restores clean detection performance — illustrating a general principle for reliability-modeling practice: not every available telemetry channel is safe to fuse without individual characterization.

## 2. Related Work

GPS/GNSS spoofing detection is typically approached at the level of the receiver or the navigation filter's own residuals [CITE]; recent work has also shown that spoofing can itself be engineered at the physical layer specifically to evade such detectors [yao2025physicalspoofing], motivating detection channels that do not depend solely on the plausibility of the externally observable signal. Within IEEE ICC specifically, GPS/GNSS reliability under adverse conditions is an active topic: GPS-denied positioning [tsikteris2025pioneer] and GNSS/auxiliary-sensor fusion through outages [qmul2026gnssdas] both address reliability failure modes adjacent to, though distinct from, the spoofing scenario studied here. Side channels have separately been shown to leak security-relevant state even through nominal protections — encrypted location-based-service traffic has been shown to leak positioning information [leaking2025lbs] — direct precedent for this paper's premise that a channel outside a system's primary output (here, hardware telemetry rather than network traffic) can carry exploitable signal about that system's internal state. Robust fusion across multiple, individually unreliable input channels has also been studied for federated wireless systems, where attention-based weighting learns to down-weight untrustworthy participants rather than exclude them by hand [zhang2024federated]; we return to this as a natural extension of our own hand-diagnosed channel exclusion (Section 5.2) in Section 6. Supervised ML classifiers have separately been applied to vehicular network attack detection using labeled datasets [qian2024vanet]; our approach differs in requiring no labeled attack data at all. Unsupervised/one-class anomaly detection more broadly is well established for HPC-based malware and side-channel-attack detection [CITE], though not previously applied, to our knowledge, to sensor-spoofing detection against a state estimator.

## 3. System and Feature Construction

We study a 6-state GPS+IMU extended Kalman filter (EKF) on a RISC-V embedded platform, with fixed process/measurement noise covariances as is standard for resource-constrained embedded targets. The **Strong Tracking Filter (STF)** is a standard adaptive-filtering technique: when a Kalman filter's predictions stop tracking its measurements well, it inflates the filter's state covariance so the filter re-converges faster. Whether that mismatch is occurring is judged from the **innovation** — the difference between the measurement the filter predicted and the measurement actually received at that step — summarized as its normalized squared value, the **innovation statistic (NIS)**, and smoothed into a rolling-mean form, **ANIS**, to average out ordinary sensor noise rather than reacting to every momentary fluctuation.

Our filter's STF branch is *gated*: it executes only when ANIS exceeds a formal chi-squared threshold, i.e., only when independent statistical evidence already suggests the model no longer fits, rather than inflating covariance at every step unconditionally. Gating is necessary because an unconditional STF re-converges onto a spoofed trajectory fast enough to suppress the very innovation-based anomaly a detector would otherwise observe. Because this gate is a data-dependent branch — it executes different instructions than the unmodified predict step — its activity leaves a measurable footprint on the processor's hardware performance monitor (HPM) counters (FP-pipeline stalls, conditional-branch retirements, jump-and-link retirements), independent of the filter's own numerical output.

The adversary spoofs the GPS measurement stream, shifting it away from the true position from a fixed onset point onward, with no corresponding change to the (physically independent) IMU stream. We evaluate two attack profiles: a **jump** — a large, near-instantaneous constant offset applied to the GPS measurement at onset — and a **drift** — a small, slowly-growing offset applied from onset onward, a harder-to-detect, low-and-slow variant of the same attack class. A third condition, **normal**, applies no attack at any point and serves as the negative baseline throughout. Figure 1 shows representative raw-GPS-measurement trajectories (ECEF coordinates) for all three conditions from a single seed: the jump attack displaces the measurement to a distant, disjoint region of the trajectory space at onset, while the drift attack diverges gradually from the same starting point as the normal trajectory — a qualitative preview of why drift is consistently the harder of the two attacks to detect throughout this paper.

*(Figure 1: `Trajectory/plots/seed1_normal_jump_drift_3d_ecef_measured_column.png` — already generated and sized for a single IEEE column (boxed in-plot legend, all three ECEF axis labels legible, no outer whitespace). Path is relative to the repo root; adjust when assembling the final submission package. Recommended as a real figure — it is the only visual in the paper that grounds "jump" and "drift" concretely rather than by name alone.)*

Specifically, we instrument five RISC-V hardware performance monitor (HPM) counters: floating-point interlock cycles (hpmcounter3), floating-point divide/square-root instructions retired (hpmcounter4), conditional-branch instructions retired (hpmcounter5), integer arithmetic instructions retired (hpmcounter8), and jump-and-link instructions retired (hpmcounter10). Each was chosen because the gated STF branch's internal computation — a trailing-window mean, a trace computation, and a matrix product, none of which execute on the unmodified predict path — plausibly shifts that specific event count when the branch fires. Empirically, all five show a measurable, consistent difference between normal-operation trials and spoofed trials once the branch becomes active under attack, which is the basis for treating each of them as an independent evidence source in the fusion methodology below.

We construct three per-counter statistical features from this telemetry, denoted **D**, **G**, and **V**, that share an identical three-stage construction and differ only in which per-window statistic seeds them. First, a statistic is computed over a sliding window of the counter's own z-scored time series: **D** is the t-statistic of a windowed ordinary-least-squares slope fit, capturing whether the counter is trending in a consistent direction within the window, self-normalized against its own residual noise; **G** is simply the window's own mean level, the most direct measure of whether activity is elevated; **V** is the log-compressed windowed sample variance, capturing whether the counter's short-term variability itself has changed. Second, this per-window statistic is differenced across consecutive windows, so each detector responds to how much its own statistic *moved* since the previous window rather than to its absolute level — which would otherwise be dominated by each counter's own baseline scale rather than by the change of interest. Third, this window-difference is z-scored against its own value, at the matching window position, across the 20-trial normal-operation baseline, giving a directly comparable, unitless score per counter per window, with a guard against near-zero baseline variance that would otherwise produce numerically unstable scores. Using three statistics that each capture a different distributional property of the counter's behavior — level, trend, and spread — rather than a single one is a deliberate redundancy: it protects against an anomaly showing up in one property but not another, and it is what gives the fusion methodology below more than a single vote per counter.

Of the HPM counters instrumented on this platform, five (hpmcounter3/4/5/8/10) carry a usable signal traceable to the gating branch; each contributes a D, G, and V score, giving up to 15 raw per-counter statistical scores per trial as the input to the fusion methodology below (Section 5.2 narrows this candidate set further, to four).

## 4. Unsupervised Fusion Methodology

### 4.1 From statistical scores to a persistence feature

For each of the 5 candidate counters (narrowed to 4 in Section 5.2) x 3 statistical detectors (D, G, V) — up to 15 features per trial — we compare two ways of collapsing a detector's per-window score sequence into a single scalar feature. The original construction takes the single largest $|score|$ observed anywhere in the trial's window range: a natural choice, but one vulnerable to an isolated noisy window, which inflates the feature identically to a genuine, sustained anomaly. We instead use the **persistence feature**: the longest run of *consecutive* windows whose $|score|$ exceeds that detector's own threshold $H$. A single noisy window still produces a run of length 1; a real, sustained anomaly produces a much longer run. Section 5.1 gives the empirical comparison.

### 4.2 One-class fusion models

We fit two representative one-class models on the resulting feature vectors, using only normal-operation trials at calibration time — no attack example is ever used to fit either model.

**Mahalanobis distance.** We estimate the covariance of the normal-trial feature distribution with Ledoit-Wolf shrinkage [ledoitwolf2004] rather than the sample covariance directly. With as few as 19 normal trials available per cross-validation fold (Section 4.3) against up to 15 features, the sample covariance is poorly conditioned; shrinkage substantially stabilized the resulting distances in our experiments and was necessary to get usable Mahalanobis scores at all at this sample size.

**Local Outlier Factor (LOF).** A density-based method that compares a point's local neighborhood density to that of its neighbors, requiring no covariance estimate.

For both methods, the anomaly threshold is set at the most extreme 5% of the training fold's own score distribution — the 95th percentile of Mahalanobis distance, or the 5th percentile of LOF's score (LOF's convention scores denser, more normal-looking points *higher*, so the extreme tail sits at the low end instead). Both are the same underlying design choice — a nominal 5% training-exceedance rate — read off each method's own scale. Section 5.4 shows this choice sits inside a stable operating plateau rather than at a sensitive cliff edge.

### 4.3 Evaluation protocol

We evaluate with leave-one-seed-out (LOSO) cross-validation: for each of 20 independent seeds, the one-class model is fit on the other 19 seeds' normal-operation trials only, then used to score the held-out seed; every seed is held out exactly once, and no attack label is ever used to fit any fold.

We report two evaluations that differ only in which trials populate the negative (ground-truth-0) class. The **standard** evaluation scores genuinely normal trials against the post-onset segment of spoofed trials — the conventional question. The **stress** evaluation additionally scores each spoofed trial's own *pre-onset* segment (before the attack starts) as an additional held-out negative case: since this segment is, by construction, still normal operation, a reliable detector must not flag it either. The stress evaluation is what exercises the two robustness results below; the standard evaluation alone cannot distinguish between the feature or channel choices compared in Sections 5.1-5.2, since both already score near-perfectly under it.

## 5. Results

All results are from leave-one-seed-out cross-validation over 20 independent trials per condition, so every reported prediction is on a trial withheld from calibration. We build up to the final configuration in three steps: first the feature-design choice (Section 5.1), then the counter-selection choice (Section 5.2) — each compared against the natural starting point of all 5 candidate counters — before reporting the resulting final configuration's headline reliability numbers (Section 5.3) and its threshold sensitivity (Section 5.4).

### 5.1 Why the persistence feature matters

We start from the natural baseline: all 5 candidate counters (Section 3), with each detector's per-window score sequence collapsed to a single feature by the *single largest* $|score|$ observed anywhere in the trial's window range. We compare this max-score feature against the persistence feature (Section 4.1) under the stress evaluation, which is the only evaluation that can distinguish between them (Section 4.3).

**Table 1: Feature design ablation, stress evaluation, 5 counters**

| Feature | Method | False-Alarm | F1 |
|---|---|---|---|
| Max score | Mahalanobis | 0.583 | 0.696 |
| Max score | LOF | 0.500 | 0.727 |
| **Persistence** | **Mahalanobis** | **0.167** | **0.889** |
| **Persistence** | **LOF** | **0.167** | **0.889** |

A single isolated noisy window inflates a max-based summary regardless of whether it is part of a genuine sustained anomaly; the persistence feature only responds to sustained deviations, cutting the stress-evaluation false-alarm rate by roughly 3x for both fusion methods with no cost to recall (both remain at 1.000, matching the final configuration reported in Table 3 below). We adopt the persistence feature from this point on.

*(Figure 2: `plots_heatmap/ml_persistence_prepost_boxplot_jump.png` — compact grouped boxplot (~3.2in x 2.3in, generated by `ml/ml_persistence_feature_heatmaps.py`): persistence values pooled across all 5 counters and 20 seeds, one pair of boxes per metric (D/G/V), pre-onset vs. post-onset. Pre-onset sits flat at zero for all three metrics; post-onset is clearly elevated, most sharply for G — direct visual support for why the persistence feature discriminates well, in a fraction of the space of the original per-seed, per-feature heatmap (which was column-width-correct but too tall/dominant for a double-column page; that version is still available as `ml_persistence_test_features_jump_heatmap_column.png` if per-seed granularity is ever wanted). Optional; the numeric story is already complete from Table 1 alone.)*

### 5.2 Not every counter is safe to fuse

With the persistence feature now fixed, we traced the remaining stress-evaluation false alarms in Table 1 to their source and found them concentrated almost entirely in one candidate counter, hpmcounter3, across all three of its statistical features, for a small subset of trials — a sustained, multi-window deviation unrelated to the anomaly being detected. Excluding this single counter from the fused feature set (5 counters $\to$ 4: hpmcounter4/5/8/10) resolves nearly all of the remaining false alarms:

**Table 2: Counter selection ablation, stress evaluation, persistence feature**

| Counter set | Method | False-Alarm | F1 |
|---|---|---|---|
| All 5 counters | Mahalanobis / LOF | 0.167 | 0.889 |
| **4 counters (hpmcounter3 excluded)** | **Mahalanobis / LOF** | **0.017** | **0.988** |

This is the paper's general reliability-modeling lesson: a fused anomaly model is only as reliable as its least-reliable input channel, and a systematic, sustained bias in one channel is not something the fusion algorithm — however capable — can be expected to average away on its own. Telemetry channels intended for a fused reliability model should be individually characterized before being trusted, not assumed interchangeable by default. We adopt this 4-counter set as the final configuration from this point on.

### 5.3 Detection reliability (final configuration)

Combining the two design choices just established — the persistence feature (Section 5.1) and the 4-counter set with hpmcounter3 excluded (Section 5.2) — gives the paper's final configuration. Table 3 reports its full reliability picture: both evaluations (Section 4.3), both fusion methods, both fit exclusively on normal-operation trials.

**Table 3: Detection reliability, final configuration**

| Evaluation | Method | Recall | False-Alarm (95% CI) | F1 |
|---|---|---|---|---|
| Standard | Mahalanobis | 1.000 | 0.000 [0.000, 0.161] | 1.000 |
| Standard | LOF | 1.000 | 0.000 [0.000, 0.161] | 1.000 |
| Stress | Mahalanobis | 1.000 | 0.017 [0.003, 0.089] | 0.988 |
| Stress | LOF | 1.000 | 0.017 [0.003, 0.089] | 0.988 |

Recall is perfect in every configuration reported in this paper — the two robustness contributions in Sections 5.1-5.2 trade off false-alarm rate, never detection sensitivity.

### 5.4 Threshold sensitivity

The percentile threshold in Section 4.2 (95th for Mahalanobis, 5th for LOF) was not chosen at an unmotivated or fragile operating point. Sweeping this percentile on the final 4-counter, persistence-feature configuration shows F1=0.988 held flat across a wide range for each method (Mahalanobis: percentile 85-96; LOF: percentile 4-8), degrading only outside that range — too loose a threshold admits false alarms, too strict a threshold begins costing recall. Both chosen operating points sit inside their respective plateau, not at its edge.

*(Figure 3: `plots_heatmap/ml_fusion_threshold_sweep.png` — two-panel sweep of False-Alarm and Recall vs. threshold percentile, one panel per method, with the chosen operating point marked. Already generated; recommended as a real figure, not a placeholder, since it directly preempts an obvious reviewer question about the 95th/5th percentile choice.)*

*(Optional Figure 4, TODO — not yet generated: a bar-chart summary of Tables 2-3 side by side (Max vs. Persistence vs. Persistence-minus-hpmcounter3), for a reader skimming figures only. Generate only if page budget allows; the numbers are already fully conveyed by the tables above.)*

## 6. Discussion and Limitations

All results are from a single embedded RISC-V platform; generalization to other microarchitectures with different HPM event sets is untested. The persistence feature and pre/post-onset split assume known onset timing, appropriate for this offline evaluation; a streaming/online deployment would need a rolling-window reformulation of the same feature — a continuously-updated run-length counter rather than a batch longest-run computation over a completed window — a reformulation of the existing causal, backward-looking per-window scores, not new detection machinery, though it introduces a latency/robustness tradeoff we do not evaluate here. The channel-exclusion result in Section 5.2 was diagnosed by hand: we traced the residual false alarms to one counter and removed it. A learned, per-channel weighting scheme — as explored for robust fusion in federated wireless systems under compromised participants [zhang2024federated] — could automate this step and generalize to deployments where the unreliable channel is not known in advance; we leave this as future work. Only two attack profiles (a large near-instantaneous jump and a slow drift) were evaluated; more sophisticated, adaptive spoofing strategies designed with knowledge of this detection scheme are not evaluated here.

## 7. Conclusion

We showed that an unsupervised, one-class fusion model over hardware-performance-counter-derived statistical features reliably detects GPS spoofing against a GPS/IMU EKF, without ever requiring a labeled attack example, achieving F1=1.000 under a conventional evaluation and F1=0.988 under a harder pre-onset stress test. Two design choices were responsible for this reliability: aggregating each per-counter score by its persistence (longest sustained run) rather than its peak value, and characterizing and excluding a single telemetry channel found to carry a systematic bias unrelated to the anomaly of interest. Both lessons are general to reliability modeling over fused hardware telemetry, independent of the specific fusion algorithm used.

## References

Real, checkable sources (see `paper/references.bib`):

- [yao2025physicalspoofing] Z. Yao et al., "A Novel Physical Spoofing Technique Using Radio Frequency Fingerprint Emulation and Model Fitting," IEEE ICC 2025.
- [tsikteris2025pioneer] S. Tsikteris et al., "PIONEER: Positioning of Targets in Featureless GPS Denied Environments," IEEE ICC 2025.
- [qmul2026gnssdas] "An Augmented GNSS-DAS Architecture for Continuous and Robust Positioning," IEEE ICC 2026. (Author list, exact track, and DOI need final verification — see references.bib note.)
- [leaking2025lbs] "Encrypted Yet Leaking: Analyzing Side-Channel Vulnerabilities in Location Privacy of LBS," IEEE ICC 2025. (Author list needs final verification.)
- [zhang2024federated] H. Zhang et al., "Federated Learning with Dual Attention for Robust Modulation Classification under Attacks," IEEE ICC 2024, pp. 5238-5243.
- [qian2024vanet] S. Qian, "Machine Learning-Based Detection of Data Replay and Data Replay Sybil Attacks for Vehicular Communication Networks," IEEE ICC 2024, pp. 5202-5207.
- [ledoitwolf2004] O. Ledoit and M. Wolf, "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices," Journal of Multivariate Analysis, 88(2):365-411, 2004.

Generic placeholders still needed (broad claims, no specific source picked yet): GNSS-dependent networked-system reliability; EKF/state-estimator spoofing degradation; chi-squared/NIS-based spoofing detection; HPC-based malware/side-channel-attack detection.

---
<!-- END DRAFT -->
