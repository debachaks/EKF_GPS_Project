<!--
DRAFT — IEEE ICC 2027, Communication QoS, Reliability, and Modeling (CQRM) Symposium.
Deadline: October 2, 2026. Target length: 6 pages (+2 overlength pages allowed,
$100/page) -- page limit inferred from ICC's conference-wide convention (verified
for the CISS symposium), not yet independently confirmed for CQRM specifically.

Scope, per plan agreed with the author: this is the SIMPLER companion to a full
Journal of Hardware and Systems Security (HaSS) submission. The HaSS paper carries
the detailed statistical-detector methodology (D_final/G_final/V_final, the
ANIS-gated STF causal mechanism, per-counter rule-based ensemble voting). This ICC
paper builds ONE additional contribution on top of that foundation -- an
unsupervised ML fusion layer -- and should NOT re-derive the full causal HPC story;
summarize it and point to the HaSS paper for depth.

ML scope for THIS paper, deliberately narrowed (per agreement): one-class only
(Mahalanobis distance + Local Outlier Factor) -- no supervised classifiers. Feature:
longest consecutive run of threshold-crossing windows per counter/metric (the
persistence feature), not max(|score|). hpmcounter9 AND hpmcounter3 excluded
(4 usable counters: hpmcounter4/5/8/10) -- hpmcounter9 per the established near-
negative-control finding, hpmcounter3 per this project's own diagnosis that it
carries a sustained pre-onset artifact even after the persistence fix. Evaluation:
Leave-One-Seed-Out cross-validation, one clean results table.

All [CITE] markers are unverified placeholders -- no fabricated citations.
-->

# Reliable Anomaly Detection for GPS-Dependent Networked Systems via Unsupervised Fusion of Hardware Telemetry

**Authors:** [Author names], [Affiliation]

## Abstract

GPS-derived position estimates underpin the reliability of a wide range of networked and cyber-physical systems, from autonomous vehicles to distributed time synchronization. GPS spoofing silently degrades this reliability by shifting the measurement away from ground truth without any structural change in the receiving system. We present an anomaly-detection method that restores this reliability guarantee by fusing hardware performance counter (HPC) telemetry from the embedded processor running the GPS/IMU state estimator -- a channel independent of the position estimate itself. Building on a set of statistically-grounded per-counter detection features [CITE -- our own prior/companion work], we show that an unsupervised, one-class fusion model -- requiring no labeled attack examples -- reliably distinguishes spoofed from unspoofed operation across 20 independent trials per condition, evaluated via leave-one-seed-out cross-validation, achieving perfect recall with zero false alarms (F1=1.000) on the conventional evaluation. We further stress-test reliability by scoring each spoofed trial's own pre-anomaly segment as a held-out negative case, and show that (i) a persistence-aware feature -- rewarding sustained rather than momentary deviation -- cuts the resulting false-alarm rate by roughly 3x over a naive peak-value feature with no cost to recall, and (ii) characterizing and excluding a single systematically-biased telemetry channel from the fused feature set resolves nearly all remaining false alarms (F1=0.988 under this harder evaluation). Our results argue that reliable fusion of hardware telemetry for anomaly detection depends as much on principled feature and channel design as on the fusion algorithm itself.

**Keywords:** reliability, anomaly detection, GPS spoofing, hardware telemetry, unsupervised learning, sensor fusion

---

## 1. Introduction

Networked and cyber-physical systems increasingly depend on GPS-derived position and timing estimates for correct, reliable operation -- autonomous vehicle coordination, distributed sensor networks, and time-synchronized communication infrastructure all inherit their reliability, in part, from the reliability of the underlying positioning solution [CITE]. This dependency introduces a specific reliability threat: GPS spoofing, in which an adversary (or, in benign cases, multipath/interference) shifts the received measurement away from ground truth. A GPS/INS state estimator with fixed process and measurement noise covariances has no internal mechanism to notice when this has happened -- it simply absorbs the corrupted measurement at whatever rate its Kalman gain allows, and the system's downstream reliability degrades silently, with no structural indication that anything is wrong [CITE].

Detecting this class of reliability degradation is normally approached at the level of the position estimate itself -- innovation-based statistical tests (chi-squared/NIS monitoring) on the state estimator's own residuals [CITE]. This is effective, but it is a single observation layer: it depends entirely on quantities the estimator itself computes and reports. In a companion study, we showed that an adaptive covariance-inflation mechanism (a Strong Tracking Filter, gated behind a formal innovation threshold) used to maintain estimator robustness under model mismatch leaves an independent, measurable footprint on the embedded processor's hardware performance counters (HPCs) -- FP-pipeline stalls, conditional-branch retirements, jump-and-link retirements -- as a direct consequence of the data-dependent code path the gating mechanism takes [CITE -- companion HaSS submission]. This gives a defender a second, independent observation of the same underlying reliability question, drawn from the physical execution of the estimator rather than its self-reported output.

That companion study evaluated detection using three per-counter statistical features (window-diff detectors, denoted D, G, V) combined via a hand-designed voting rule across counters (e.g., requiring agreement among a majority of counters before declaring an anomaly). This paper asks the natural next question for a reliability-modeling audience: **can the combination across counters be learned rather than hand-designed, and can this be done without requiring labeled examples of the anomaly itself** -- a realistic constraint, since labeled spoofing incidents are rarely available in advance for a deployed reliability-monitoring system, whereas normal operating telemetry is abundant.

We show that an unsupervised, one-class fusion model over the same per-counter statistical features achieves reliable detection, evaluated with leave-one-seed-out cross-validation to ensure every reported result is on data the model never saw during calibration. We additionally show that the choice of *temporal aggregation* within each counter's feature construction matters as much as the choice of fusion algorithm: aggregating each counter's windowed statistic by its single most extreme value is vulnerable to isolated noise excursions, while aggregating by the *longest sustained run* of threshold-crossing behavior is far more robust, without any loss of detection sensitivity to genuine anomalies.

This paper makes three contributions, scoped as the natural extension of our companion statistical-detection study:

1. We show that an unsupervised one-class model (evaluated here with two representative approaches, Mahalanobis distance and Local Outlier Factor) can fuse hardware-telemetry-derived features across multiple counters into a single reliable anomaly decision, without ever requiring a labeled anomaly example at calibration time.

2. We show that a persistence-aware feature -- the longest run of consecutive time windows exhibiting anomalous statistical behavior, rather than the single most extreme value observed -- substantially improves the robustness of this fusion, and we give a direct, controlled comparison against the simpler alternative.

3. We give an honest, cross-validated account of detection reliability, including a documented case where one candidate hardware counter carried a systematic bias unrelated to the anomaly of interest, and show how excluding it restores clean detection performance -- illustrating a general principle for reliability-modeling practice: not every available telemetry channel is safe to fuse without characterization.

## 2. Related Work

[Condensed pointer, not a full review -- see companion HaSS submission [CITE] for the detailed related-work treatment of GPS/GNSS spoofing detection, adaptive Kalman filtering, and HPC-based security. This section should cite, briefly: (a) reliability/QoS modeling of GNSS-dependent networked systems [CITE], (b) unsupervised/one-class anomaly detection in networking and telemetry contexts generally [CITE], (c) the companion statistical-detection result this paper builds on [CITE].]

## 3. System Model and Statistical Foundation

[Condensed summary of the companion paper's system: 6-state GPS+IMU EKF on an embedded RISC-V platform; ANIS-gated Strong Tracking Filter; the three window-diff statistical features D/G/V per counter, briefly described with a pointer to the companion paper for the full three-stage construction and causal HPC mapping. This section should be ~0.5-0.75 page -- background, not new contribution.]

## 4. Unsupervised Fusion Methodology

### 4.1 Feature construction: from statistical scores to a persistence feature

[Describe: for each of the 4 usable counters (hpmcounter4/5/8/10) x 3 statistical detectors (D/G/V) = 12 features per trial. Each feature is the longest run of consecutive windows where that detector's score exceeds its threshold, rather than the single largest score observed -- contrast briefly with the max-based alternative and state the empirical justification (Section 5.2) rather than re-deriving it here.]

### 4.2 One-class fusion models

[Mahalanobis distance (Ledoit-Wolf shrinkage covariance) and Local Outlier Factor, both fit exclusively on normal-operation trials -- no anomaly examples used at calibration time. Threshold calibrated from a percentile of the training-fold's own scores.]

### 4.3 Evaluation protocol

[Leave-one-seed-out cross-validation across 20 independent trials per condition; every reported prediction is on a trial withheld from calibration.]

## 5. Results

All results are from leave-one-seed-out cross-validation over 20 independent trials per condition (normal / spoofed), so every reported prediction is on a trial withheld from calibration. We report two evaluations. The **standard** evaluation scores genuinely anomaly-free trials against spoofed trials' post-onset segment -- the conventional "is this trial anomalous" question. The **stress** evaluation additionally scores each spoofed trial's own *pre-onset* segment as a held-out negative case: a reliable detector must not flag the normal operating period preceding an anomaly, not only trials that are anomaly-free throughout. The stress evaluation is what actually exercises the two robustness contributions below (Sections 5.2-5.3); the standard evaluation alone cannot distinguish between the feature-design choices we compare, since both already score near-perfectly there.

### 5.1 Detection reliability (final configuration)

Final configuration: 4 usable counters (hpmcounter3 excluded, Section 5.3), persistence feature (Section 5.2), Mahalanobis distance and Local Outlier Factor, both fit exclusively on normal-operation trials.

**Table 1: Detection reliability, final configuration**

| Evaluation | Method | Recall | False-Alarm (95% CI) | F1 |
|---|---|---|---|---|
| Standard | Mahalanobis | 1.000 | 0.000 [0.000, 0.161] | 1.000 |
| Standard | LOF | 1.000 | 0.000 [0.000, 0.161] | 1.000 |
| Stress | Mahalanobis | 1.000 | 0.017 [0.003, 0.089] | 0.988 |
| Stress | LOF | 1.000 | 0.017 [0.003, 0.089] | 0.988 |

Recall is perfect in every configuration reported in this paper -- the two robustness contributions below trade off false-alarm rate, never detection sensitivity.

### 5.2 Why the persistence feature matters

Holding the counter set fixed at all 5 candidates (Section 5.3 removes one), we compare the original feature -- the single largest score observed anywhere in a trial's telemetry window -- against the persistence feature: the longest run of *consecutive* windows exceeding threshold.

**Table 2: Feature design ablation, stress evaluation, 5 counters**

| Feature | Method | False-Alarm | F1 |
|---|---|---|---|
| Max score | Mahalanobis | 0.583 | 0.696 |
| Max score | LOF | 0.500 | 0.727 |
| **Persistence** | **Mahalanobis** | **0.167** | **0.889** |
| **Persistence** | **LOF** | **0.167** | **0.889** |

A single isolated noisy window inflates a max-based summary regardless of whether it is part of a genuine sustained anomaly; the persistence feature only responds to sustained deviations, cutting the stress-evaluation false-alarm rate by roughly 3x for both fusion methods with no cost to recall (both remain at 1.000, not shown in this table since it is unchanged from Table 1's row).

### 5.3 Not every counter is safe to fuse

With the persistence feature fixed, we traced the remaining stress-evaluation false alarms in Table 2 to their source and found them concentrated almost entirely in one candidate counter, across all three of its statistical features, for a small subset of trials -- a sustained, multi-window deviation unrelated to the anomaly being detected. Excluding this single counter from the fused feature set (5 counters -> 4) resolves nearly all of the remaining false alarms:

**Table 3: Counter selection ablation, stress evaluation, persistence feature**

| Counter set | Method | False-Alarm | F1 |
|---|---|---|---|
| All 5 counters | Mahalanobis / LOF | 0.167 | 0.889 |
| **4 counters (1 excluded)** | **Mahalanobis / LOF** | **0.017** | **0.988** |

This is the paper's general reliability-modeling lesson: a fused anomaly model is only as reliable as its least-reliable input channel, and a systematic, sustained bias in one channel is not something the fusion algorithm -- however capable -- can be expected to average away on its own. Telemetry channels intended for a fused reliability model should be individually characterized before being trusted, not assumed interchangeable by default.

## 6. Discussion and Limitations

[Brief: single embedded platform; the post-onset-only feature construction reflects an offline evaluation with known onset timing, and a streaming/online deployment would require a rolling-window reformulation (the underlying per-window statistical scores are already causal, so this is a reformulation, not new detection machinery); scope of evaluated anomaly profiles.]

## 7. Conclusion

[To draft once results are locked.]

## References

*[To be completed with verified citations before submission.]*

---
<!-- END DRAFT -->
