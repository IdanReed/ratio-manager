//! Online reinforcement learning against the simulator.
//!
//! The third arm of the model comparison, and the one that needs no logged
//! data at all: a policy trained by acting in the simulator and being paid in
//! its reward. The other two arms learn from the fleet's history — the ranker
//! by regression, the decision transformer by sequence modelling — and are
//! therefore bounded by what the behaviour policy already did. This one is
//! not, which is its whole point and also its whole risk: it is only ever as
//! good as the simulator, so it must not be believed until the simulator
//! passes [`crate::validate`].
//!
//! Deliberately a **linear softmax policy trained by REINFORCE with a
//! baseline**, not a deep network. Three reasons, in order:
//!
//!  1. At this problem size the decision is "which box, or none", over a
//!     handful of features. A linear policy can express the whole of what the
//!     bucketed-median baseline expresses and more.
//!  2. Its weights can be read. When it does something odd, the reason is a
//!     number in a named slot rather than an afternoon with a debugger.
//!  3. It trains in seconds on a CPU, so it can run in the test suite. An
//!     RL arm nobody can re-run is an RL arm nobody will re-run.
//!
//! A deeper policy is the obvious next step once the simulator is calibrated
//! and there is something worth fitting. Until then this establishes whether
//! training against the simulator works at all, which is the question that
//! actually blocks the others.

use crate::env::{Act, Policy, Sim};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Features of one (release, box) pair, at decision time only.
///
/// Kept small and shared with the Python side's intent: anything measured
/// after the decision is a label, not a feature.
pub const N_FEATURES: usize = 8;

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "bias",
    "log_size_gb",
    "demand",
    "placeable_tb",
    "downloading",
    "box_is_home",
    "box_is_seedit4me",
    "box_is_pulsed",
];

/// Features, each scaled to roughly `[0, 1]` over its observed range.
///
/// The scaling is not cosmetic. Unscaled, `demand` runs to 60 and
/// `placeable` to tens of terabytes, so a weight of order one saturates the
/// softmax and the policy goes deterministic within a handful of episodes —
/// its mean propensity hits 1.00 and it stops learning anything. The
/// propensity log is what surfaces that, and `exploration_does_not_collapse`
/// is what keeps it fixed.
fn features(sim: &Sim, bx: &str, release: usize) -> [f64; N_FEATURES] {
    let r = &sim.releases[release];
    let demand = r.listed_leechers / (r.listed_seeders + 1.0);
    let (placeable_tb, downloading) = match sim.box_by_name(bx) {
        Some(b) => (
            (b.free_gb(sim.reserve_gb) - sim.in_flight_gb(bx)) / 1024.0,
            sim.held(bx).filter(|t| !t.done()).count() as f64,
        ),
        None => (0.0, 0.0),
    };
    [
        1.0,
        // Releases run to ~100 GB, so ln(1+size) tops out near 4.6.
        (1.0 + r.size.max(0.0)).ln() / 5.0,
        // Demand past 10 leechers per seeder is already "a lot".
        (demand / 10.0).min(3.0),
        // Beyond a few TB of headroom the exact figure stops mattering.
        (placeable_tb / 5.0).clamp(-1.0, 3.0),
        (downloading / 10.0).min(3.0),
        if bx == "home" { 1.0 } else { 0.0 },
        if bx == "seedit4me" { 1.0 } else { 0.0 },
        if bx == "pulsed" { 1.0 } else { 0.0 },
    ]
}

/// One decision the policy made, kept until its return is known.
#[derive(Debug, Clone)]
struct Step {
    /// Features of each option, in order; the last is "pass".
    options: Vec<[f64; N_FEATURES]>,
    chosen: usize,
    /// The probability the policy gave the action it took. Logged for the same
    /// reason the live bandit logs it: without it, no off-policy estimate of
    /// this policy's value is possible.
    probability: f64,
    /// Which torrent the add created, so its earnings can be found later.
    torrent: Option<usize>,
}

/// A linear softmax policy over box choices.
#[derive(Debug, Clone)]
pub struct LinearPolicy {
    pub weights: Vec<f64>,
    /// Softmax temperature. Above 1 explores more.
    pub temperature: f64,
    /// Whether to sample (training) or take the argmax (evaluation).
    pub explore: bool,
    rng: StdRng,
    steps: Vec<Step>,
    label: String,
}

impl LinearPolicy {
    pub fn new(seed: u64) -> Self {
        Self {
            weights: vec![0.0; N_FEATURES],
            temperature: 1.0,
            explore: true,
            rng: StdRng::seed_from_u64(seed),
            steps: Vec::new(),
            label: "reinforce".to_string(),
        }
    }

    /// Weights by name, largest magnitude first. The reason to keep it linear.
    pub fn weight_table(&self) -> Vec<(&'static str, f64)> {
        let mut v: Vec<(&'static str, f64)> =
            FEATURE_NAMES.iter().copied().zip(self.weights.iter().copied()).collect();
        v.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        v
    }

    fn score(&self, f: &[f64; N_FEATURES]) -> f64 {
        f.iter().zip(self.weights.iter()).map(|(a, b)| a * b).sum()
    }

    /// Softmax over the options. The last option is always "pass", scored 0,
    /// so the policy can decline without a separate mechanism.
    fn probabilities(&self, options: &[[f64; N_FEATURES]]) -> Vec<f64> {
        let mut logits: Vec<f64> = options.iter().map(|f| self.score(f) / self.temperature).collect();
        logits.push(0.0); // pass
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        exp.into_iter().map(|e| e / sum).collect()
    }

    /// REINFORCE with a mean-return baseline, using each add's realized
    /// earnings as its return.
    ///
    /// The baseline is the episode's mean return. Without it the gradient
    /// pushes up every action that earned anything at all, which on this
    /// reward (always non-negative) means it pushes up everything and the
    /// policy drifts toward adding indiscriminately.
    pub fn update(&mut self, sim: &Sim, lr: f64) {
        if self.steps.is_empty() {
            return;
        }
        // Return per step: what the add it made went on to earn, normalized by
        // the release's size so a big release is not automatically a good
        // decision. A pass earns nothing, which is the correct baseline for it.
        let returns: Vec<f64> = self
            .steps
            .iter()
            .map(|s| match s.torrent {
                Some(id) => {
                    let t = &sim.torrents[id];
                    let size = sim.releases[t.release].size.max(1e-9);
                    t.uploaded_gb / size
                }
                None => 0.0,
            })
            .collect();
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let sd = var.sqrt().max(1e-6);

        let mut grad = vec![0.0; N_FEATURES];
        for (s, ret) in self.steps.iter().zip(returns.iter()) {
            let advantage = (ret - mean) / sd;
            let probs = self.probabilities(&s.options);
            // d/dw log softmax(chosen) = f_chosen - sum_i p_i f_i, with the
            // pass option contributing no features.
            let mut expected = vec![0.0; N_FEATURES];
            for (i, f) in s.options.iter().enumerate() {
                for k in 0..N_FEATURES {
                    expected[k] += probs[i] * f[k];
                }
            }
            if s.chosen < s.options.len() {
                for k in 0..N_FEATURES {
                    grad[k] += advantage * (s.options[s.chosen][k] - expected[k]);
                }
            } else {
                // Chose to pass: push down whatever made the adds attractive.
                for k in 0..N_FEATURES {
                    grad[k] += advantage * (0.0 - expected[k]);
                }
            }
        }
        for k in 0..N_FEATURES {
            self.weights[k] += lr * grad[k] / n;
            // Keep the weights finite under a bad reward scale rather than
            // letting one episode blow the policy up.
            self.weights[k] = self.weights[k].clamp(-20.0, 20.0);
        }
        self.steps.clear();
    }

    /// Forget the episode's steps without learning from them.
    pub fn reset(&mut self) {
        self.steps.clear();
    }

    /// The probability the policy gave each action it actually took this
    /// episode.
    ///
    /// This is the propensity log, and it is not a diagnostic nicety: without
    /// it no off-policy estimate of this policy's value is possible, and the
    /// live bandit has the same requirement for the same reason. Its mean also
    /// says how deterministic the policy has become — a mean near 1 means
    /// exploration has stopped, whether or not that was intended.
    pub fn propensities(&self) -> Vec<f64> {
        self.steps.iter().map(|s| s.probability).collect()
    }

    /// Mean propensity over the episode, or `None` if it took no decisions.
    pub fn mean_propensity(&self) -> Option<f64> {
        let p = self.propensities();
        if p.is_empty() {
            None
        } else {
            Some(p.iter().sum::<f64>() / p.len() as f64)
        }
    }

    /// The probability the policy would give each option in this state, for
    /// inspection and for logging a propensity.
    pub fn inspect(&self, sim: &Sim, release: usize) -> Vec<(String, f64)> {
        let boxes: Vec<String> = sim.boxes.iter().map(|b| b.name.clone()).collect();
        let options: Vec<[f64; N_FEATURES]> =
            boxes.iter().map(|b| features(sim, b, release)).collect();
        let probs = self.probabilities(&options);
        boxes
            .into_iter()
            .chain(std::iter::once("pass".to_string()))
            .zip(probs)
            .collect()
    }
}

impl Policy for LinearPolicy {
    fn on_announce(&mut self, sim: &Sim, release: usize) -> Vec<Act> {
        let r = &sim.releases[release];
        if !r.freeleech {
            return Vec::new();
        }
        // Only boxes that could actually take it are options; offering the
        // policy actions the guard will refuse teaches it nothing.
        let boxes: Vec<String> = sim
            .boxes
            .iter()
            .filter(|b| !b.spent() && sim.has_room(&b.name, r.size))
            .map(|b| b.name.clone())
            .collect();
        if boxes.is_empty() {
            return Vec::new();
        }
        let options: Vec<[f64; N_FEATURES]> =
            boxes.iter().map(|b| features(sim, b, release)).collect();
        let probs = self.probabilities(&options);

        let chosen = if self.explore {
            let x: f64 = self.rng.gen();
            let mut acc = 0.0;
            let mut pick = probs.len() - 1;
            for (i, p) in probs.iter().enumerate() {
                acc += p;
                if x <= acc {
                    pick = i;
                    break;
                }
            }
            pick
        } else {
            probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(probs.len() - 1)
        };

        self.steps.push(Step {
            options,
            chosen,
            probability: probs[chosen],
            // Filled in below if the add is accepted.
            torrent: None,
        });

        if chosen >= boxes.len() {
            return Vec::new(); // passed
        }
        // The torrent id the add will take, if the guard allows it. `apply`
        // pushes onto `sim.torrents`, so the next index is it.
        let next_id = sim.torrents.len();
        if let Some(s) = self.steps.last_mut() {
            s.torrent = Some(next_id);
        }
        vec![Act::Add { box_name: boxes[chosen].clone(), release }]
    }

    fn on_tick(&mut self, sim: &Sim) -> Vec<Act> {
        // The same eviction rule as every other policy here: the comparison is
        // about which adds get made.
        let mut acts = Vec::new();
        for t in sim.torrents.iter().filter(|t| t.live()) {
            let size = sim.releases[t.release].size;
            let unregistered = t.registered_at.is_none() && sim.now - t.added_at > 3600.0;
            if t.ratio(size) >= 1.0 || unregistered {
                acts.push(Act::Remove { box_name: t.box_name.clone(), torrent: t.id });
            }
        }
        acts
    }

    fn name(&self) -> &str {
        &self.label
    }
}

/// How a training run went, so a result can be read rather than trusted.
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// Mean net reward per episode, in training order.
    pub curve: Vec<f64>,
    pub weights: Vec<(&'static str, f64)>,
    /// Mean propensity of the last episode: how deterministic the policy has
    /// become. Near 1 means exploration has stopped.
    pub mean_propensity: Option<f64>,
}

impl TrainReport {
    /// Mean of the first and last fifth of the curve. Comparing those two is
    /// the only claim worth making from a short REINFORCE run: that it moved
    /// in the right direction, not that it converged.
    pub fn improvement(&self) -> (f64, f64) {
        if self.curve.is_empty() {
            return (0.0, 0.0);
        }
        let k = (self.curve.len() / 5).max(1);
        let first: f64 = self.curve[..k].iter().sum::<f64>() / k as f64;
        let last: f64 = self.curve[self.curve.len() - k..].iter().sum::<f64>() / k as f64;
        (first, last)
    }

    pub fn render(&self) -> String {
        let (a, b) = self.improvement();
        let mut s = format!(
            "trained {} episodes: first fifth {a:.1} -> last fifth {b:.1} net{}\n",
            self.curve.len(),
            match self.mean_propensity {
                Some(p) => format!(" (mean propensity {p:.2})"),
                None => String::new(),
            }
        );
        s.push_str("  weights:\n");
        for (n, w) in &self.weights {
            s.push_str(&format!("    {n:<18} {w:>8.3}\n"));
        }
        s
    }
}

/// Train a linear policy against the simulator.
///
/// `make_sim` builds a fresh episode per iteration; varying its releases and
/// its parameter draw across episodes is what keeps the policy from fitting
/// one week's artefacts.
pub fn train<F>(
    policy: &mut LinearPolicy,
    episodes: usize,
    until: f64,
    lr: f64,
    mut make_sim: F,
) -> TrainReport
where
    F: FnMut(u64) -> Sim,
{
    let mut curve = Vec::with_capacity(episodes);
    let mut mean_propensity = None;
    policy.explore = true;
    for e in 0..episodes {
        let mut sim = make_sim(e as u64);
        policy.reset();
        let r = sim.run(policy, until);
        curve.push(r.total());
        // Read before the update: it consumes the episode's steps.
        mean_propensity = policy.mean_propensity();
        policy.update(&sim, lr);
    }
    TrainReport { curve, weights: policy.weight_table(), mean_propensity }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Release;
    use crate::params::Params;
    use crate::traces;

    fn episode(seed: u64) -> Sim {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xbeef);
        let releases: Vec<Release> = traces::synthetic(1.0, 300.0, 0.25, &mut rng);
        Sim::new(Params::default(), releases, seed)
    }

    #[test]
    fn probabilities_are_a_distribution_including_the_pass_option() {
        let p = LinearPolicy::new(1);
        let opts = vec![[1.0; N_FEATURES], [0.5; N_FEATURES]];
        let probs = p.probabilities(&opts);
        assert_eq!(probs.len(), 3, "two boxes plus pass");
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum {sum}");
        assert!(probs.iter().all(|x| *x >= 0.0));
    }

    #[test]
    fn at_zero_weights_every_option_is_equally_likely() {
        let p = LinearPolicy::new(2);
        let opts = vec![[1.0; N_FEATURES], [9.0; N_FEATURES], [-3.0; N_FEATURES]];
        let probs = p.probabilities(&opts);
        for w in probs.windows(2) {
            assert!((w[0] - w[1]).abs() < 1e-9, "{probs:?}");
        }
    }

    #[test]
    fn the_feature_names_match_the_vector() {
        assert_eq!(FEATURE_NAMES.len(), N_FEATURES);
        let p = LinearPolicy::new(3);
        assert_eq!(p.weights.len(), N_FEATURES);
        assert_eq!(p.weight_table().len(), N_FEATURES);
    }

    #[test]
    fn an_update_with_no_steps_is_a_no_op() {
        let mut p = LinearPolicy::new(4);
        let sim = episode(0);
        let before = p.weights.clone();
        p.update(&sim, 0.1);
        assert_eq!(before, p.weights);
    }

    #[test]
    fn training_moves_the_weights_and_does_not_blow_them_up() {
        let mut p = LinearPolicy::new(5);
        let report = train(&mut p, 12, 86_400.0, 0.2, episode);
        assert_eq!(report.curve.len(), 12);
        assert!(
            p.weights.iter().any(|w| w.abs() > 1e-6),
            "no weight moved: {:?}",
            p.weights
        );
        assert!(
            p.weights.iter().all(|w| w.is_finite() && w.abs() <= 20.0),
            "weights diverged: {:?}",
            p.weights
        );
    }

    #[test]
    fn a_trained_policy_beats_adding_nothing() {
        // The weakest claim worth making, and the one that says training
        // against the simulator works at all. Beating the fleet's own baseline
        // is a question for a calibrated simulator, not this test.
        let mut p = LinearPolicy::new(6);
        train(&mut p, 15, 86_400.0, 0.2, episode);
        p.explore = false;
        let mut sim = episode(99);
        let r = sim.run(&mut p, 86_400.0);
        assert!(r.uploaded_gb > 0.0, "the trained policy never added anything");

        let mut nothing = crate::policies::Nothing;
        let mut sim2 = episode(99);
        let r2 = sim2.run(&mut nothing, 86_400.0);
        assert!(
            r.total() > r2.total(),
            "trained {:.1} did not beat doing nothing {:.1}",
            r.total(),
            r2.total()
        );
    }

    #[test]
    fn exploration_does_not_collapse() {
        // The bug the propensity log caught: unscaled features saturate the
        // softmax, the policy goes deterministic within a few episodes, and
        // it stops learning while still reporting progress. A mean propensity
        // at 1.00 is the signature.
        let mut p = LinearPolicy::new(20);
        let report = train(&mut p, 25, 86_400.0, 0.2, episode);
        let mp = report.mean_propensity.expect("the policy took no decisions");
        assert!(
            mp < 0.98,
            "mean propensity {mp:.3}: exploration collapsed, so training stopped early"
        );
    }

    #[test]
    fn evaluation_is_deterministic_once_exploration_is_off() {
        let mut p = LinearPolicy::new(7);
        train(&mut p, 6, 43_200.0, 0.2, episode);
        p.explore = false;
        let run = |p: &mut LinearPolicy| {
            let mut sim = episode(123);
            sim.run(p, 43_200.0).uploaded_gb
        };
        let a = run(&mut p);
        let b = run(&mut p);
        assert_eq!(a, b, "the greedy policy was not reproducible");
    }

    #[test]
    fn inspect_reports_a_probability_per_box_plus_pass() {
        let p = LinearPolicy::new(8);
        let sim = episode(1);
        if sim.releases.is_empty() {
            return;
        }
        let rows = p.inspect(&sim, 0);
        assert_eq!(rows.len(), sim.boxes.len() + 1);
        assert_eq!(rows.last().unwrap().0, "pass");
        let sum: f64 = rows.iter().map(|r| r.1).sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_logged_propensity_is_never_zero() {
        // Same requirement as the live bandit's: a zero propensity makes every
        // off-policy estimate that uses it wrong.
        let mut p = LinearPolicy::new(9);
        let mut sim = episode(2);
        sim.run(&mut p, 43_200.0);
        assert!(!p.steps.is_empty(), "no decision was taken");
        for s in &p.steps {
            assert!(s.probability > 0.0 && s.probability <= 1.0, "{}", s.probability);
        }
    }
}
