//! The validation gate.
//!
//! The simulator earns belief by reproducing what the fleet actually did, and
//! loses it otherwise. Nothing here trains or tunes; it only reports whether
//! the simulator is yet worth training against.
//!
//! The targets are the fleet's measured per-race ratios under
//! `ratio-picker.py`: seedit4me a median around 1.1–1.5, pulsed around 0.65,
//! home 0.05–0.08. They are intervals, not points, because they were measured
//! over days in which the boxes' conditions moved.

use crate::env::{median, Sim};
use std::collections::BTreeMap;

/// One box's target interval for its median race ratio.
#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub lo: f64,
    pub hi: f64,
}

/// The fleet's measured medians.
pub fn fleet_targets() -> BTreeMap<&'static str, Target> {
    let mut m = BTreeMap::new();
    m.insert("seedit4me", Target { lo: 1.1, hi: 1.5 });
    m.insert("pulsed", Target { lo: 0.4, hi: 0.9 });
    m.insert("home", Target { lo: 0.02, hi: 0.15 });
    m
}

#[derive(Debug, Clone)]
pub struct BoxReport {
    pub box_name: String,
    pub races: usize,
    pub median_ratio: f64,
    pub target: Option<Target>,
    pub passed: bool,
    /// Why it did not pass, when it did not.
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub boxes: Vec<BoxReport>,
    /// A box with no races at all cannot be judged, and a report in which no
    /// box raced is a failure of the rollout, not a pass.
    pub judged: usize,
}

impl Report {
    /// The gate: every box that raced must land inside its interval, and at
    /// least one box must have raced.
    pub fn passed(&self) -> bool {
        self.judged > 0 && self.boxes.iter().all(|b| b.passed)
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for b in &self.boxes {
            let t = match b.target {
                Some(t) => format!("target {:.2}–{:.2}", t.lo, t.hi),
                None => "no target".to_string(),
            };
            s.push_str(&format!(
                "{:<11} {:>5} races  median {:>6.3}  {}  {}{}\n",
                b.box_name,
                b.races,
                b.median_ratio,
                t,
                if b.passed { "pass" } else { "FAIL" },
                if b.note.is_empty() { String::new() } else { format!("  — {}", b.note) }
            ));
        }
        s.push_str(if self.passed() {
            "gate: PASS — the simulator reproduces the fleet's per-box ratios\n"
        } else {
            "gate: FAIL — do not train against this calibration\n"
        });
        s
    }
}

/// The minimum races a box needs before its median means anything.
pub const MIN_RACES: usize = 20;

pub fn check(sim: &Sim) -> Report {
    let targets = fleet_targets();
    let ratios = sim.race_ratios();
    let mut boxes = Vec::new();
    let mut judged = 0;
    for b in &sim.boxes {
        let rs = ratios.get(&b.name).cloned().unwrap_or_default();
        let target = targets.get(b.name.as_str()).copied();
        let med = median(rs.clone());
        let (passed, note) = if rs.len() < MIN_RACES {
            // Not enough to judge. This is not a pass: a rollout that raced
            // nothing has told us nothing, and calling it a pass is how a
            // broken simulator gets believed.
            (false, format!("fewer than {MIN_RACES} races — not judgeable"))
        } else if let Some(t) = target {
            judged += 1;
            if med >= t.lo && med <= t.hi {
                (true, String::new())
            } else {
                (false, format!("median {med:.3} outside {:.2}–{:.2}", t.lo, t.hi))
            }
        } else {
            (false, "no measured target for this box".to_string())
        };
        boxes.push(BoxReport {
            box_name: b.name.clone(),
            races: rs.len(),
            median_ratio: med,
            target,
            passed,
            note,
        });
    }
    Report { boxes, judged }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_with_no_races_fails() {
        let r = Report { boxes: Vec::new(), judged: 0 };
        assert!(!r.passed(), "an empty rollout must not pass the gate");
    }

    #[test]
    fn the_targets_keep_the_boxes_in_their_measured_order() {
        let t = fleet_targets();
        assert!(t["seedit4me"].lo > t["pulsed"].hi);
        assert!(t["pulsed"].lo > t["home"].hi);
    }

    #[test]
    fn rendering_says_fail_when_it_failed() {
        let r = Report {
            boxes: vec![BoxReport {
                box_name: "pulsed".into(),
                races: 50,
                median_ratio: 9.9,
                target: Some(Target { lo: 0.4, hi: 0.9 }),
                passed: false,
                note: "median 9.900 outside 0.40–0.90".into(),
            }],
            judged: 1,
        };
        let s = r.render();
        assert!(s.contains("FAIL"));
        assert!(s.contains("do not train"));
    }
}
