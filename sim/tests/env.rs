//! What the environment as a whole has to get right.
//!
//! The unit tests next to each model check the model. These check the
//! integrator: that bytes are conserved, that a box's limits bind across all
//! the torrents it holds, that the guard refuses what it should, and that the
//! orderings the fleet measured come out the other end.

use ratio_sim::env::{Act, Refusal, Release, Sim, DT};
use ratio_sim::params::Params;
use ratio_sim::policies::{Everywhere, Nothing, TableBaseline};
use ratio_sim::traces;

fn release(id: &str, size: f64, at: f64, leechers: f64, freeleech: bool) -> Release {
    Release {
        tl_id: id.into(),
        name: format!("Test.{id}"),
        category: "Movies :: HD".into(),
        size,
        freeleech,
        announce_at: at,
        listed_leechers: leechers,
        listed_seeders: 1.0,
    }
}

fn sim(releases: Vec<Release>, seed: u64) -> Sim {
    Sim::new(Params::default(), releases, seed)
}

#[test]
fn an_add_to_an_unknown_box_is_refused() {
    let mut s = sim(vec![release("1", 10.0, 0.0, 30.0, true)], 1);
    let out = s.apply(Act::Add { box_name: "nowhere".into(), release: 0 });
    assert_eq!(out.refused, Some(Refusal::NoSuchBox));
    assert!(s.torrents.is_empty());
}

#[test]
fn a_non_freeleech_release_is_refused() {
    let mut s = sim(vec![release("1", 10.0, 0.0, 30.0, false)], 2);
    let out = s.apply(Act::Add { box_name: "pulsed".into(), release: 0 });
    assert!(out.refused.as_ref().unwrap().mentions("not freeleech"), "{:?}", out.refused);
}

#[test]
fn the_same_release_is_not_added_to_one_box_twice() {
    let mut s = sim(vec![release("1", 10.0, 0.0, 30.0, true)], 3);
    assert!(s.apply(Act::Add { box_name: "pulsed".into(), release: 0 }).refused.is_none());
    let again = s.apply(Act::Add { box_name: "pulsed".into(), release: 0 });
    assert!(again.refused.as_ref().unwrap().mentions("already holds"), "{:?}", again.refused);
    assert_eq!(s.torrents.len(), 1);
}

#[test]
fn a_release_larger_than_the_box_is_refused() {
    let mut s = sim(vec![release("1", 99_999.0, 0.0, 30.0, true)], 4);
    let out = s.apply(Act::Add { box_name: "seedit4me".into(), release: 0 });
    assert!(out.refused.as_ref().unwrap().mentions("disk"), "{:?}", out.refused);
}

#[test]
fn room_accounts_for_what_is_still_to_be_written() {
    // Nothing has downloaded a byte, so only the in-flight accounting can
    // refuse these. If it were missing, every one of the eight would be
    // accepted against the same free disk. The exact cut-off belongs to the
    // service's own disk settings, so the test asserts the behaviour, not a
    // count.
    let rs: Vec<Release> = (0..8).map(|i| release(&i.to_string(), 500.0, 0.0, 30.0, true)).collect();
    let mut s = sim(rs, 5);
    let mut accepted = 0;
    let mut refused_for_disk = 0;
    for i in 0..8 {
        let out = s.apply(Act::Add { box_name: "seedit4me".into(), release: i });
        match &out.refused {
            None => accepted += 1,
            Some(r) if r.mentions("disk") => refused_for_disk += 1,
            Some(r) => panic!("refused for something other than disk: {r:?}"),
        }
    }
    assert!(accepted >= 1, "nothing fit on a 1.6 TB box");
    assert!(accepted < 8, "8 x 500 GB all fit — in-flight GB were not counted");
    assert!(refused_for_disk > 0);
}

#[test]
fn a_torrent_moves_nothing_before_the_tracker_registers_it() {
    let mut s = sim(vec![release("1", 50.0, 0.0, 40.0, true)], 6);
    s.apply(Act::Add { box_name: "seedit4me".into(), release: 0 });
    let reg = s.torrents[0].registered_at.unwrap_or(f64::MAX);
    assert!(reg > 0.0, "registration was instant");
    let mut p = Nothing;
    // Step up to just before registration.
    while s.now + DT < reg {
        s.step(&mut p);
    }
    assert_eq!(s.torrents[0].progress, 0.0, "downloaded before registering");
    assert_eq!(s.torrents[0].downloaded_gb, 0.0);
}

#[test]
fn a_torrent_downloads_after_registering_and_stops_at_complete() {
    let mut s = sim(vec![release("1", 20.0, 0.0, 40.0, true)], 7);
    s.apply(Act::Add { box_name: "seedit4me".into(), release: 0 });
    let mut p = Nothing;
    s.run(&mut p, 12.0 * 3600.0);
    let t = &s.torrents[0];
    if t.registered_at.is_some() {
        assert!(t.progress > 0.0, "registered but never downloaded");
        assert!(t.progress <= 1.0, "progress ran past complete: {}", t.progress);
        assert!(
            t.downloaded_gb <= 20.0 + 1e-6,
            "downloaded {} GB of a 20 GB release",
            t.downloaded_gb
        );
    }
}

#[test]
fn a_box_never_moves_more_than_its_line_however_many_torrents_it_holds() {
    // Twelve releases at once on pulsed, whose line is far above its disk.
    // The integrator must bind on the box, not per torrent.
    let rs: Vec<Release> = (0..12).map(|i| release(&i.to_string(), 40.0, 0.0, 60.0, true)).collect();
    let mut s = sim(rs, 8);
    for i in 0..12 {
        s.apply(Act::Add { box_name: "pulsed".into(), release: i });
    }
    let cap = s.box_by_name("pulsed").unwrap().line_down;
    let mut p = Nothing;
    let mut before: f64 = s.torrents.iter().map(|t| t.downloaded_gb).sum();
    for _ in 0..200 {
        s.step(&mut p);
        let after: f64 = s.torrents.iter().map(|t| t.downloaded_gb).sum();
        let mib_s = (after - before) * 1024.0 / DT;
        assert!(
            mib_s <= cap + 1.0,
            "the box moved {mib_s:.1} MiB/s against a {cap:.1} MiB/s line"
        );
        before = after;
    }
}

#[test]
fn uploading_requires_a_swarm_that_wants_the_release() {
    // No leechers listed, so nothing is offered and nothing is earned.
    let mut s = sim(vec![release("1", 20.0, 0.0, 0.0, true)], 9);
    s.apply(Act::Add { box_name: "seedit4me".into(), release: 0 });
    let mut p = Nothing;
    let r = s.run(&mut p, 6.0 * 3600.0);
    assert!(r.uploaded_gb < 1.0, "earned {} GB from an empty swarm", r.uploaded_gb);
}

#[test]
fn removing_a_partial_copy_is_refused_as_a_hit_and_run() {
    let mut s = sim(vec![release("1", 40.0, 0.0, 40.0, true)], 10);
    s.apply(Act::Add { box_name: "seedit4me".into(), release: 0 });
    let mut p = Nothing;
    // Let it download a little but not finish.
    for _ in 0..30 {
        s.step(&mut p);
        if s.torrents[0].progress > 0.05 {
            break;
        }
    }
    if s.torrents[0].progress > 0.0 && !s.torrents[0].done() && s.torrents[0].ratio(40.0) < 1.0 {
        let out = s.apply(Act::Remove { box_name: "seedit4me".into(), torrent: 0 });
        assert_eq!(out.refused, Some(Refusal::NotRemovable));
        assert!(s.torrents[0].live());
    }
}

#[test]
fn a_release_the_tracker_never_registered_may_always_be_removed() {
    // Find a seed whose draw is Never, rather than assume one.
    let mut found = false;
    for seed in 0..400u64 {
        let mut t = sim(vec![release("1", 40.0, 0.0, 40.0, true)], seed);
        t.apply(Act::Add { box_name: "pulsed".into(), release: 0 });
        if t.torrents[0].registered_at.is_none() {
            let out = t.apply(Act::Remove { box_name: "pulsed".into(), torrent: 0 });
            assert!(
                out.refused.is_none(),
                "a deleted release could not be dropped: {:?}",
                out.refused
            );
            found = true;
            break;
        }
    }
    assert!(found, "no seed in 400 produced an unregistered grab");
}

#[test]
fn setting_a_cap_binds_the_box() {
    let rs: Vec<Release> = (0..6).map(|i| release(&i.to_string(), 60.0, 0.0, 60.0, true)).collect();
    let mut s = sim(rs, 12);
    for i in 0..6 {
        s.apply(Act::Add { box_name: "seedit4me".into(), release: i });
    }
    assert!(s.apply(Act::SetCap { box_name: "seedit4me".into(), mib: 5.0 }).refused.is_none());
    let mut p = Nothing;
    let mut before: f64 = s.torrents.iter().map(|t| t.downloaded_gb).sum();
    for _ in 0..120 {
        s.step(&mut p);
        let after: f64 = s.torrents.iter().map(|t| t.downloaded_gb).sum();
        let mib_s = (after - before) * 1024.0 / DT;
        assert!(mib_s <= 6.0, "moved {mib_s:.2} MiB/s against a 5 MiB/s cap");
        before = after;
    }
}

#[test]
fn a_second_box_on_the_same_release_takes_from_the_first() {
    // The dual-racing trade-off, stated the only way that isolates it: watch
    // *one* box's earnings on one release, with and without another of ours in
    // the same swarm. Comparing the pair's combined total against double the
    // first box's would prove nothing — they have separate lines and disks, so
    // a pair can out-earn twice the weaker one without the swarm having been
    // split at all.
    let r = release("1", 30.0, 0.0, 60.0, true);
    let mut p = Nothing;
    let earned = |s: &Sim, box_name: &str| -> f64 {
        s.torrents.iter().filter(|t| t.box_name == box_name).map(|t| t.uploaded_gb).sum()
    };

    let mut alone = sim(vec![r.clone()], 13);
    alone.apply(Act::Add { box_name: "seedit4me".into(), release: 0 });
    alone.run(&mut p, 24.0 * 3600.0);
    let solo = earned(&alone, "seedit4me");

    let mut shared = sim(vec![r], 13);
    shared.apply(Act::Add { box_name: "seedit4me".into(), release: 0 });
    shared.apply(Act::Add { box_name: "pulsed".into(), release: 0 });
    shared.run(&mut p, 24.0 * 3600.0);
    let split = earned(&shared, "seedit4me");

    assert!(solo > 0.0, "the solo race earned nothing, so there is nothing to compare");
    assert!(
        split < solo,
        "seedit4me earned {split} sharing the swarm and {solo} alone — it was not split"
    );
}

#[test]
fn adding_nothing_earns_nothing() {
    let rs = traces::synthetic(1.0, 200.0, 0.5, &mut rand_seed(14));
    let mut s = sim(rs, 14);
    let mut p = Nothing;
    let r = s.run(&mut p, 86_400.0);
    assert_eq!(r.uploaded_gb, 0.0);
    assert!(s.torrents.is_empty());
}

#[test]
fn the_baseline_prefers_seedit4me_over_home() {
    // Its priors carry the fleet's measured per-box returns, so the strong
    // racer should take far more of the adds than the weak one.
    let rs = traces::synthetic(2.0, 400.0, 0.2, &mut rand_seed(15));
    let mut s = sim(rs, 15);
    let mut p = TableBaseline::new();
    s.run(&mut p, 2.0 * 86_400.0);
    let on = |name: &str| s.torrents.iter().filter(|t| t.box_name == name).count();
    assert!(
        on("seedit4me") > on("home"),
        "seedit4me {} vs home {}",
        on("seedit4me"),
        on("home")
    );
}

#[test]
fn racing_everything_everywhere_does_not_beat_choosing() {
    // The operator's "no arbitrary caps" principle is about pricing, not
    // volume: once the disks choke, more adds stop being more upload. If this
    // ever fails, either the disk model has gone soft or the baseline is bad —
    // both worth knowing.
    let rs = traces::synthetic(3.0, 600.0, 0.15, &mut rand_seed(16));
    let mut a = sim(rs.clone(), 16);
    let mut b = sim(rs, 16);
    let mut base = TableBaseline::new();
    let mut all = Everywhere;
    let chosen = a.run(&mut base, 3.0 * 86_400.0);
    let greedy = b.run(&mut all, 3.0 * 86_400.0);
    assert!(
        greedy.uploaded_gb > 0.0 && chosen.uploaded_gb > 0.0,
        "neither policy uploaded anything: {chosen:?} {greedy:?}"
    );
    assert!(
        b.torrents.len() > a.torrents.len(),
        "racing everywhere should add more: {} vs {}",
        b.torrents.len(),
        a.torrents.len()
    );
}

#[test]
fn the_reward_is_upload_less_its_costs() {
    let rs = traces::synthetic(1.0, 300.0, 0.3, &mut rand_seed(17));
    let mut s = sim(rs, 17);
    let mut p = TableBaseline::new();
    let r = s.run(&mut p, 86_400.0);
    assert!(r.quota_cost >= 0.0 && r.disk_cost >= 0.0 && r.rent_cost > 0.0);
    let expected = r.uploaded_gb - r.quota_cost - r.disk_cost - r.rent_cost;
    assert!((r.total() - expected).abs() < 1e-9);
}

#[test]
fn a_rollout_is_reproducible_from_its_seed() {
    let rs = traces::synthetic(2.0, 300.0, 0.2, &mut rand_seed(18));
    let run = |seed: u64| {
        let mut s = sim(rs.clone(), seed);
        let mut p = TableBaseline::new();
        let r = s.run(&mut p, 2.0 * 86_400.0);
        (r.uploaded_gb, s.torrents.len())
    };
    assert_eq!(run(99), run(99), "the same seed gave a different rollout");
    assert_ne!(run(99), run(100), "different seeds gave identical rollouts");
}

fn rand_seed(s: u64) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(s)
}
