//! RFC-0103 at the run: a loop `analysis::ahead` lowers gives the value,
//! the lines and the trap it gives in place, under the sequential and the
//! tokio executor, and its io spawns overlap.

use acvus_interpreter::{Executor, Lower, TokioExecutor};
use acvus_interpreter_test::ahead::{On, Outcome, Ran, SLOW_CALLS, SLOW_STEP, run};
use acvus_mir::ty::{Task, Ty};

/// The run lowered and in place, under both executors: every outcome is
/// the in-place sequential one, and the lowered runs hold `ForAhead`.
async fn same_everywhere(source: &str, ret: Ty) -> (Outcome, Ran) {
    let reference = run(source, ret.clone(), Lower::InPlace, On::Sequential).await;
    assert!(!reference.lowered, "a run prepared in place holds no `ForAhead`");
    let mut tokio_lowered = None;
    for on in [On::Sequential, On::Tokio] {
        for lower in [Lower::Ahead, Lower::InPlace] {
            let ran = run(source, ret.clone(), lower, on).await;
            assert_eq!(
                ran.outcome, reference.outcome,
                "{lower:?} on {on:?} differs from in place on the sequential executor"
            );
            if lower == Lower::Ahead {
                assert!(
                    ran.lowered,
                    "the loop runs in place, declined: {:?}",
                    ran.declined
                );
                if let On::Tokio = on {
                    tokio_lowered = Some(ran);
                }
            }
        }
    }
    (
        reference.outcome,
        tokio_lowered.expect("the tokio executor ran the lowering"),
    )
}

fn urls(n: usize) -> String {
    let texts: Vec<String> = (0..n).map(|i| format!("\"u/{i}\".to_string()")).collect();
    format!("let urls = vec([{}]);", texts.join(", "))
}

fn texts(items: &[&str]) -> String {
    let texts: Vec<String> = items
        .iter()
        .map(|text| format!("\"{text}\".to_string()"))
        .collect();
    format!("let xs = vec([{}]);", texts.join(", "))
}

fn tripled(n: usize) -> String {
    (0..n)
        .map(|i| format!(" {} ", i * 3))
        .collect::<Vec<_>>()
        .join(",")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn io_per_element_comes_back_in_index_order_though_it_completes_out_of_order() {
    let source = format!(
        "{}
         let out = vec([]);
         for u in &urls {{ out.push(ah::fetch(u.to_string())); }}
         out.into_iter().join(\",\".to_string())",
        urls(10)
    );
    let (outcome, lowered) = same_everywhere(&source, Ty::String).await;
    assert_eq!(outcome.value, Ok(serde_json::json!(tripled(10))));
    let completed = lowered.wire.completed();
    let issued: Vec<String> = (0..10).map(|i| format!("u/{i}")).collect();
    assert_ne!(
        completed, issued,
        "the calls completed in index order, so the test shows no reordering"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn heavy_per_element_does_its_work_apart() {
    let items: Vec<String> = (0..24).map(|i| (i * 7 + 3).to_string()).collect();
    let items: Vec<&str> = items.iter().map(String::as_str).collect();
    let source = format!(
        "{}
         let total = 0;
         for s in &xs {{ total = total + ah::crunch(s.to_string()).unwrap(); }}
         total",
        texts(&items)
    );
    same_everywhere(&source, Ty::I64).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn light_processing_around_the_call_is_the_rest_of_the_iteration() {
    let source = format!(
        "{}
         let out = vec([]);
         for u in &urls {{
             let request = u.to_string() + \"?page=\" + \"1\";
             let body = ah::fetch(request);
             let n = i64::from_str(body.trim()).unwrap();
             if n % 2 == 0 {{ out.push(n.to_string()); }}
         }}
         out.into_iter().join(\"+\".to_string())",
        urls(12)
    );
    let (outcome, _) = same_everywhere(&source, Ty::String).await;
    assert_eq!(outcome.value, Ok(serde_json::json!("0+6+12+18+24+30")));
}

/// `t` and the reference the concatenation after the call reads are
/// written by the prefix and read by the rest, and a later index's prefix
/// writes the same registers, so each index's pair moves through the ring.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_let_slot_the_rest_reads_is_each_index_s_own() {
    let source = format!(
        "{}
         for s in &xs {{
             let t = s.to_string() + \"0\";
             let v = ah::crunch(t.to_string()).unwrap();
             ah::emit(&(t + \":\" + v.to_string()));
         }}
         xs.len()",
        texts(&["1", "2", "3", "4", "5", "6"])
    );
    let (outcome, _) = same_everywhere(&source, Ty::Int(acvus_mir::ty::IntTy::U64)).await;
    let heads: Vec<String> = outcome
        .lines
        .iter()
        .map(|line| line.split(':').next().expect("a line holds a `:`").to_owned())
        .collect();
    assert_eq!(heads, ["10", "20", "30", "40", "50", "60"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_object_element_is_read_through_its_field() {
    let items: Vec<String> = (0..6)
        .map(|i| format!("{{ url: \"u/{i}\".to_string(), weight: {i}, }}"))
        .collect();
    let source = format!(
        "let items = vec([{}]);
         let out = vec([]);
         for item in &items {{ out.push(ah::fetch(item.url.to_string())); }}
         out.into_iter().join(\",\".to_string())",
        items.join(", ")
    );
    let (outcome, _) = same_everywhere(&source, Ty::String).await;
    assert_eq!(outcome.value, Ok(serde_json::json!(tripled(6))));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn more_elements_than_the_bound_keep_the_bound_in_flight() {
    let n = 80;
    let source = format!(
        "{}
         let out = vec([]);
         for u in &urls {{ out.push(ah::fetch(u.to_string())); }}
         out.into_iter().join(\",\".to_string())",
        urls(n)
    );
    let (outcome, lowered) = same_everywhere(&source, Ty::String).await;
    assert_eq!(outcome.value, Ok(serde_json::json!(tripled(n))));
    let bound = TokioExecutor.ahead(Task::Async).get();
    let most = lowered.wire.most_in_flight();
    assert!(
        most > 1 && most <= bound,
        "{most} io calls were in flight at once, and the bound is {bound}"
    );
    let sequential = run(&source, Ty::String, Lower::Ahead, On::Sequential).await;
    assert_eq!(
        sequential.wire.most_in_flight(),
        1,
        "the sequential executor runs each job at its evaluation"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_trap_midway_keeps_the_earlier_lines_and_stops_there() {
    let source = format!(
        "{}
         let total = 0;
         for s in &xs {{
             let v = ah::crunch(s.to_string()).unwrap();
             ah::emit(&v.to_string());
             total = total + v;
         }}
         total",
        texts(&["1", "2", "3", "x", "5", "6", "7", "8", "9"])
    );
    let (outcome, _) = same_everywhere(&source, Ty::I64).await;
    assert_eq!(outcome.lines.len(), 3, "{outcome:?}");
    assert!(outcome.value.is_err(), "{outcome:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_job_that_traps_surfaces_at_its_own_evaluation() {
    let source = format!(
        "{}
         let total = 0;
         for s in &xs {{
             let v = ah::crunch(s.to_string()).unwrap();
             ah::emit(&v.to_string());
             total = total + v;
         }}
         total",
        texts(&["1", "2", "boom", "4", "5", "boom", "7"])
    );
    let (outcome, _) = same_everywhere(&source, Ty::I64).await;
    assert_eq!(outcome.lines.len(), 2, "{outcome:?}");
    let Err(message) = &outcome.value else {
        panic!("the run finished: {outcome:?}")
    };
    assert!(message.contains("crunch: the job traps"), "{message}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn effectful_io_under_anyorder_runs_ahead() {
    let source = format!(
        "{}
         let out = vec([]);
         anyorder {{ for u in &urls {{ out.push(ah::post(u.to_string())); }} }}
         out.into_iter().join(\",\".to_string())",
        urls(10)
    );
    let (outcome, _) = same_everywhere(&source, Ty::String).await;
    assert_eq!(outcome.value, Ok(serde_json::json!(tripled(10))));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_break_after_the_call_leaves_the_work_issued_past_it() {
    let items: Vec<String> = (0..30).map(|i| i.to_string()).collect();
    let items: Vec<&str> = items.iter().map(String::as_str).collect();
    let source = format!(
        "{}
         let total = 0;
         for s in &xs {{
             let v = ah::crunch(s.to_string()).unwrap();
             if v > 6000 {{ break; }}
             total = total + v;
         }}
         total",
        texts(&items)
    );
    let (outcome, _) = same_everywhere(&source, Ty::I64).await;
    assert!(outcome.value.is_ok(), "{outcome:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_range_source_lays_its_counter_at_each_index() {
    let source = "let total = 0;
                  for i in 0..40 { total = total + ah::weigh(i * 3 + 1); }
                  total";
    same_everywhere(source, Ty::I64).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_element_lent_to_a_heavy_call_runs_ahead() {
    let items: Vec<String> = (0..16).map(|i| (i * 5).to_string()).collect();
    let items: Vec<&str> = items.iter().map(String::as_str).collect();
    let source = format!(
        "{}
         let total = 0;
         for s in &xs {{ total = total + ah::crunch_ref(s).unwrap(); }}
         total",
        texts(&items)
    );
    same_everywhere(&source, Ty::I64).await;
}

/// The two loans `prepare` refuses to run ahead: one the prefix makes, which
/// a later index's prefix would overwrite under the job reading it, and one
/// held past an exit the loop can take.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_loan_the_next_prefix_overwrites_or_an_exit_outlives_runs_in_place() {
    let cases = [
        (
            "let total = 0;
             for s in &xs { let t = s.to_string() + \"0\"; total = total + ah::crunch_ref(&t).unwrap(); }
             total",
            "LendsPrefix",
        ),
        (
            "let total = 0;
             for s in &xs { let v = ah::crunch_ref(s).unwrap(); if v > 30000 { break; } total = total + v; }
             total",
            "LoanPastExit",
        ),
    ];
    for (body, why) in cases {
        let source = format!("{}\n{body}", texts(&["1", "2", "30", "4", "50"]));
        let reference = run(&source, Ty::I64, Lower::InPlace, On::Sequential).await;
        for on in [On::Sequential, On::Tokio] {
            let ran = run(&source, Ty::I64, Lower::Ahead, on).await;
            assert!(!ran.lowered, "{why}: the loop is lowered");
            assert_eq!(ran.declined.len(), 1, "{why}: {:?}", ran.declined);
            assert!(
                format!("{:?}", ran.declined[0].why).starts_with(why),
                "{why}: {:?}",
                ran.declined
            );
            assert_eq!(ran.outcome, reference.outcome, "{why} on {on:?}");
        }
    }
}

/// In place the calls wait one after another, and lowered ahead they
/// overlap, so the run takes well under the sum of the waits. The bound is
/// half the sum, where a full overlap is near the longest single wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn io_calls_lowered_ahead_overlap() {
    let source = format!(
        "{}
         let out = vec([]);
         for u in &urls {{ out.push(ah::fetch_slow(u.to_string())); }}
         out.len()",
        urls(usize::try_from(SLOW_CALLS).expect("eight calls fit a usize"))
    );
    let waits: u32 = (0..SLOW_CALLS).map(|i| 5 - i % 5).sum();
    let sum = SLOW_STEP * waits;
    let in_place = run(&source, Ty::Int(acvus_mir::ty::IntTy::U64), Lower::InPlace, On::Tokio).await;
    let ahead = run(&source, Ty::Int(acvus_mir::ty::IntTy::U64), Lower::Ahead, On::Tokio).await;
    assert!(ahead.lowered, "declined: {:?}", ahead.declined);
    assert_eq!(ahead.outcome, in_place.outcome);
    assert!(
        in_place.took >= sum,
        "in place took {:?}, under the {sum:?} its waits add to",
        in_place.took
    );
    assert!(
        ahead.took < sum / 2,
        "lowered ahead took {:?}, not under half the {sum:?} the waits add to",
        ahead.took
    );
}

/// Without `anyorder`, an effectful io call keeps its order: the stage
/// table puts its spawn in the ordered stage, `analysis::ahead` runs the
/// loop in place, and `prepare` has nothing to decline.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn effectful_io_in_order_runs_in_place() {
    let source = format!(
        "{}
         let out = vec([]);
         for u in &urls {{ out.push(ah::post(u.to_string())); }}
         out.into_iter().join(\",\".to_string())",
        urls(6)
    );
    let reference = run(&source, Ty::String, Lower::InPlace, On::Sequential).await;
    for on in [On::Sequential, On::Tokio] {
        let ran = run(&source, Ty::String, Lower::Ahead, on).await;
        assert!(!ran.lowered, "an ordered io loop is lowered");
        assert!(ran.declined.is_empty(), "{:?}", ran.declined);
        assert_eq!(ran.outcome, reference.outcome);
        assert_eq!(ran.wire.most_in_flight(), 1, "ordered io calls overlap");
    }
}

/// The prefix holds an `if` whose arms rejoin before the call: the region
/// runs ahead as one operation, and its join's parameter is the argument.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_branch_before_the_call_runs_ahead_with_it() {
    let source = "let total = 0;
                  for i in 0..24 {
                      let x = if i % 3 == 0 { i * 5 } else { i + 7 };
                      total = total + ah::weigh(x);
                  }
                  total";
    same_everywhere(source, Ty::Int(acvus_mir::ty::IntTy::I64)).await;
}
