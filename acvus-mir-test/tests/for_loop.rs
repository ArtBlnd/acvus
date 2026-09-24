//! A `for` loop is a terminator (RFC-0057): the four heads it admits, what
//! the header holds, and what it refuses.

use acvus_mir::analysis::domtree::DomTree;
use acvus_mir::analysis::loops::{for_headers, natural_loops_innermost_first};
use acvus_mir::cfg::{BlockIdx, Terminator, promote};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::analysis::loop_deps::{LoopDeps, Order, Storage, Token};
use acvus_mir_test::{
    compile_script_at, compile_script_ir, compile_script_optimized,
    lowered_script_module,
};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ir(source: &str) -> String {
    let i = Interner::new();
    compile_script_ir(&i, source, &FxHashMap::default()).unwrap_or_else(|e| panic!("{source}\n{e}"))
}

fn optimized(source: &str) -> String {
    let i = Interner::new();
    compile_script_optimized(&i, source, &FxHashMap::default())
        .unwrap_or_else(|e| panic!("{source}\n{e}"))
}

fn refusals(source: &str) -> Vec<String> {
    let i = Interner::new();
    compile_script_optimized(&i, source, &FxHashMap::default())
        .expect_err(&format!("{source} is refused"))
        .lines()
        .map(|line| line.split("] ").nth(1).unwrap_or(line).to_string())
        .collect()
}

fn refusal(source: &str) -> String {
    let found = refusals(source);
    assert!(!found.is_empty(), "{source} is refused");
    found.into_iter().next().expect("one refusal")
}

fn main_body(ir: &str) -> &str {
    ir.split("=== main ===")
        .nth(1)
        .expect("an entry body")
        .split("=== ")
        .next()
        .expect("the entry body ends at the next section")
}

fn count(body: &str, needle: &str) -> usize {
    body.matches(needle).count()
}

fn label_of(edge: &str) -> &str {
    edge.trim().split(['(', ' ', ':']).next().expect("a label")
}

// -- The four heads ---------------------------------------------------

#[test]
fn a_shared_head_is_one_as_slice_and_a_slice_terminator() {
    let ir = ir("let v = vec([1, 2, 3]); let acc = 0; for x in &v { acc = acc + *x; } acc");
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice"), 1, "{body}");
    assert_eq!(count(body, "for slice("), 1, "{body}");
    assert_eq!(count(body, "for slice_mut("), 0, "{body}");
}

#[test]
fn a_mutable_head_is_a_slice_mut_terminator() {
    let ir = ir("let v = vec([1, 2, 3]); for x in &mut v { *x = 0; } 0");
    let body = main_body(&ir);
    assert_eq!(count(body, "for slice_mut("), 1, "{body}");
}

#[test]
fn an_array_by_value_is_an_array_terminator() {
    let ir = ir("let a = [1, 2, 3]; let acc = 0; for x in a { acc = acc + x; } acc");
    let body = main_body(&ir);
    assert_eq!(count(body, "for array("), 1, "{body}");
}

#[test]
fn an_array_behind_a_reference_is_a_slice_terminator_too() {
    let ir = ir("let a = [1, 2, 3]; let acc = 0; for x in &a { acc = acc + *x; } acc");
    let body = main_body(&ir);
    assert_eq!(count(body, "as_slice"), 1, "{body}");
    assert_eq!(count(body, "for slice("), 1, "{body}");
}

#[test]
fn a_range_is_a_range_terminator() {
    let ir = ir("let n = 3u64; let acc = 0u64; for i in 0u64..n { acc = acc + i; } acc");
    let body = main_body(&ir);
    assert_eq!(count(body, "for range("), 1, "{body}");
}

// -- What the terminator removes --------------------------------------

#[test]
fn a_traversal_writes_no_comparison_no_advance_and_no_element_read() {
    let ir = optimized("let v = vec([1, 2, 3]); let acc = 0; for x in &v { acc = acc + *x; } acc");
    let body = main_body(&ir);
    assert_eq!(count(body, "for slice("), 1, "{body}");
    assert_eq!(count(body, " lt "), 0, "{body}");
    assert_eq!(count(body, "index"), 0, "{body}");
}

#[test]
fn the_header_holds_nothing_but_the_terminator() {
    let ir = ir("let v = vec([1, 2, 3]); let acc = 0; for x in &v { acc = acc + *x; } acc");
    let body = main_body(&ir);
    let lines: Vec<&str> = body.lines().collect();
    let header = lines
        .iter()
        .position(|line| line.contains("for slice("))
        .expect("a header");
    let above = lines[header - 1];
    assert!(
        above.contains("):") || above.trim_end().ends_with(':'),
        "the header holds one instruction, and the line above it is its label:\n{body}"
    );
}

// -- break and continue -----------------------------------------------

#[test]
fn a_break_leaves_for_the_exit_and_a_continue_for_the_header() {
    let ir = ir("let v = vec([1, 2, 3]); let acc = 0; \
         for x in &v { if *x == 2 { continue; }; if *x == 3 { break; }; acc = acc + *x; } acc");
    let body = main_body(&ir);
    let lines: Vec<&str> = body.lines().collect();
    let at = lines
        .iter()
        .position(|line| line.contains("for slice("))
        .expect("a header");
    let header = label_of(
        lines[at - 1]
            .split("| ")
            .nth(1)
            .expect("the header's label"),
    );
    let exit = label_of(lines[at].split("else ").nth(1).expect("the exit edge"));
    assert_eq!(
        count(body, &format!("jump {exit}(")) + count(body, &format!("jump {exit}\n")),
        1,
        "the break is the one jump to the exit {exit}:\n{body}"
    );
    assert_eq!(
        count(body, &format!("jump {header}(")) + count(body, &format!("jump {header}\n")),
        3,
        "the block above, the continue and the latch jump to {header}:\n{body}"
    );
}

#[test]
fn a_while_takes_break_and_continue_too() {
    let ir = ir("let i = 0; let acc = 0; \
         while i < 10 { i = i + 1; if i == 3 { continue; }; if i == 5 { break; }; acc = acc + i; } acc");
    let body = main_body(&ir);
    assert!(count(body, "jump ") >= 4, "{body}");
}

// -- What the terminator answers for the passes ------------------------

#[test]
fn a_for_header_is_named_by_its_terminator_and_is_the_natural_loop() {
    let i = Interner::new();
    let module = lowered_script_module(
        &i,
        "let v = vec([1, 2, 3]); let acc = 0; for x in &v { acc = acc + *x; } acc",
        &[],
    )
    .expect("it compiles");
    let cfg = promote(module.main);
    let headers = for_headers(&cfg);
    let named: Vec<BlockIdx> = headers.keys().copied().collect();
    assert_eq!(named.len(), 1, "one `for` is one header: {headers:?}");
    let domtree = DomTree::build(&cfg);
    let found: Vec<BlockIdx> = natural_loops_innermost_first(&cfg, &domtree)
        .iter()
        .map(|l| l.header)
        .collect();
    assert_eq!(
        found, named,
        "the header the terminator names is the header the back edge finds"
    );
}

/// RFC-0057 rule 3's question, which RFC-0066 rule 6 states per target: the
/// iterations run apart when the loop's one cycle is its `&mut` source's
/// element's, `Disjoint`, and every other stage is free.
#[test]
fn iterations_run_apart_when_nothing_crosses_the_latch() {
    let i = Interner::new();
    for (source, apart) in [
        (
            "let v = vec([1, 2, 3]); for x in &mut v { *x = 0; } 0",
            true,
        ),
        (
            "let v = vec([1, 2, 3]); let acc = 0; for x in &mut v { acc = acc + *x; } acc",
            false,
        ),
        (
            "let v = vec([1, 2, 3]); let acc = 0; for x in &v { acc = acc + *x; } acc",
            false,
        ),
    ] {
        let compiled = compile_script_at(&i, source, &FxHashMap::default(), Opt::Full)
            .unwrap_or_else(|e| panic!("{source}\n{e}"));
        let cfg = promote(compiled.module.main);
        let headers: Vec<BlockIdx> = (0..cfg.blocks.len())
            .map(BlockIdx)
            .filter(|at| matches!(cfg.blocks[at.0].terminator, Terminator::For { .. }))
            .collect();
        let [header] = headers[..] else {
            panic!("{source} holds one `for`")
        };
        let deps = LoopDeps::of(&cfg, header).unwrap_or_else(|fault| panic!("{}", fault.shown()));
        let judged = deps.judge(&cfg, &compiled.laws);
        let runs_apart = deps.cycles.iter().zip(&judged).all(|(cycle, judged)| {
            cycle.tokens == [Token::Storage(Storage::Element)] && judged.order == Order::Disjoint
        });
        assert_eq!(runs_apart, apart, "{source}: {:?}", deps.cycles);
    }
}

// -- What it refuses --------------------------------------------------

#[test]
fn a_vec_by_value_names_the_borrow_it_wanted() {
    assert_eq!(
        refusal("let v = vec([1, 2, 3]); for x in v { } 0"),
        "a container is not consumed by a loop; write `&v` or `&mut v`"
    );
}

#[test]
fn a_head_of_no_admitted_shape_names_the_four() {
    assert_eq!(
        refusal("let n = 1; for x in n { } 0"),
        "a `for` traverses `&v`, `&mut v`, an array by value, or `lo..hi`; `i64` is none of them"
    );
}

#[test]
fn a_range_of_two_widths_is_refused() {
    let found = refusal("let hi = 3u32; for i in 0u64..hi { } 0");
    assert!(found.contains("u64") && found.contains("u32"), "{found}");
}

/// How many `Drop`s of the array the entry's `for` traverses stand in it.
fn array_drops(ir: &str) -> usize {
    let body = main_body(ir);
    let array = body
        .lines()
        .filter_map(|line| line.split('|').nth(1))
        .find_map(|inst| inst.trim().split_once(" = list ["))
        .map(|(array, _)| array)
        .expect("the entry builds the array");
    count(body, &format!("drop {array}\n"))
}

/// The array's `Drop` stands on the terminator's exit and on the `break`
/// edge, and it releases the elements the loop has not taken (RFC-0057
/// rule 6), so a `break` is admitted at an element that owns something.
#[test]
fn a_break_out_of_an_array_of_owners_drops_the_array_on_its_edge() {
    let ir = optimized(
        "let a = [\"x\".to_string(), \"yz\".to_string()]; let n = 0; \
         for s in a { if len(&s) == 2 { break; }; n = n + 1; } n",
    );
    assert_eq!(array_drops(&ir), 2, "{ir}");
}

/// A `?` is one more edge out of the loop, and it carries the array's `Drop`
/// as a `break` does.
#[test]
fn a_try_out_of_an_array_of_owners_drops_the_array_on_its_edge() {
    let ir = optimized(
        "let a = [\"x\".to_string(), \"yz\".to_string()]; let n = 0; for s in a { \
         let r = if len(&s) == 2 { Err(0) } else { Ok(1) }; n = n + r?; } Ok(n)",
    );
    assert_eq!(array_drops(&ir), 2, "{ir}");
}

#[test]
fn a_break_out_of_an_array_of_words_is_admitted() {
    let ir = ir("let a = [1, 2, 3]; let acc = 0; for x in a { acc = acc + x; break; } acc");
    assert_eq!(count(main_body(&ir), "for array("), 1);
}

#[test]
fn a_break_outside_every_loop_is_refused() {
    assert_eq!(refusal("break; 0"), "`break` is only inside a loop");
    assert_eq!(refusal("continue; 0"), "`continue` is only inside a loop");
}

#[test]
fn a_range_outside_a_head_is_not_an_expression() {
    let i = Interner::new();
    let parsed = acvus_ast::parse_script(&i, "let r = 0..3; 0");
    assert!(parsed.is_err(), "`..` is a `for` head and not a value");
}

#[test]
fn a_shape_write_to_a_container_under_a_traversal_is_refused() {
    let found = refusals("let v = vec([1, 2, 3]); for x in &v { push(&mut v, 4); } 0");
    assert!(!found.is_empty(), "a write under a live borrow is refused");
}
