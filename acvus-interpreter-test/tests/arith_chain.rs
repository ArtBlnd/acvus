//! An arithmetic chain stated at the contract it arrives at: the value a
//! script produces. Where the intent is that a body runs without a frame,
//! or that a run too long for one tree becomes several, no value can show
//! it, so those tests read the prepared code as well.

use acvus_interpreter::code::{ChainBounds, Code, ExprBody, Off, Prepared, Shape};
use acvus_interpreter::{PrepareCtx, Value, prepare_module};
use acvus_interpreter_test::listing::{ChainShape, chains_of};
use acvus_interpreter_test::*;
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;

fn ctx(i: &Interner, name: &str, ty: IntTy, bits: u64) -> Context {
    [(
        i.intern(name),
        typed(Ty::Int(ty), Value::from_bits(ty, bits)),
    )]
    .into_iter()
    .collect()
}

fn floats(i: &Interner, names: &[(&str, f64)]) -> Context {
    names
        .iter()
        .map(|(n, v)| (i.intern(n), typed(Ty::Float, Value::float(*v))))
        .collect()
}

/// The prepared bodies of a script. A test that states where a body runs
/// has to read them: whether a closure kept its frame is a fact about the
/// preparation, and the value the closure returns is the same either way.
fn prepared(i: &Interner, source: &str, context: Context, ret: Ty) -> Prepared {
    let (context_types, _) = split_context(i, context);
    let ast = ParsedAst::Script(acvus_ast::parse_script(i, source).expect("parse error"));
    let cr = compile_source_with_externs(i, ast, &context_types, acvus_ext::std_registries(), ret);
    let ctx = PrepareCtx {
        interner: i,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
    };
    assert_eq!(cr.modules.len(), 1, "these scripts are one module");
    let (_, module) = cr.modules.iter().next().expect("one module");
    prepare_module(module, &ctx)
}

fn chains(code: &Code) -> Vec<ChainShape> {
    chains_of(code)
}

fn only_closure(prepared: &Prepared) -> &Code {
    assert_eq!(prepared.closures.len(), 1, "these scripts have one closure");
    prepared
        .closures
        .values()
        .next()
        .expect("one closure")
        .as_ref()
}

/// One integer width and the value `@n * 2 + 1 - 3` has at it.
struct Width {
    ty: IntTy,
    n: u64,
    expected: i128,
}

#[tokio::test]
async fn a_chain_runs_at_each_integer_width() {
    let i = Interner::new();
    let widths = [
        Width {
            ty: IntTy::I8,
            n: 100,
            expected: -58,
        },
        Width {
            ty: IntTy::I16,
            n: 300,
            expected: 598,
        },
        Width {
            ty: IntTy::I32,
            n: 70_000,
            expected: 139_998,
        },
        Width {
            ty: IntTy::I64,
            n: 5_000_000_000,
            expected: 9_999_999_998,
        },
        Width {
            ty: IntTy::U8,
            n: 100,
            expected: 198,
        },
        Width {
            ty: IntTy::U16,
            n: 300,
            expected: 598,
        },
        Width {
            ty: IntTy::U32,
            n: 70_000,
            expected: 139_998,
        },
        Width {
            ty: IntTy::U64,
            n: 5_000_000_000,
            expected: 9_999_999_998,
        },
    ];
    for Width { ty, n, expected } in widths {
        let v = run_script(&i, "@n * 2 + 1 - 3", ctx(&i, "n", ty, n), Ty::Int(ty)).await;
        assert_eq!(ty.read(v.bits()), expected, "a chain at {ty:?}");
    }
}

#[tokio::test]
async fn a_float_chain_is_the_same_expression_rust_computes() {
    let i = Interner::new();
    let (x, y, c) = (0.3f64, 0.7f64, -0.4f64);
    let v = run_script(
        &i,
        "@x * @x - @y * @y + @c",
        floats(&i, &[("x", x), ("y", y), ("c", c)]),
        Ty::Float,
    )
    .await;
    assert_eq!(v.as_float(), x * x - y * y + c);
}

#[tokio::test]
async fn the_three_node_shape_reads_its_leaves_in_order() {
    let i = Interner::new();
    let (a, b, c, d) = (2.0f64, 3.0f64, 5.0f64, 7.0f64);
    let v = run_script(
        &i,
        "@a * @b - @c * @d",
        floats(&i, &[("a", a), ("b", b), ("c", c), ("d", d)]),
        Ty::Float,
    )
    .await;
    assert_eq!(v.as_float(), a * b - c * d);
    assert_eq!(v.as_float(), -29.0);
}

#[tokio::test]
async fn a_chain_of_operators_outside_the_alphabet_is_the_same_expression_rust_computes() {
    let i = Interner::new();
    let (a, b, c, d) = (24.0f64, 5.0f64, 3.0f64, 2.0f64);
    let v = run_script(
        &i,
        "@a / @b / @c / @d",
        floats(&i, &[("a", a), ("b", b), ("c", c), ("d", d)]),
        Ty::Float,
    )
    .await;
    assert_eq!(v.as_float(), a / b / c / d);

    let v = run_script(
        &i,
        "0 - (@n % 7) % 5",
        ctx(&i, "n", IntTy::I64, 93),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), -((93i64 % 7) % 5));
}

#[tokio::test]
async fn a_chain_ending_in_a_compare_drives_a_while() {
    let i = Interner::new();
    let v = run_script_mode(
        &i,
        "let s = 0; let k = 0; while k * 2 + 1 < @n { s = s + k; k = k + 1; } s",
        ctx(&i, "n", IntTy::I64, 21),
        Ty::I64,
    )
    .await;
    let mut s = 0i64;
    let mut k = 0i64;
    while k * 2 + 1 < 21 {
        s += k;
        k += 1;
    }
    assert_eq!(v.as_int(), s);
}

#[tokio::test]
#[should_panic(expected = "attempt to divide by zero")]
async fn a_division_by_zero_inside_a_chain_still_panics() {
    let i = Interner::new();
    run_script(&i, "(@n * 2) / 0", ctx(&i, "n", IntTy::I64, 3), Ty::I64).await;
}

#[tokio::test]
#[should_panic(expected = "attempt to divide with overflow")]
async fn the_minimum_divided_by_minus_one_inside_a_chain_still_panics() {
    let i = Interner::new();
    run_script(
        &i,
        "(@n * 1) / (0 - 1)",
        ctx(&i, "n", IntTy::I64, i64::MIN as u64),
        Ty::I64,
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
async fn a_remainder_by_zero_inside_a_chain_still_panics() {
    let i = Interner::new();
    run_script(&i, "(@n * 2) % 0", ctx(&i, "n", IntTy::I64, 3), Ty::I64).await;
}

#[tokio::test]
async fn a_chain_reads_a_constant_from_a_register_the_entry_filled() {
    let i = Interner::new();
    let v = run_script(&i, "@n * 3 + 4", ctx(&i, "n", IntTy::I64, 5), Ty::I64).await;
    assert_eq!(v.as_int(), 19);

    let module = prepared(&i, "@n * 3 + 4", ctx(&i, "n", IntTy::I64, 5), Ty::I64);
    let Code::Body(body) = &*module.main else {
        panic!("a script's entry body runs on a frame")
    };
    let filled: Vec<i64> = body
        .entry_konsts
        .iter()
        .map(|konst| konst.value.as_int())
        .collect();
    assert!(
        filled.contains(&3) && filled.contains(&4),
        "the chain's constants are not registers the entry fills: {filled:?}"
    );
    let chain = chains(&module.main);
    let chain = chain.first().expect("one chain");
    let konst_slots: Vec<Off> = body.entry_konsts.iter().map(|konst| konst.slot).collect();
    let reads_a_konst = chain.leaves.iter().any(|off| {
        konst_slots
            .iter()
            .any(|at| ChainBounds::byte_offset_of_word(*at) == *off)
    });
    assert!(reads_a_konst, "no leaf reads a constant's register");
}

#[tokio::test]
async fn a_constant_a_call_reads_is_not_hoisted() {
    let i = Interner::new();
    let module = prepared(
        &i,
        "let s = \"abc\"; s.len()",
        Context::default(),
        Ty::Int(IntTy::U64),
    );
    let Code::Body(body) = &*module.main else {
        panic!("a script's entry body runs on a frame")
    };
    assert!(
        body.entry_konsts.is_empty(),
        "a string a call reads became an entry constant"
    );
}

#[tokio::test]
async fn a_lambda_that_is_one_chain_runs_without_a_frame() {
    let i = Interner::new();
    let source = "let f = |x| -> x * 2 + 1; f(3)";
    let v = run_script_mode(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 7);

    let module = prepared(&i, source, Context::default(), Ty::I64);
    let Code::Expr(expr) = only_closure(&module) else {
        panic!("a lambda that is one chain is a Code::Expr")
    };
    assert_eq!(expr.arity, 1);
    let ExprBody::Chain(body) = &expr.body else {
        panic!("a lambda with an operator is a chain, not a bare argument")
    };
    assert_eq!(body.plan.shape, Shape::NNLLL);
    assert_eq!(
        body.konsts.len(),
        2,
        "the operand space holds the two constants the chain reads"
    );
}

#[tokio::test]
async fn an_identity_lambda_returns_its_argument_without_a_chain() {
    let i = Interner::new();
    let module = prepared(&i, "let f = |x| -> x; f(3)", Context::default(), Ty::I64);
    let Code::Expr(expr) = only_closure(&module) else {
        panic!("an identity lambda is a Code::Expr")
    };
    assert!(matches!(expr.body, ExprBody::Argument(0)));
}

#[tokio::test]
async fn a_lambda_of_two_statements_keeps_its_frame() {
    let i = Interner::new();
    let source = "let f = |x| -> { let y = x * 2; y + y }; f(3)";
    let v = run_script_mode(&i, source, Context::default(), Ty::I64).await;
    assert_eq!(v.as_int(), 12);

    let module = prepared(&i, source, Context::default(), Ty::I64);
    assert!(
        matches!(only_closure(&module), Code::Body(_)),
        "a body that is not one chain runs through Code::Body"
    );
}

#[tokio::test]
async fn a_lambda_that_is_one_chain_runs_without_a_frame_under_map() {
    let i = Interner::new();
    let v = run_script(
        &i,
        "range(0, @n) | map(|x| -> x * 2 + 1) | sum",
        ctx(&i, "n", IntTy::I64, 10),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), (0..10i64).map(|x| x * 2 + 1).sum::<i64>());
}

#[tokio::test]
async fn a_run_of_more_than_three_nodes_is_several_chains_through_a_register() {
    let i = Interner::new();
    let source = "@n * 2 + @n * 3 + @n * 4 + @n * 5";
    let v = run_script(&i, source, ctx(&i, "n", IntTy::I64, 6), Ty::I64).await;
    assert_eq!(v.as_int(), 6 * 2 + 6 * 3 + 6 * 4 + 6 * 5);

    let module = prepared(&i, source, ctx(&i, "n", IntTy::I64, 6), Ty::I64);
    let found = chains(&module.main);
    assert!(
        found.len() >= 2,
        "a run of seven nodes is more than one chain, found {}",
        found.len()
    );
    for chain in &found {
        assert!(
            chain.shape.interior() + 1 <= ChainBounds::MAX_NODES,
            "a prepared chain has {} nodes",
            chain.shape.interior() + 1
        );
    }
    let Code::Body(body) = &*module.main else {
        panic!("a script's entry body runs on a frame")
    };
    let written: Vec<u16> = found.iter().map(|chain| chain.dst).collect();
    let through_a_register = found.iter().any(|chain| {
        chain.leaves.iter().any(|off| {
            written
                .iter()
                .any(|at| ChainBounds::byte_offset_of_word(Off::of(*at)) == *off)
        })
    });
    assert!(
        through_a_register,
        "the split pieces do not meet in a register"
    );
}
/// Every binary tree of one to three nodes has a shape and no two trees
/// share one — the eight `Shape` names, enumerated rather than counted by
/// hand.
#[test]
fn every_binary_tree_of_at_most_three_nodes_has_its_own_shape() {
    fn trees(nodes: usize) -> Vec<String> {
        if nodes == 0 {
            return vec!["L".to_string()];
        }
        let mut out = Vec::new();
        for left in 0..nodes {
            for l in trees(left) {
                for r in trees(nodes - 1 - left) {
                    out.push(format!("N{l}{r}"));
                }
            }
        }
        out
    }
    let words: Vec<String> = (1..=ChainBounds::MAX_NODES).flat_map(trees).collect();
    assert_eq!(words.len(), 8);
    let mut shapes: Vec<String> = words
        .iter()
        .map(|word| {
            let shape = Shape::of_word(word)
                .unwrap_or_else(|| panic!("no shape is named by the preorder word {word}"));
            assert_eq!(
                shape.interior() + 1,
                word.matches('N').count(),
                "{shape:?} and {word} disagree on node count"
            );
            assert_eq!(
                shape.leaves(),
                word.matches('L').count(),
                "{shape:?} and {word} disagree on leaf count"
            );
            format!("{shape:?}")
        })
        .collect();
    shapes.sort();
    shapes.dedup();
    assert_eq!(shapes.len(), 8, "two trees share one shape");
}

#[test]
fn a_run_of_four_nodes_names_no_shape() {
    assert!(Shape::of_word("NNNNLLLLL").is_none());
}

#[test]
fn every_chain_leaf_is_inside_the_frame_it_reads_unchecked() {
    let i = Interner::new();
    let module = prepared(
        &i,
        "let s = 0; let k = 0; while k < @n { s = s + k * 2 - 1; k = k + 1; } s",
        ctx(&i, "n", IntTy::I64, 4),
        Ty::I64,
    );
    let Code::Body(body) = &*module.main else {
        panic!("a script's entry body runs on a frame")
    };
    for chain in chains(&module.main) {
        for offset in &chain.leaves {
            assert!(
                *offset < ChainBounds::byte_offset_of_word(Off::of(body.frame_len)),
                "a chain reads at offset {offset}, past the frame of {} registers",
                body.frame_len
            );
        }
    }
    assert!(
        !body.entry_konsts.is_empty(),
        "the literals 2 and 1 are registers the entry fills"
    );
}
