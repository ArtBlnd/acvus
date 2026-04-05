//! Full optimization pipeline E2E tests — cross-function calls + IO.
//!
//! Each test produces TWO snapshots:
//! - `{name}@raw` — unoptimized, all modules printed (no inlining)
//! - `{name}@optimized` — full pipeline: SROA → SSA → DSE → DCE → Inline → Pass2 → Validate
//!
//! Tests exercise: inlining, Spawn/Eval splitting, code motion, DSE, DCE, phi insertion.

use std::collections::BTreeSet;

use acvus_mir::graph::{Constraint, FnConstraint, FnKind, Function, QualifiedRef, Signature};
use acvus_mir::ty::{Effect, EffectSet, EffectTarget, Param, Ty};
use acvus_mir_test::{compile_multi_fn_optimized, compile_multi_fn_raw};
use acvus_utils::Interner;

fn test_effectful(interner: &Interner) -> Effect {
    Effect::Resolved(EffectSet {
        reads: BTreeSet::new(),
        writes: BTreeSet::from([EffectTarget::Token(QualifiedRef::root(interner.intern("__test")))]),
    })
}

fn sig(i: &Interner, params: &[(&str, Ty)]) -> Option<Signature> {
    Some(Signature {
        params: params
            .iter()
            .map(|(name, ty)| Param::new(i.intern(name), ty.clone()))
            .collect(),
    })
}

fn io_extern(i: &Interner, name: &str, params: &[(&str, Ty)], ret: Ty) -> Function {
    let sig_params: Vec<Param> = params
        .iter()
        .map(|(n, ty)| Param::new(i.intern(n), ty.clone()))
        .collect();
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern,
        constraint: FnConstraint {
            signature: Some(Signature {
                params: sig_params.clone(),
            }),
            output: Constraint::Exact(Ty::Fn {
                params: sig_params,
                ret: Box::new(ret),
                captures: vec![],
                effect: test_effectful(i),
            }),
            effect: None,
            hint: None,
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 11. Order Processing Pipeline
//     6 functions (main + 4 helper + 1 IO extern)
//     - Inline: 4 helpers flattened into main
//     - SROA: multiple flat context reads
//     - SSA: shipping branch phi, sequential computation chain
//     - SpawnSplit: send_email → Spawn + Eval
//     - CodeMotion: Spawn hoisted before Eval
//     - DSE: context write-backs after phi
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn order_processing_pipeline() {
    let i = Interner::new();
    let opt = compile_multi_fn_optimized(
        &i,
        (
            "main",
            r#"
                subtotal = @item_count * 100;
                discount = calc_discount(subtotal, @discount_rate);
                tax = calc_tax(subtotal - discount, @tax_rate);
                total = subtotal - discount + tax;
                true = total > @free_ship_min { @shipping = 0; };
                true = total <= @free_ship_min { @shipping = @default_ship; };
                final_total = total + @shipping;
                receipt = format_receipt(@customer, subtotal, discount, tax, @shipping, final_total);
                @total_out = final_total;
                @receipt_out = receipt;
                send_email(receipt);
                final_total
            "#,
        ),
        &[
            (
                "calc_discount",
                "$subtotal * $rate / 100",
                sig(&i, &[("subtotal", Ty::Int), ("rate", Ty::Int)]),
            ),
            (
                "calc_tax",
                "$amount * $rate / 100",
                sig(&i, &[("amount", Ty::Int), ("rate", Ty::Int)]),
            ),
            (
                "format_receipt",
                r#"$name + " sub:" + to_string($sub) + " disc:" + to_string($disc) + " tax:" + to_string($tax) + " ship:" + to_string($ship) + " total:" + to_string($total)"#,
                sig(
                    &i,
                    &[
                        ("name", Ty::String),
                        ("sub", Ty::Int),
                        ("disc", Ty::Int),
                        ("tax", Ty::Int),
                        ("ship", Ty::Int),
                        ("total", Ty::Int),
                    ],
                ),
            ),
        ],
        &[
            ("item_count", Ty::Int),
            ("discount_rate", Ty::Int),
            ("tax_rate", Ty::Int),
            ("free_ship_min", Ty::Int),
            ("default_ship", Ty::Int),
            ("shipping", Ty::Int),
            ("customer", Ty::String),
            ("total_out", Ty::Int),
            ("receipt_out", Ty::String),
        ],
        &[io_extern(&i, "send_email", &[("msg", Ty::String)], Ty::Int)],
    );
    let raw = compile_multi_fn_raw(
        &i,
        (
            "main",
            r#"
                subtotal = @item_count * 100;
                discount = calc_discount(subtotal, @discount_rate);
                tax = calc_tax(subtotal - discount, @tax_rate);
                total = subtotal - discount + tax;
                true = total > @free_ship_min { @shipping = 0; };
                true = total <= @free_ship_min { @shipping = @default_ship; };
                final_total = total + @shipping;
                receipt = format_receipt(@customer, subtotal, discount, tax, @shipping, final_total);
                @total_out = final_total;
                @receipt_out = receipt;
                send_email(receipt);
                final_total
            "#,
        ),
        &[
            (
                "calc_discount",
                "$subtotal * $rate / 100",
                sig(&i, &[("subtotal", Ty::Int), ("rate", Ty::Int)]),
            ),
            (
                "calc_tax",
                "$amount * $rate / 100",
                sig(&i, &[("amount", Ty::Int), ("rate", Ty::Int)]),
            ),
            (
                "format_receipt",
                r#"$name + " sub:" + to_string($sub) + " disc:" + to_string($disc) + " tax:" + to_string($tax) + " ship:" + to_string($ship) + " total:" + to_string($total)"#,
                sig(
                    &i,
                    &[
                        ("name", Ty::String),
                        ("sub", Ty::Int),
                        ("disc", Ty::Int),
                        ("tax", Ty::Int),
                        ("ship", Ty::Int),
                        ("total", Ty::Int),
                    ],
                ),
            ),
        ],
        &[
            ("item_count", Ty::Int),
            ("discount_rate", Ty::Int),
            ("tax_rate", Ty::Int),
            ("free_ship_min", Ty::Int),
            ("default_ship", Ty::Int),
            ("shipping", Ty::Int),
            ("customer", Ty::String),
            ("total_out", Ty::Int),
            ("receipt_out", Ty::String),
        ],
        &[io_extern(&i, "send_email", &[("msg", Ty::String)], Ty::Int)],
    );

    let opt = opt.unwrap();
    let raw = raw.unwrap();
    insta::assert_snapshot!("order_processing_pipeline@optimized", opt);
    insta::assert_snapshot!("order_processing_pipeline@raw", raw);
}

// ═══════════════════════════════════════════════════════════════════════
// 12. User Analytics — loop + classify + context accumulation + IO report
//     4 functions (main + 2 helper + 1 IO extern)
//     - Loop: user iteration, 5 context writes per iteration
//     - Inline: classify_age (nested branches), build_summary (string chain)
//     - SSA: 5+ loop phi + branch phi inside loop
//     - SpawnSplit: send_report → Spawn + Eval
//     - DSE: loop header phi write-backs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn user_analytics_dashboard() {
    let i = Interner::new();
    let user_ty = Ty::Object(
        [
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]
        .into_iter()
        .collect(),
    );
    let target = (
        "main",
        r#"
                user in @users {
                    group = classify_age(user.age);
                    true = group > 1 { @senior_count = @senior_count + 1; };
                    true = group == 1 { @adult_count = @adult_count + 1; };
                    true = group == 0 { @minor_count = @minor_count + 1; };
                    @total_age = @total_age + user.age;
                    @name_list = @name_list + user.name + ", ";
                };
                total_users = @senior_count + @adult_count + @minor_count;
                summary = build_summary(@senior_count, @adult_count, @minor_count, @total_age, @name_list);
                send_report(summary);
                @output = summary;
                summary
            "#,
    );
    let helpers: &[_] = &[
        // Returns Int: 2=senior, 1=adult, 0=minor.
        // Each branch writes to a local; tail is the final value.
        (
            "classify_age",
            r#"
                    r = 0;
                    true = $age >= 65 { r = 2; };
                    true = $age >= 18 { r = 1; };
                    r
                "#,
            sig(&i, &[("age", Ty::Int)]),
        ),
        (
            "build_summary",
            r#"
                    "S:" + to_string($seniors) + " A:" + to_string($adults) + " M:" + to_string($minors) + " age:" + to_string($total_age) + " | " + $names
                "#,
            sig(
                &i,
                &[
                    ("seniors", Ty::Int),
                    ("adults", Ty::Int),
                    ("minors", Ty::Int),
                    ("total_age", Ty::Int),
                    ("names", Ty::String),
                ],
            ),
        ),
    ];
    let contexts: &[_] = &[
        ("users", Ty::List(Box::new(user_ty))),
        ("senior_count", Ty::Int),
        ("adult_count", Ty::Int),
        ("minor_count", Ty::Int),
        ("total_age", Ty::Int),
        ("name_list", Ty::String),
        ("output", Ty::String),
    ];
    let extern_fns: &[_] = &[io_extern(
        &i,
        "send_report",
        &[("summary", Ty::String)],
        Ty::Int,
    )];
    let opt = compile_multi_fn_optimized(&i, target, helpers, contexts, extern_fns).unwrap();
    let raw = compile_multi_fn_raw(&i, target, helpers, contexts, extern_fns).unwrap();
    insta::assert_snapshot!("user_analytics_dashboard@optimized", opt);
    insta::assert_snapshot!("user_analytics_dashboard@raw", raw);
}

// ═══════════════════════════════════════════════════════════════════════
// 13. Data Enrichment — two independent IO fetches + conditional third IO
//     5 functions (main + 2 helper + 3 IO extern)
//     - SpawnSplit: fetch_profile + fetch_history → two parallel Spawns
//     - CodeMotion: both Spawns hoisted to function start
//     - Inline: compute_score, format_label
//     - SSA: alert_count branch phi
//     - DSE: context writes
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn data_enrichment_multi_io() {
    let i = Interner::new();
    let target = (
        "main",
        r#"
                profile = fetch_profile(@user_id);
                history = fetch_history(@user_id);
                score = compute_score(profile, history, @weight);
                label = format_label(profile, score);
                true = score > @threshold {
                    notify_alert(@user_id, score);
                    @alert_count = @alert_count + 1;
                };
                @result_label = label;
                @result_score = score;
                score
            "#,
    );
    let helpers: &[_] = &[
        (
            "compute_score",
            "($profile + $history) * $weight / 100",
            sig(
                &i,
                &[
                    ("profile", Ty::Int),
                    ("history", Ty::Int),
                    ("weight", Ty::Int),
                ],
            ),
        ),
        (
            "format_label",
            r#""User(" + to_string($profile) + " score:" + to_string($score) + ")""#,
            sig(&i, &[("profile", Ty::Int), ("score", Ty::Int)]),
        ),
    ];
    let contexts: &[_] = &[
        ("user_id", Ty::Int),
        ("weight", Ty::Int),
        ("threshold", Ty::Int),
        ("alert_count", Ty::Int),
        ("result_label", Ty::String),
        ("result_score", Ty::Int),
    ];
    let extern_fns: &[_] = &[
        io_extern(&i, "fetch_profile", &[("id", Ty::Int)], Ty::Int),
        io_extern(&i, "fetch_history", &[("id", Ty::Int)], Ty::Int),
        io_extern(
            &i,
            "notify_alert",
            &[("id", Ty::Int), ("score", Ty::Int)],
            Ty::Int,
        ),
    ];
    let opt = compile_multi_fn_optimized(&i, target, helpers, contexts, extern_fns).unwrap();
    let raw = compile_multi_fn_raw(&i, target, helpers, contexts, extern_fns).unwrap();
    insta::assert_snapshot!("data_enrichment_multi_io@optimized", opt);
    insta::assert_snapshot!("data_enrichment_multi_io@raw", raw);
}

// ═══════════════════════════════════════════════════════════════════════
// 14. Batch Processing — loop + validate/transform helpers + error accumulation + IO
//     4 functions (main + 2 helper + 1 IO extern)
//     - Loop: item iteration with branch (valid/invalid)
//     - Inline: validate_item (nested compare), transform_value (arithmetic)
//     - SSA: 4 context loop phi × branch phi — most complex phi pattern
//     - SpawnSplit: publish_results → Spawn + Eval
//     - DSE: loop header dead write-backs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn batch_processing_with_errors() {
    let i = Interner::new();
    let item_ty = Ty::Object(
        [(i.intern("value"), Ty::Int)].into_iter().collect(),
    );
    let target = (
        "main",
        r#"
                item in @items {
                    v = item.value;
                    ok = validate_item(v, @min_val, @max_val);
                    true = ok > 0 {
                        transformed = transform_value(v, @multiplier);
                        @sum = @sum + transformed;
                        @ok_count = @ok_count + 1;
                    };
                    true = ok == 0 {
                        @err_count = @err_count + 1;
                        @last_err = "bad:" + to_string(v);
                    };
                };
                publish_results(@sum, @ok_count, @err_count);
                @sum
            "#,
    );
    let helpers: &[_] = &[
        // Returns 1 if valid (in range), 0 otherwise.
        (
            "validate_item",
            r#"
                    r = 0;
                    true = $val >= $min { true = $val <= $max { r = 1; }; };
                    r
                "#,
            sig(
                &i,
                &[
                    ("val", Ty::Int),
                    ("min", Ty::Int),
                    ("max", Ty::Int),
                ],
            ),
        ),
        (
            "transform_value",
            "$val * $mult + $val / 2",
            sig(&i, &[("val", Ty::Int), ("mult", Ty::Int)]),
        ),
    ];
    let contexts: &[_] = &[
        ("items", Ty::List(Box::new(item_ty))),
        ("min_val", Ty::Int),
        ("max_val", Ty::Int),
        ("multiplier", Ty::Int),
        ("sum", Ty::Int),
        ("ok_count", Ty::Int),
        ("err_count", Ty::Int),
        ("last_err", Ty::String),
    ];
    let extern_fns: &[_] = &[io_extern(
        &i,
        "publish_results",
        &[("sum", Ty::Int), ("ok", Ty::Int), ("err", Ty::Int)],
        Ty::Int,
    )];
    let opt = compile_multi_fn_optimized(&i, target, helpers, contexts, extern_fns).unwrap();
    let raw = compile_multi_fn_raw(&i, target, helpers, contexts, extern_fns).unwrap();
    insta::assert_snapshot!("batch_processing_with_errors@optimized", opt);
    insta::assert_snapshot!("batch_processing_with_errors@raw", raw);
}

// ═══════════════════════════════════════════════════════════════════════
// 15. Multi-Stage Pipeline — cascading helpers + two IO calls
//     5 functions (main + 3 helper + 2 IO extern)
//     - Inline: 3 sequential helpers → flat computation chain
//     - SpawnSplit: fetch_data (start) + log_pipeline (end)
//     - CodeMotion: fetch_data Spawn at start, log_pipeline Spawn after stage3
//     - SSA: sequential (no branches)
//     - DSE: intermediate context writes are live (observable)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn multi_stage_pipeline() {
    let i = Interner::new();
    let target = (
        "main",
        r#"
                raw = fetch_data(@source_id);
                s1 = normalize(raw, @scale);
                @stage1 = s1;
                s2 = enrich(s1, @offset);
                @stage2 = s2;
                s3 = finalize(s2, @precision);
                @stage3 = s3;
                log_pipeline(s1, s2, s3);
                s3
            "#,
    );
    let helpers: &[_] = &[
        (
            "normalize",
            "$val * $scale / 1000",
            sig(&i, &[("val", Ty::Int), ("scale", Ty::Int)]),
        ),
        (
            "enrich",
            "$val + $offset + $val / 10",
            sig(&i, &[("val", Ty::Int), ("offset", Ty::Int)]),
        ),
        (
            "finalize",
            "($val / $prec) * $prec",
            sig(&i, &[("val", Ty::Int), ("prec", Ty::Int)]),
        ),
    ];
    let contexts: &[_] = &[
        ("source_id", Ty::Int),
        ("scale", Ty::Int),
        ("offset", Ty::Int),
        ("precision", Ty::Int),
        ("stage1", Ty::Int),
        ("stage2", Ty::Int),
        ("stage3", Ty::Int),
    ];
    let extern_fns: &[_] = &[
        io_extern(&i, "fetch_data", &[("id", Ty::Int)], Ty::Int),
        io_extern(
            &i,
            "log_pipeline",
            &[("s1", Ty::Int), ("s2", Ty::Int), ("s3", Ty::Int)],
            Ty::Int,
        ),
    ];
    let opt = compile_multi_fn_optimized(&i, target, helpers, contexts, extern_fns).unwrap();
    let raw = compile_multi_fn_raw(&i, target, helpers, contexts, extern_fns).unwrap();
    insta::assert_snapshot!("multi_stage_pipeline@optimized", opt);
    insta::assert_snapshot!("multi_stage_pipeline@raw", raw);
}
