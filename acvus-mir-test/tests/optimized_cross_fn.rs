//! Full optimization pipeline E2E tests - cross-function calls + IO.
//!
//! Each test produces TWO snapshots:
//! - `{name}@raw` - unoptimized, all modules printed (no inlining)
//! - `{name}@optimized` - full pipeline: SROA -> SSA -> DSE -> DCE -> Inline -> Pass2 -> Validate
//!
//! Tests exercise: inlining, Spawn/Eval splitting, code motion, DSE, DCE, phi insertion.

use acvus_mir::graph::{FnKind, Function, QualifiedRef};
use acvus_mir::ty::{ParamTerm, Poly, PolyParam, Ty, TyTerm, lift_to_poly};
use acvus_mir_test::{compile_multi_fn_optimized, compile_multi_fn_raw};
use acvus_utils::Interner;

fn sig(i: &Interner, params: &[(&str, Ty)]) -> Vec<PolyParam> {
    params
        .iter()
        .map(|(name, ty)| ParamTerm::<Poly>::new(i.intern(name), lift_to_poly(ty)))
        .collect()
}

fn io_extern(i: &Interner, name: &str, params: &[(&str, Ty)], ret: Ty) -> Function {
    let infer_params: Vec<ParamTerm<Poly>> = params
        .iter()
        .map(|(n, ty)| ParamTerm::<Poly>::new(i.intern(n), lift_to_poly(ty)))
        .collect();
    Function {
        qref: QualifiedRef::root(i.intern(name)),
        kind: FnKind::Extern {
            bounds: vec![],
            instances: Default::default(),
        },
        ty: TyTerm::Fn {
            params: infer_params,
            ret: Box::new(lift_to_poly(&ret)),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    }
}

// =======================================================================
// 11. Order Processing Pipeline
//     6 functions (main + 4 helper + 1 IO extern)
//     - Inline: 4 helpers flattened into main
//     - SROA: multiple flat context reads
//     - SSA: shipping branch phi, sequential computation chain
//     - SpawnSplit: send_email -> Spawn + Eval
//     - CodeMotion: Spawn hoisted before Eval
//     - DSE: context write-backs after phi
// =======================================================================

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
                sig(&i, &[("subtotal", Ty::I64), ("rate", Ty::I64)]),
            ),
            (
                "calc_tax",
                "$amount * $rate / 100",
                sig(&i, &[("amount", Ty::I64), ("rate", Ty::I64)]),
            ),
            (
                "format_receipt",
                r#"$name + " sub:" + $sub.to_string() + " disc:" + $disc.to_string() + " tax:" + $tax.to_string() + " ship:" + $ship.to_string() + " total:" + $total.to_string()"#,
                sig(
                    &i,
                    &[
                        ("name", Ty::String),
                        ("sub", Ty::I64),
                        ("disc", Ty::I64),
                        ("tax", Ty::I64),
                        ("ship", Ty::I64),
                        ("total", Ty::I64),
                    ],
                ),
            ),
        ],
        &[
            ("item_count", Ty::I64),
            ("discount_rate", Ty::I64),
            ("tax_rate", Ty::I64),
            ("free_ship_min", Ty::I64),
            ("default_ship", Ty::I64),
            ("shipping", Ty::I64),
            ("customer", Ty::String),
            ("total_out", Ty::I64),
            ("receipt_out", Ty::String),
        ],
        &[io_extern(&i, "send_email", &[("msg", Ty::String)], Ty::I64)],
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
                sig(&i, &[("subtotal", Ty::I64), ("rate", Ty::I64)]),
            ),
            (
                "calc_tax",
                "$amount * $rate / 100",
                sig(&i, &[("amount", Ty::I64), ("rate", Ty::I64)]),
            ),
            (
                "format_receipt",
                r#"$name + " sub:" + $sub.to_string() + " disc:" + $disc.to_string() + " tax:" + $tax.to_string() + " ship:" + $ship.to_string() + " total:" + $total.to_string()"#,
                sig(
                    &i,
                    &[
                        ("name", Ty::String),
                        ("sub", Ty::I64),
                        ("disc", Ty::I64),
                        ("tax", Ty::I64),
                        ("ship", Ty::I64),
                        ("total", Ty::I64),
                    ],
                ),
            ),
        ],
        &[
            ("item_count", Ty::I64),
            ("discount_rate", Ty::I64),
            ("tax_rate", Ty::I64),
            ("free_ship_min", Ty::I64),
            ("default_ship", Ty::I64),
            ("shipping", Ty::I64),
            ("customer", Ty::String),
            ("total_out", Ty::I64),
            ("receipt_out", Ty::String),
        ],
        &[io_extern(&i, "send_email", &[("msg", Ty::String)], Ty::I64)],
    );

    let opt = opt.unwrap();
    let raw = raw.unwrap();
    insta::assert_snapshot!("order_processing_pipeline@optimized", opt);
    insta::assert_snapshot!("order_processing_pipeline@raw", raw);
}

// =======================================================================
// 12. User Analytics - loop + classify + context accumulation + IO report
//     4 functions (main + 2 helper + 1 IO extern)
//     - Loop: user iteration, 5 context writes per iteration
//     - Inline: classify_age (nested branches), build_summary (string chain)
//     - SSA: 5+ loop phi + branch phi inside loop
//     - SpawnSplit: send_report -> Spawn + Eval
//     - DSE: loop header phi write-backs
// =======================================================================

// =======================================================================
// 13. Data Enrichment - two independent IO fetches + conditional third IO
//     5 functions (main + 2 helper + 3 IO extern)
//     - SpawnSplit: fetch_profile + fetch_history -> two parallel Spawns
//     - CodeMotion: both Spawns hoisted to function start
//     - Inline: compute_score, format_label
//     - SSA: alert_count branch phi
//     - DSE: context writes
// =======================================================================

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
                    ("profile", Ty::I64),
                    ("history", Ty::I64),
                    ("weight", Ty::I64),
                ],
            ),
        ),
        (
            "format_label",
            r#"a = "User("; b = $profile.to_string(); c = " score:"; d = $score.to_string(); e = ")"; ab = concat(&a, &b); abc = concat(&ab, &c); abcd = concat(&abc, &d); concat(&abcd, &e)"#,
            sig(&i, &[("profile", Ty::I64), ("score", Ty::I64)]),
        ),
    ];
    let contexts: &[_] = &[
        ("user_id", Ty::I64),
        ("weight", Ty::I64),
        ("threshold", Ty::I64),
        ("alert_count", Ty::I64),
        ("result_label", Ty::String),
        ("result_score", Ty::I64),
    ];
    let extern_fns: &[_] = &[
        io_extern(&i, "fetch_profile", &[("id", Ty::I64)], Ty::I64),
        io_extern(&i, "fetch_history", &[("id", Ty::I64)], Ty::I64),
        io_extern(
            &i,
            "notify_alert",
            &[("id", Ty::I64), ("score", Ty::I64)],
            Ty::I64,
        ),
    ];
    let opt = compile_multi_fn_optimized(&i, target, helpers, contexts, extern_fns).unwrap();
    let raw = compile_multi_fn_raw(&i, target, helpers, contexts, extern_fns).unwrap();
    insta::assert_snapshot!("data_enrichment_multi_io@optimized", opt);
    insta::assert_snapshot!("data_enrichment_multi_io@raw", raw);
}

// =======================================================================
// 14. Batch Processing - loop + validate/transform helpers + error accumulation + IO
//     4 functions (main + 2 helper + 1 IO extern)
//     - Loop: item iteration with branch (valid/invalid)
//     - Inline: validate_item (nested compare), transform_value (arithmetic)
//     - SSA: 4 context loop phi x branch phi - most complex phi pattern
//     - SpawnSplit: publish_results -> Spawn + Eval
//     - DSE: loop header dead write-backs
// =======================================================================

// =======================================================================
// 15. Multi-Stage Pipeline - cascading helpers + two IO calls
//     5 functions (main + 3 helper + 2 IO extern)
//     - Inline: 3 sequential helpers -> flat computation chain
//     - SpawnSplit: fetch_data (start) + log_pipeline (end)
//     - CodeMotion: fetch_data Spawn at start, log_pipeline Spawn after stage3
//     - SSA: sequential (no branches)
//     - DSE: intermediate context writes are live (observable)
// =======================================================================

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
            sig(&i, &[("val", Ty::I64), ("scale", Ty::I64)]),
        ),
        (
            "enrich",
            "$val + $offset + $val / 10",
            sig(&i, &[("val", Ty::I64), ("offset", Ty::I64)]),
        ),
        (
            "finalize",
            "($val / $prec) * $prec",
            sig(&i, &[("val", Ty::I64), ("prec", Ty::I64)]),
        ),
    ];
    let contexts: &[_] = &[
        ("source_id", Ty::I64),
        ("scale", Ty::I64),
        ("offset", Ty::I64),
        ("precision", Ty::I64),
        ("stage1", Ty::I64),
        ("stage2", Ty::I64),
        ("stage3", Ty::I64),
    ];
    let extern_fns: &[_] = &[
        io_extern(&i, "fetch_data", &[("id", Ty::I64)], Ty::I64),
        io_extern(
            &i,
            "log_pipeline",
            &[("s1", Ty::I64), ("s2", Ty::I64), ("s3", Ty::I64)],
            Ty::I64,
        ),
    ];
    let opt = compile_multi_fn_optimized(&i, target, helpers, contexts, extern_fns).unwrap();
    let raw = compile_multi_fn_raw(&i, target, helpers, contexts, extern_fns).unwrap();
    insta::assert_snapshot!("multi_stage_pipeline@optimized", opt);
    insta::assert_snapshot!("multi_stage_pipeline@raw", raw);
}
