use acvus_macro::{acvus_script, acvus_template};

#[test]
fn script_no_placeholder() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_script!("1 + 2");
    let script = make(&interner);
    assert!(script.tail.is_some());
}

#[test]
fn script_context_read() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_script!("@x + 1");
    let script = make(&interner);
    assert!(script.tail.is_some());
}

#[test]
fn script_pipe_with_lambda() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_script!("@data | map(|x| -> x + 1)");
    let script = make(&interner);
    assert!(script.tail.is_some());
}

#[test]
fn script_lambda_pipe_in_body() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_script!("@data | map(|x| -> x | to_string)");
    let script = make(&interner);
    assert!(script.tail.is_some());
}

#[test]
fn script_placeholder_substitution() {
    let interner = acvus_utils::Interner::new();

    // Create an Expr to substitute: literal 42
    let replacement = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(42),
        span: acvus_ast::Span::ZERO,
    };

    let make = acvus_script!("%result + 1");
    let script = make(&interner, replacement);

    // The tail should be BinaryOp(Literal(42), Add, Literal(1))
    let tail = script.tail.as_ref().unwrap();
    match tail.as_ref() {
        acvus_ast::Expr::BinaryOp { left, op, .. } => {
            assert_eq!(*op, acvus_ast::BinOp::Add);
            match left.as_ref() {
                acvus_ast::Expr::Literal {
                    value: acvus_ast::Literal::Int(42),
                    ..
                } => {}
                other => panic!("expected Literal(42), got {other:?}"),
            }
        }
        other => panic!("expected BinaryOp, got {other:?}"),
    }
}

#[test]
fn script_multiple_placeholders() {
    let interner = acvus_utils::Interner::new();

    let a = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(10),
        span: acvus_ast::Span::ZERO,
    };
    let b = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(20),
        span: acvus_ast::Span::ZERO,
    };

    let make = acvus_script!("%a + %b");
    let script = make(&interner, a, b);

    let tail = script.tail.as_ref().unwrap();
    match tail.as_ref() {
        acvus_ast::Expr::BinaryOp { left, right, .. } => {
            match left.as_ref() {
                acvus_ast::Expr::Literal {
                    value: acvus_ast::Literal::Int(10),
                    ..
                } => {}
                other => panic!("expected Literal(10), got {other:?}"),
            }
            match right.as_ref() {
                acvus_ast::Expr::Literal {
                    value: acvus_ast::Literal::Int(20),
                    ..
                } => {}
                other => panic!("expected Literal(20), got {other:?}"),
            }
        }
        other => panic!("expected BinaryOp, got {other:?}"),
    }
}

#[test]
fn script_placeholder_with_context() {
    let interner = acvus_utils::Interner::new();

    let result = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::String("hello".into()),
        span: acvus_ast::Span::ZERO,
    };

    let make = acvus_script!("@history = append(@history, %result); @history");
    let script = make(&interner, result);

    // Should have 1 stmt (context store) + tail (@history)
    assert_eq!(script.stmts.len(), 1);
    assert!(script.tail.is_some());
}

#[test]
fn template_no_placeholder() {
    let interner = acvus_utils::Interner::new();
    let make = acvus_template!("hello world");
    let template = make(&interner);
    assert!(!template.body.is_empty());
}

#[test]
fn template_with_placeholder() {
    let interner = acvus_utils::Interner::new();

    let value = acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(99),
        span: acvus_ast::Span::ZERO,
    };

    let make = acvus_template!("result: {{ %value }}");
    let template = make(&interner, value);
    assert!(!template.body.is_empty());
}

// ── Splice tests ─────────────────────────────────────────────────────

fn int_expr(n: i64) -> acvus_ast::Expr {
    acvus_ast::Expr::Literal {
        value: acvus_ast::Literal::Int(n),
        span: acvus_ast::Span::ZERO,
    }
}

fn ident_expr(interner: &acvus_utils::Interner, name: &str) -> acvus_ast::Expr {
    acvus_ast::Expr::Ident {
        name: interner.intern(name),
        ref_kind: acvus_ast::RefKind::Value,
        span: acvus_ast::Span::ZERO,
    }
}

/// Flatten a left-associative pipe chain into a vec of stage expressions.
fn flatten_pipe(expr: &acvus_ast::Expr) -> Vec<&acvus_ast::Expr> {
    match expr {
        acvus_ast::Expr::Pipe { left, right, .. } => {
            let mut stages = flatten_pipe(left);
            stages.push(right);
            stages
        }
        other => vec![other],
    }
}

/// Flatten a left-associative binary op chain (same op) into a vec of operand expressions.
fn flatten_binop<'a>(
    expr: &'a acvus_ast::Expr,
    target_op: acvus_ast::BinOp,
) -> Vec<&'a acvus_ast::Expr> {
    match expr {
        acvus_ast::Expr::BinaryOp {
            left, op, right, ..
        } if *op == target_op => {
            let mut parts = flatten_binop(left, target_op);
            parts.push(right);
            parts
        }
        other => vec![other],
    }
}

fn assert_int(expr: &acvus_ast::Expr, expected: i64) {
    match expr {
        acvus_ast::Expr::Literal {
            value: acvus_ast::Literal::Int(n),
            ..
        } => {
            assert_eq!(*n, expected, "expected Int({expected}), got Int({n})");
        }
        other => panic!("expected Int({expected}), got {other:?}"),
    }
}

fn assert_ident(expr: &acvus_ast::Expr, interner: &acvus_utils::Interner, expected: &str) {
    match expr {
        acvus_ast::Expr::Ident { name, .. } => {
            let expected_astr = interner.intern(expected);
            assert_eq!(*name, expected_astr, "expected ident '{expected}'");
        }
        other => panic!("expected Ident({expected}), got {other:?}"),
    }
}

#[test]
fn splice_list_elements() {
    let interner = acvus_utils::Interner::new();
    let items = vec![int_expr(2), int_expr(3), int_expr(4)];

    let make = acvus_script!("[%first, *rest, %last]");
    let script = make(&interner, int_expr(1), items, int_expr(5));
    let tail = script.tail.as_ref().unwrap();

    match tail.as_ref() {
        acvus_ast::Expr::List {
            head, rest, tail, ..
        } => {
            assert!(rest.is_none());
            assert!(tail.is_empty());
            assert_eq!(head.len(), 5);
            assert_int(&head[0], 1);
            assert_int(&head[1], 2);
            assert_int(&head[2], 3);
            assert_int(&head[3], 4);
            assert_int(&head[4], 5);
        }
        other => panic!("expected List, got {other:?}"),
    }
}

#[test]
fn splice_func_args() {
    let interner = acvus_utils::Interner::new();
    let extra_args = vec![int_expr(2), int_expr(3)];

    let make = acvus_script!("f(%a, *rest)");
    let script = make(&interner, int_expr(1), extra_args);
    let tail = script.tail.as_ref().unwrap();

    match tail.as_ref() {
        acvus_ast::Expr::FuncCall { args, .. } => {
            assert_eq!(args.len(), 3);
            assert_int(&args[0], 1);
            assert_int(&args[1], 2);
            assert_int(&args[2], 3);
        }
        other => panic!("expected FuncCall, got {other:?}"),
    }
}

#[test]
fn splice_pipe_chain() {
    let interner = acvus_utils::Interner::new();
    let transforms = vec![
        ident_expr(&interner, "map_fn"),
        ident_expr(&interner, "filter_fn"),
    ];

    let make = acvus_script!("%input | *transforms | to_string");
    let script = make(&interner, int_expr(1), transforms);
    let tail = script.tail.as_ref().unwrap();

    // Should be: 1 | map_fn | filter_fn | to_string (left-associative)
    let stages = flatten_pipe(tail);
    assert_eq!(stages.len(), 4);
    assert_int(stages[0], 1);
    assert_ident(stages[1], &interner, "map_fn");
    assert_ident(stages[2], &interner, "filter_fn");
    assert_ident(stages[3], &interner, "to_string");
}

#[test]
fn splice_binop_chain() {
    let interner = acvus_utils::Interner::new();
    let middle = vec![int_expr(2), int_expr(3)];

    let make = acvus_script!("%a + *middle + %b");
    let script = make(&interner, int_expr(1), middle, int_expr(4));
    let tail = script.tail.as_ref().unwrap();

    // Should be: ((1 + 2) + 3) + 4 (left-associative Add chain)
    let parts = flatten_binop(tail, acvus_ast::BinOp::Add);
    assert_eq!(parts.len(), 4);
    assert_int(parts[0], 1);
    assert_int(parts[1], 2);
    assert_int(parts[2], 3);
    assert_int(parts[3], 4);
}

#[test]
fn splice_empty_vec() {
    let interner = acvus_utils::Interner::new();
    let empty: Vec<acvus_ast::Expr> = vec![];

    let make = acvus_script!("[%a, *rest]");
    let script = make(&interner, int_expr(1), empty);
    let tail = script.tail.as_ref().unwrap();

    match tail.as_ref() {
        acvus_ast::Expr::List {
            head, rest, tail, ..
        } => {
            assert!(rest.is_none());
            assert!(tail.is_empty());
            assert_eq!(head.len(), 1);
            assert_int(&head[0], 1);
        }
        other => panic!("expected List, got {other:?}"),
    }
}

#[test]
fn splice_empty_pipe() {
    let interner = acvus_utils::Interner::new();
    let empty: Vec<acvus_ast::Expr> = vec![];

    // %a | *empty → just %a (no pipe, splice is empty)
    let make = acvus_script!("%a | *transforms");
    let script = make(&interner, int_expr(42), empty);
    let tail = script.tail.as_ref().unwrap();

    assert_int(tail, 42);
}

#[test]
fn splice_mixed_single_and_splice() {
    let interner = acvus_utils::Interner::new();
    let items = vec![int_expr(2), int_expr(3)];

    let make = acvus_script!("f(%a, *items, %b)");
    let script = make(&interner, int_expr(1), items, int_expr(4));
    let tail = script.tail.as_ref().unwrap();

    match tail.as_ref() {
        acvus_ast::Expr::FuncCall { args, .. } => {
            assert_eq!(args.len(), 4);
            assert_int(&args[0], 1);
            assert_int(&args[1], 2);
            assert_int(&args[2], 3);
            assert_int(&args[3], 4);
        }
        other => panic!("expected FuncCall, got {other:?}"),
    }
}

/// Roundtrip: splice result should equal manually constructed equivalent.
#[test]
fn splice_pipe_roundtrip() {
    let interner = acvus_utils::Interner::new();

    // Manual: %a | %b | %c
    let make_manual = acvus_script!("%a | %b | %c");
    let manual = make_manual(
        &interner,
        ident_expr(&interner, "x"),
        ident_expr(&interner, "f"),
        ident_expr(&interner, "g"),
    );

    // Splice: %a | *rest
    let make_splice = acvus_script!("%a | *rest");
    let spliced = make_splice(
        &interner,
        ident_expr(&interner, "x"),
        vec![ident_expr(&interner, "f"), ident_expr(&interner, "g")],
    );

    // Both should produce the same pipe chain structure
    let manual_stages = flatten_pipe(manual.tail.as_ref().unwrap());
    let splice_stages = flatten_pipe(spliced.tail.as_ref().unwrap());

    assert_eq!(manual_stages.len(), splice_stages.len());
    for (m, s) in manual_stages.iter().zip(splice_stages.iter()) {
        match (m, s) {
            (acvus_ast::Expr::Ident { name: mn, .. }, acvus_ast::Expr::Ident { name: sn, .. }) => {
                assert_eq!(mn, sn)
            }
            _ => panic!("stage mismatch: {m:?} vs {s:?}"),
        }
    }
}
