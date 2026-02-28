use std::fmt;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};

use crate::ir::{ClosureBody, InstKind, Label, MirBody, MirModule, Val};

fn fmt_val(r: Val) -> String {
    format!("r{}", r.0)
}

fn fmt_label(l: Label) -> String {
    format!("L{}", l.0)
}

fn fmt_literal(lit: &Literal) -> String {
    match lit {
        Literal::Int(n) => n.to_string(),
        Literal::Float(f) => format!("{f:?}"),
        Literal::String(s) => format!("{s:?}"),
        Literal::Bool(b) => b.to_string(),
    }
}

fn fmt_binop(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Eq => "==",
        BinOp::Neq => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Lte => "<=",
        BinOp::Gte => ">=",
    }
}

fn fmt_unaryop(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Not => "!",
    }
}

fn fmt_range_kind(kind: RangeKind) -> &'static str {
    match kind {
        RangeKind::Exclusive => "..",
        RangeKind::InclusiveEnd => "..=",
        RangeKind::ExclusiveStart => "=..",
    }
}

fn fmt_vals(regs: &[Val]) -> String {
    regs.iter()
        .map(|r| fmt_val(*r))
        .collect::<Vec<_>>()
        .join(", ")
}

fn write_body(f: &mut fmt::Formatter<'_>, body: &MirBody, indent: &str) -> fmt::Result {
    for (i, inst) in body.insts.iter().enumerate() {
        let is_label = matches!(&inst.kind, InstKind::BlockLabel { .. });
        // Fixed-width index column, then content indent for non-labels.
        if is_label {
            write!(f, "{indent}{i:>4} │ ")?;
        } else {
            write!(f, "{indent}{i:>4} │   ")?;
        }

        match &inst.kind {
            // Output
            InstKind::EmitText(idx) => writeln!(f, "emit_text T{idx}")?,
            InstKind::EmitValue(r) => writeln!(f, "emit {}", fmt_val(*r))?,

            // Constants / variables
            InstKind::Const { dst, value } => {
                writeln!(f, "{} = const {}", fmt_val(*dst), fmt_literal(value))?
            }
            InstKind::StorageLoad { dst, name } => {
                writeln!(f, "{} = storage_load ${name}", fmt_val(*dst))?
            }
            InstKind::StorageStore { name, src } => {
                writeln!(f, "storage_store ${name} = {}", fmt_val(*src))?
            }

            // Arithmetic / logic
            InstKind::BinOp {
                dst,
                op,
                left,
                right,
            } => writeln!(
                f,
                "{} = {} {} {}",
                fmt_val(*dst),
                fmt_val(*left),
                fmt_binop(*op),
                fmt_val(*right)
            )?,
            InstKind::UnaryOp { dst, op, operand } => writeln!(
                f,
                "{} = {}{}",
                fmt_val(*dst),
                fmt_unaryop(*op),
                fmt_val(*operand)
            )?,
            InstKind::FieldGet { dst, object, field } => {
                writeln!(f, "{} = {}.{field}", fmt_val(*dst), fmt_val(*object))?
            }

            // Calls
            InstKind::Call { dst, func, args } => {
                writeln!(f, "{} = call {func}({})", fmt_val(*dst), fmt_vals(args))?
            }
            InstKind::AsyncCall { dst, func, args } => writeln!(
                f,
                "{} = async_call {func}({})",
                fmt_val(*dst),
                fmt_vals(args)
            )?,
            InstKind::Await { dst, src } => {
                writeln!(f, "{} = await {}", fmt_val(*dst), fmt_val(*src))?
            }

            // Composite constructors
            InstKind::MakeList { dst, elements } => {
                writeln!(f, "{} = list [{}]", fmt_val(*dst), fmt_vals(elements))?
            }
            InstKind::MakeObject { dst, fields } => {
                let fields_str: String = fields
                    .iter()
                    .map(|(k, r)| format!("{k}: {}", fmt_val(*r)))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(f, "{} = object {{{fields_str}}}", fmt_val(*dst))?
            }
            InstKind::MakeRange {
                dst,
                start,
                end,
                kind,
            } => writeln!(
                f,
                "{} = range {}{}{}",
                fmt_val(*dst),
                fmt_val(*start),
                fmt_range_kind(*kind),
                fmt_val(*end)
            )?,

            // Pattern matching
            InstKind::TestLiteral { dst, src, value } => writeln!(
                f,
                "{} = test {} == {}",
                fmt_val(*dst),
                fmt_val(*src),
                fmt_literal(value)
            )?,
            InstKind::TestListLen {
                dst,
                src,
                min_len,
                exact,
            } => {
                let op = if *exact { "==" } else { ">=" };
                writeln!(
                    f,
                    "{} = test len({}) {op} {min_len}",
                    fmt_val(*dst),
                    fmt_val(*src)
                )?
            }
            InstKind::TestObjectKey { dst, src, key } => writeln!(
                f,
                "{} = test has_key({}, {key:?})",
                fmt_val(*dst),
                fmt_val(*src)
            )?,
            InstKind::TestRange {
                dst,
                src,
                start,
                end,
                kind,
            } => writeln!(
                f,
                "{} = test {} in {start}{}{end}",
                fmt_val(*dst),
                fmt_val(*src),
                fmt_range_kind(*kind)
            )?,
            InstKind::ListIndex { dst, list, index } => {
                writeln!(f, "{} = {}[{index}]", fmt_val(*dst), fmt_val(*list))?
            }
            InstKind::ListSlice {
                dst,
                list,
                skip_head,
                skip_tail,
            } => writeln!(
                f,
                "{} = {}[{skip_head}..-{skip_tail}]",
                fmt_val(*dst),
                fmt_val(*list)
            )?,
            InstKind::ObjectGet { dst, object, key } => {
                writeln!(f, "{} = {}.{key}", fmt_val(*dst), fmt_val(*object))?
            }

            // Closures
            InstKind::MakeClosure {
                dst,
                body,
                captures,
            } => writeln!(
                f,
                "{} = closure {} [{}]",
                fmt_val(*dst),
                fmt_label(*body),
                fmt_vals(captures)
            )?,
            InstKind::CallClosure { dst, closure, args } => writeln!(
                f,
                "{} = call_closure {}({})",
                fmt_val(*dst),
                fmt_val(*closure),
                fmt_vals(args)
            )?,

            // Iteration
            InstKind::IterInit { dst, src } => {
                writeln!(f, "{} = iter_init {}", fmt_val(*dst), fmt_val(*src))?
            }
            InstKind::IterNext {
                dst_value,
                dst_done,
                iter,
            } => writeln!(
                f,
                "{}, {} = iter_next {}",
                fmt_val(*dst_value),
                fmt_val(*dst_done),
                fmt_val(*iter)
            )?,

            // Control flow
            InstKind::BlockLabel { label, params } => {
                if params.is_empty() {
                    writeln!(f, "{}:", fmt_label(*label))?
                } else {
                    let params_str = params
                        .iter()
                        .map(|v| {
                            let ty = body.val_types.get(v).map(|t| format!("{t}")).unwrap_or_else(|| "?".into());
                            format!("{}: {ty}", fmt_val(*v))
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(f, "{}({params_str}):", fmt_label(*label))?
                }
            }
            InstKind::Jump { label, args } => {
                if args.is_empty() {
                    writeln!(f, "jump {}", fmt_label(*label))?
                } else {
                    writeln!(f, "jump {}({})", fmt_label(*label), fmt_vals(args))?
                }
            }
            InstKind::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let then_str = if then_args.is_empty() {
                    fmt_label(*then_label)
                } else {
                    format!("{}({})", fmt_label(*then_label), fmt_vals(then_args))
                };
                let else_str = if else_args.is_empty() {
                    fmt_label(*else_label)
                } else {
                    format!("{}({})", fmt_label(*else_label), fmt_vals(else_args))
                };
                writeln!(
                    f,
                    "jump_if {} then {} else {}",
                    fmt_val(*cond),
                    then_str,
                    else_str
                )?
            }
            InstKind::Return(r) => writeln!(f, "return {}", fmt_val(*r))?,
            InstKind::Nop => writeln!(f, "nop")?,
        }
    }

    // Print value types with origin names.
    if !body.val_types.is_empty() {
        writeln!(f)?;
        let mut entries: Vec<_> = body.val_types.iter().collect();
        entries.sort_by_key(|(v, _)| v.0);
        for (val, ty) in entries {
            let origin = body.debug.label(*val);
            writeln!(f, "{indent}  ; {} ({origin}) : {ty}", fmt_val(*val))?;
        }
    }

    Ok(())
}

impl fmt::Display for MirModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Text constants table.
        if !self.texts.is_empty() {
            writeln!(f, "=== texts ===")?;
            for (i, text) in self.texts.iter().enumerate() {
                writeln!(f, "  T{i} = {text:?}")?;
            }
            writeln!(f)?;
        }

        writeln!(f, "=== main ===")?;
        write_body(f, &self.main, "  ")?;

        let mut labels: Vec<_> = self.closures.keys().collect();
        labels.sort_by_key(|l| l.0);
        for label in labels {
            let closure = &self.closures[label];
            write_closure(f, *label, closure)?;
        }

        Ok(())
    }
}

fn write_closure(f: &mut fmt::Formatter<'_>, label: Label, closure: &ClosureBody) -> fmt::Result {
    writeln!(f)?;
    write!(f, "=== closure {} (", fmt_label(label))?;
    for (i, name) in closure.param_names.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{name}")?;
    }
    write!(f, ")")?;
    if !closure.capture_names.is_empty() {
        write!(f, " [captures: {}]", closure.capture_names.join(", "))?;
    }
    writeln!(f, " ===")?;
    write_body(f, &closure.body, "  ")?;
    Ok(())
}

impl fmt::Display for MirBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_body(f, self, "")
    }
}

/// Convenience: dump a MirModule to a String.
pub fn dump(module: &MirModule) -> String {
    module.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::Ty;
    use std::collections::{BTreeMap, HashMap};

    fn compile_and_dump(
        source: &str,
        storage: HashMap<String, Ty>,
        externs: HashMap<String, (Vec<Ty>, Ty)>,
    ) -> String {
        let template = acvus_ast::parse(source).expect("parse failed");
        let (module, _) = crate::compile(&template, storage, externs).expect("compile failed");
        dump(&module)
    }

    #[test]
    fn print_text_only() {
        let out = compile_and_dump("hello world", HashMap::new(), HashMap::new());
        assert!(out.contains("T0 = \"hello world\""));
        assert!(out.contains("emit_text T0"));
    }

    #[test]
    fn print_string_emit() {
        let out = compile_and_dump(r#"{{ "hello" }}"#, HashMap::new(), HashMap::new());
        assert!(out.contains("r0 = const \"hello\""));
        assert!(out.contains("emit r0"));
    }

    #[test]
    fn print_arithmetic() {
        let storage = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
        let out = compile_and_dump(
            "{{ x = $a + $b }}{{ x | to_string }}{{_}}{{/}}",
            storage,
            HashMap::new(),
        );
        assert!(out.contains("+"));
        assert!(out.contains("call to_string"));
    }

    #[test]
    fn print_match_block() {
        let storage = HashMap::from([("name".into(), Ty::String)]);
        // Use a literal pattern to trigger full match block with iteration.
        let out = compile_and_dump(
            r#"{{ true = $name == "test" }}matched{{/}}"#,
            storage,
            HashMap::new(),
        );
        assert!(out.contains("iter_init"));
        assert!(out.contains("iter_next"));
        assert!(out.contains("jump_if"));
        assert!(out.contains("emit_text T"));
    }

    #[test]
    fn print_closure() {
        let storage = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
        let out = compile_and_dump(
            "{{ x = $items | filter(x -> x != 0) }}{{ x | to_string }}{{_}}{{/}}",
            storage,
            HashMap::new(),
        );
        assert!(out.contains("closure L"));
        assert!(out.contains("=== closure"));
        assert!(out.contains("!="));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_async_call() {
        let externs = HashMap::from([("fetch".into(), (vec![Ty::Int], Ty::String))]);
        let out = compile_and_dump(
            "{{ x = fetch(1) }}{{ x }}{{_}}{{/}}",
            HashMap::new(),
            externs,
        );
        assert!(out.contains("async_call fetch"));
        assert!(out.contains("await"));
    }

    #[test]
    fn print_object_field() {
        let storage = HashMap::from([(
            "user".into(),
            Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("age".into(), Ty::Int),
            ])),
        )]);
        let out = compile_and_dump("{{ $user.name }}", storage, HashMap::new());
        assert!(out.contains(".name"));
    }

    #[test]
    fn print_storage_write() {
        let storage = HashMap::from([("count".into(), Ty::Int)]);
        let out = compile_and_dump("{{ $count = 42 }}", storage, HashMap::new());
        assert!(out.contains("storage_store $count"));
    }

    #[test]
    fn snapshot_full_example() {
        let storage = HashMap::from([(
            "users".into(),
            Ty::List(Box::new(Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
            ])))),
        )]);
        let out = compile_and_dump(
            r#"{{ { name, } = $users }}{{ name }}{{/}}"#,
            storage,
            HashMap::new(),
        );
        // Just verify it doesn't panic and produces reasonable output.
        assert!(out.contains("=== main ==="));
        assert!(out.contains("iter_init"));
        assert!(out.contains("emit"));
    }
}
