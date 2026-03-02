use std::collections::HashMap;
use std::fmt;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};

use crate::ir::{ClosureBody, InstKind, Label, MirBody, MirModule, ValueId};

fn fmt_val(r: ValueId) -> String {
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
        Literal::Byte(b) => format!("0x{b:02x}"),
        Literal::List(elems) => {
            let items: Vec<String> = elems.iter().map(fmt_literal).collect();
            format!("[{}]", items.join(", "))
        }
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
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::Xor => "^",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::Mod => "%",
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

fn fmt_use(r: ValueId, consts: &HashMap<ValueId, &Literal>) -> String {
    match consts.get(&r) {
        Some(lit) => format!("{} ({})", fmt_literal(lit), fmt_val(r)),
        None => fmt_val(r),
    }
}

fn fmt_uses(regs: &[ValueId], consts: &HashMap<ValueId, &Literal>) -> String {
    regs.iter()
        .map(|r| fmt_use(*r, consts))
        .collect::<Vec<_>>()
        .join(", ")
}

fn write_body(f: &mut fmt::Formatter<'_>, body: &MirBody, indent: &str) -> fmt::Result {
    let consts: HashMap<ValueId, &Literal> = body
        .insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::Const { dst, value } => Some((*dst, value)),
            _ => None,
        })
        .collect();

    for (i, inst) in body.insts.iter().enumerate() {
        // Constants are shown inline at use sites; skip their definition lines.
        if matches!(&inst.kind, InstKind::Const { .. }) {
            continue;
        }

        let is_label = matches!(&inst.kind, InstKind::BlockLabel { .. });
        // Fixed-width index column, then content indent for non-labels.
        if is_label {
            write!(f, "{indent}{i:>4} │ ")?;
        } else {
            write!(f, "{indent}{i:>4} │   ")?;
        }

        match &inst.kind {
            // Output
            InstKind::Yield(r) => writeln!(f, "yield {}", fmt_use(*r, &consts))?,

            // Constants / variables
            InstKind::Const { .. } => unreachable!(),
            InstKind::ContextLoad { dst, name } => {
                writeln!(f, "{} = context_load @{name}", fmt_val(*dst))?
            }
            InstKind::VarLoad { dst, name } => {
                writeln!(f, "{} = var_load ${name}", fmt_val(*dst))?
            }
            InstKind::VarStore { name, src } => {
                writeln!(f, "var_store ${name} = {}", fmt_use(*src, &consts))?
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
                fmt_use(*left, &consts),
                fmt_binop(*op),
                fmt_use(*right, &consts)
            )?,
            InstKind::UnaryOp { dst, op, operand } => writeln!(
                f,
                "{} = {}{}",
                fmt_val(*dst),
                fmt_unaryop(*op),
                fmt_use(*operand, &consts)
            )?,
            InstKind::FieldGet { dst, object, field } => {
                writeln!(f, "{} = {}.{field}", fmt_val(*dst), fmt_use(*object, &consts))?
            }

            // Calls
            InstKind::Call { dst, func, args } => {
                writeln!(
                    f,
                    "{} = call {func}({})",
                    fmt_val(*dst),
                    fmt_uses(args, &consts)
                )?
            }
            InstKind::AsyncCall { dst, func, args } => writeln!(
                f,
                "{} = async_call {func}({})",
                fmt_val(*dst),
                fmt_uses(args, &consts)
            )?,
            InstKind::Await { dst, src } => {
                writeln!(f, "{} = await {}", fmt_val(*dst), fmt_use(*src, &consts))?
            }

            // Composite constructors
            InstKind::MakeList { dst, elements } => {
                writeln!(
                    f,
                    "{} = list [{}]",
                    fmt_val(*dst),
                    fmt_uses(elements, &consts)
                )?
            }
            InstKind::MakeObject { dst, fields } => {
                let fields_str: String = fields
                    .iter()
                    .map(|(k, r)| format!("{k}: {}", fmt_use(*r, &consts)))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(f, "{} = object {{{fields_str}}}", fmt_val(*dst))?
            }
            InstKind::MakeTuple { dst, elements } => {
                writeln!(
                    f,
                    "{} = tuple ({})",
                    fmt_val(*dst),
                    fmt_uses(elements, &consts)
                )?
            }
            InstKind::TupleIndex { dst, tuple, index } => {
                writeln!(
                    f,
                    "{} = {}.{index}",
                    fmt_val(*dst),
                    fmt_use(*tuple, &consts)
                )?
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
                fmt_use(*start, &consts),
                fmt_range_kind(*kind),
                fmt_use(*end, &consts)
            )?,

            // Pattern matching
            InstKind::TestLiteral { dst, src, value } => writeln!(
                f,
                "{} = test {} == {}",
                fmt_val(*dst),
                fmt_use(*src, &consts),
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
                    fmt_use(*src, &consts)
                )?
            }
            InstKind::TestObjectKey { dst, src, key } => writeln!(
                f,
                "{} = test has_key({}, {key:?})",
                fmt_val(*dst),
                fmt_use(*src, &consts)
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
                fmt_use(*src, &consts),
                fmt_range_kind(*kind)
            )?,
            InstKind::ListIndex { dst, list, index } => {
                writeln!(
                    f,
                    "{} = {}[{index}]",
                    fmt_val(*dst),
                    fmt_use(*list, &consts)
                )?
            }
            InstKind::ListGet { dst, list, index } => writeln!(
                f,
                "{} = {}[{}]",
                fmt_val(*dst),
                fmt_use(*list, &consts),
                fmt_use(*index, &consts)
            )?,
            InstKind::ListSlice {
                dst,
                list,
                skip_head,
                skip_tail,
            } => writeln!(
                f,
                "{} = {}[{skip_head}..-{skip_tail}]",
                fmt_val(*dst),
                fmt_use(*list, &consts)
            )?,
            InstKind::ObjectGet { dst, object, key } => {
                writeln!(
                    f,
                    "{} = {}.{key}",
                    fmt_val(*dst),
                    fmt_use(*object, &consts)
                )?
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
                fmt_uses(captures, &consts)
            )?,
            InstKind::CallClosure { dst, closure, args } => writeln!(
                f,
                "{} = call_closure {}({})",
                fmt_val(*dst),
                fmt_use(*closure, &consts),
                fmt_uses(args, &consts)
            )?,

            // Iteration
            InstKind::IterInit { dst, src } => {
                writeln!(
                    f,
                    "{} = iter_init {}",
                    fmt_val(*dst),
                    fmt_use(*src, &consts)
                )?
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
                fmt_use(*iter, &consts)
            )?,

            // Control flow
            InstKind::BlockLabel { label, params } => {
                if params.is_empty() {
                    writeln!(f, "{}:", fmt_label(*label))?
                } else {
                    let params_str = params
                        .iter()
                        .map(|v| {
                            let ty = body
                                .val_types
                                .get(v)
                                .map(|t| format!("{t}"))
                                .unwrap_or_else(|| "?".into());
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
                    writeln!(
                        f,
                        "jump {}({})",
                        fmt_label(*label),
                        fmt_uses(args, &consts)
                    )?
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
                    format!(
                        "{}({})",
                        fmt_label(*then_label),
                        fmt_uses(then_args, &consts)
                    )
                };
                let else_str = if else_args.is_empty() {
                    fmt_label(*else_label)
                } else {
                    format!(
                        "{}({})",
                        fmt_label(*else_label),
                        fmt_uses(else_args, &consts)
                    )
                };
                writeln!(
                    f,
                    "jump_if {} then {} else {}",
                    fmt_use(*cond, &consts),
                    then_str,
                    else_str
                )?
            }
            InstKind::Return(r) => writeln!(f, "return {}", fmt_use(*r, &consts))?,
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
    use crate::extern_module::{ExternModule, ExternRegistry};
    use crate::ty::Ty;
    use std::collections::{BTreeMap, HashMap};

    fn compile_and_dump(
        source: &str,
        context: HashMap<String, Ty>,
        registry: &crate::extern_module::ExternRegistry,
    ) -> String {
        let template = acvus_ast::parse(source).expect("parse failed");
        let (module, _) = crate::compile(&template, context, registry).expect("compile failed");
        dump(&module)
    }

    #[test]
    fn print_text_only() {
        let out = compile_and_dump("hello world", HashMap::new(), &ExternRegistry::new());
        assert!(!out.contains("const"));
        assert!(out.contains("yield \"hello world\" (r"));
    }

    #[test]
    fn print_string_emit() {
        let out = compile_and_dump(r#"{{ "hello" }}"#, HashMap::new(), &ExternRegistry::new());
        assert!(!out.contains("const"));
        assert!(out.contains("yield \"hello\" (r0)"));
    }

    #[test]
    fn print_arithmetic() {
        let context = HashMap::from([("a".into(), Ty::Int), ("b".into(), Ty::Int)]);
        let out = compile_and_dump(
            "{{ x = @a + @b }}{{ x | to_string }}{{_}}{{/}}",
            context,
            &ExternRegistry::new(),
        );
        assert!(out.contains("+"));
        assert!(out.contains("call to_string"));
    }

    #[test]
    fn print_match_block() {
        let context = HashMap::from([("name".into(), Ty::String)]);
        let out = compile_and_dump(
            r#"{{ true = @name == "test" }}matched{{/}}"#,
            context,
            &ExternRegistry::new(),
        );
        assert!(!out.contains("iter_init"));
        assert!(!out.contains("iter_next"));
        assert!(out.contains("jump_if"));
        assert!(out.contains("yield"));
    }

    #[test]
    fn print_closure() {
        let context = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
        let out = compile_and_dump(
            "{{ x = @items | filter(x -> x != 0) }}{{ x | len | to_string }}{{_}}{{/}}",
            context,
            &ExternRegistry::new(),
        );
        assert!(out.contains("closure L"));
        assert!(out.contains("=== closure"));
        assert!(out.contains("!="));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_async_call() {
        let mut ext = ExternModule::new("test");
        ext.add_fn("fetch", vec![Ty::Int], Ty::String, false);
        let mut registry = ExternRegistry::new();
        registry.register(&ext);
        let out = compile_and_dump(
            "{{ x = fetch(1) }}{{ x }}{{_}}{{/}}",
            HashMap::new(),
            &registry,
        );
        assert!(out.contains("async_call fetch"));
        assert!(out.contains("await"));
    }

    #[test]
    fn print_object_field() {
        let context = HashMap::from([(
            "user".into(),
            Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("age".into(), Ty::Int),
            ])),
        )]);
        let out = compile_and_dump("{{ @user.name }}", context, &ExternRegistry::new());
        assert!(out.contains(".name"));
    }

    #[test]
    fn print_var_write() {
        let out = compile_and_dump("{{ $count = 42 }}", HashMap::new(), &ExternRegistry::new());
        assert!(out.contains("var_store $count"));
    }

    #[test]
    fn snapshot_full_example() {
        let context = HashMap::from([(
            "users".into(),
            Ty::List(Box::new(Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
            ])))),
        )]);
        let out = compile_and_dump(
            r#"{{ { name, } in @users }}{{ name }}{{/}}"#,
            context,
            &ExternRegistry::new(),
        );
        assert!(out.contains("=== main ==="));
        assert!(out.contains("iter_init"));
        assert!(out.contains("yield"));
    }
}
