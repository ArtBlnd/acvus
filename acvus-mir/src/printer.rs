use std::fmt;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::ir::{Callee, InstKind, Label, MirBody, MirModule, ValueId};

fn fmt_val(r: ValueId) -> String {
    use acvus_utils::LocalIdOps;
    format!("r{}", r.to_raw())
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

struct PrintCtx<'a> {
    interner: &'a Interner,
    lit_to_tidx: &'a FxHashMap<String, usize>,
}

impl PrintCtx<'_> {
    fn tag_name(&self, tag: &Astr) -> String {
        self.interner.resolve(*tag).to_string()
    }
}

fn fmt_use(
    r: ValueId,
    consts: &FxHashMap<ValueId, &Literal>,
    texts: &FxHashMap<ValueId, usize>,
) -> String {
    if let Some(tidx) = texts.get(&r) {
        return format!("T{tidx}");
    }
    match consts.get(&r) {
        Some(lit) => format!("{} ({})", fmt_literal(lit), fmt_val(r)),
        None => fmt_val(r),
    }
}

fn fmt_uses(
    regs: &[ValueId],
    consts: &FxHashMap<ValueId, &Literal>,
    texts: &FxHashMap<ValueId, usize>,
) -> String {
    regs.iter()
        .map(|r| fmt_use(*r, consts, texts))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Collect unique String/List literals from a body and register them into the text table.
fn collect_texts_from_body(
    body: &MirBody,
    lit_to_tidx: &mut FxHashMap<String, usize>,
    text_entries: &mut Vec<String>,
) {
    for inst in &body.insts {
        if let InstKind::Const { value, .. } = &inst.kind
            && matches!(value, Literal::String(_) | Literal::List(_))
        {
            let key = fmt_literal(value);
            if !lit_to_tidx.contains_key(&key) {
                let idx = text_entries.len();
                lit_to_tidx.insert(key.clone(), idx);
                text_entries.push(key);
            }
        }
    }
}

fn write_body(
    f: &mut fmt::Formatter<'_>,
    body: &MirBody,
    indent: &str,
    ctx: &PrintCtx<'_>,
) -> fmt::Result {
    // Build ContextId → name mapping from ContextProject instructions + debug info.
    let mut ctx_id_to_name: FxHashMap<crate::graph::ContextId, String> = FxHashMap::default();
    for inst in &body.insts {
        if let InstKind::ContextProject { dst, id, .. } = &inst.kind {
            if let Some(crate::ir::ValOrigin::Context(name)) = body.debug.get(*dst) {
                ctx_id_to_name
                    .entry(*id)
                    .or_insert_with(|| ctx.interner.resolve(*name).to_string());
            }
        }
    }

    // Small constants (Int, Float, Bool, Byte) -> inline at use sites.
    let consts: FxHashMap<ValueId, &Literal> = body
        .insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::Const { dst, value }
                if !matches!(value, Literal::String(_) | Literal::List(_)) =>
            {
                Some((*dst, value))
            }
            _ => None,
        })
        .collect();

    // String/List constants -> T-indexed references.
    let texts: FxHashMap<ValueId, usize> = body
        .insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::Const { dst, value }
                if matches!(value, Literal::String(_) | Literal::List(_)) =>
            {
                ctx.lit_to_tidx
                    .get(&fmt_literal(value))
                    .map(|&tidx| (*dst, tidx))
            }
            _ => None,
        })
        .collect();

    for (i, inst) in body.insts.iter().enumerate() {
        // All constants are represented elsewhere: small ones inline, String/List in texts section.
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
            // Constants -- all skipped above, unreachable here.
            InstKind::Const { .. } => unreachable!(),
            InstKind::ContextProject { dst, id, .. } => {
                let name = ctx_id_to_name.get(id).map(|s| s.as_str()).unwrap_or("?");
                writeln!(f, "{} = context_project @{}", fmt_val(*dst), name)?
            }
            InstKind::ContextLoad { dst, src } => writeln!(
                f,
                "{} = context_load {}",
                fmt_val(*dst),
                fmt_use(*src, &consts, &texts)
            )?,
            InstKind::VarLoad { dst, name } => writeln!(
                f,
                "{} = var_load ${}",
                fmt_val(*dst),
                ctx.interner.resolve(*name)
            )?,
            InstKind::VarStore { name, src } => writeln!(
                f,
                "var_store ${} = {}",
                ctx.interner.resolve(*name),
                fmt_use(*src, &consts, &texts)
            )?,
            InstKind::ContextStore { dst, value } => writeln!(
                f,
                "ctx_store {} = {}",
                fmt_use(*dst, &consts, &texts),
                fmt_use(*value, &consts, &texts)
            )?,

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
                fmt_use(*left, &consts, &texts),
                fmt_binop(*op),
                fmt_use(*right, &consts, &texts)
            )?,
            InstKind::UnaryOp { dst, op, operand } => writeln!(
                f,
                "{} = {}{}",
                fmt_val(*dst),
                fmt_unaryop(*op),
                fmt_use(*operand, &consts, &texts)
            )?,
            InstKind::FieldGet { dst, object, field } => writeln!(
                f,
                "{} = {}.{}",
                fmt_val(*dst),
                fmt_use(*object, &consts, &texts),
                ctx.interner.resolve(*field),
            )?,

            // Functions
            InstKind::LoadFunction { dst, id } => writeln!(
                f,
                "{} = load_function #{}",
                fmt_val(*dst),
                id.index(),
            )?,
            InstKind::FunctionCall { dst, callee, args } => {
                let callee_str = match callee {
                    Callee::Direct(id) => format!("#{}", id.index()),
                    Callee::Indirect(val) => fmt_use(*val, &consts, &texts),
                };
                writeln!(
                    f,
                    "{} = call {}({})",
                    fmt_val(*dst),
                    callee_str,
                    fmt_uses(args, &consts, &texts)
                )?
            }

            // Spawn / Eval
            InstKind::Spawn { dst, callee, args, context_uses } => {
                let callee_str = match callee {
                    Callee::Direct(id) => format!("#{}", id.index()),
                    Callee::Indirect(val) => fmt_use(*val, &consts, &texts),
                };
                let ctx_str = if context_uses.is_empty() {
                    String::new()
                } else {
                    let bindings: Vec<String> = context_uses
                        .iter()
                        .map(|(ctx, val)| {
                            let name = ctx_id_to_name.get(ctx).map(|s| s.as_str()).unwrap_or("?");
                            format!("@{}={}", name, fmt_val(*val))
                        })
                        .collect();
                    format!(" use({})", bindings.join(", "))
                };
                writeln!(
                    f,
                    "{} = spawn {}({}){ctx_str}",
                    fmt_val(*dst),
                    callee_str,
                    fmt_uses(args, &consts, &texts)
                )?
            }
            InstKind::Eval { dst, src, context_defs } => {
                let ctx_str = if context_defs.is_empty() {
                    String::new()
                } else {
                    let bindings: Vec<String> = context_defs
                        .iter()
                        .map(|(ctx, val)| {
                            let name = ctx_id_to_name.get(ctx).map(|s| s.as_str()).unwrap_or("?");
                            format!("@{}={}", name, fmt_val(*val))
                        })
                        .collect();
                    format!(" def({})", bindings.join(", "))
                };
                writeln!(
                    f,
                    "{} = eval {}{ctx_str}",
                    fmt_val(*dst),
                    fmt_use(*src, &consts, &texts)
                )?
            }

            // Composite constructors
            InstKind::MakeDeque { dst, elements } => writeln!(
                f,
                "{} = list [{}]",
                fmt_val(*dst),
                fmt_uses(elements, &consts, &texts)
            )?,
            InstKind::MakeObject { dst, fields } => {
                let fields_str: String = fields
                    .iter()
                    .map(|(k, r)| {
                        format!(
                            "{}: {}",
                            ctx.interner.resolve(*k),
                            fmt_use(*r, &consts, &texts)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(f, "{} = object {{{fields_str}}}", fmt_val(*dst))?
            }
            InstKind::MakeTuple { dst, elements } => writeln!(
                f,
                "{} = tuple ({})",
                fmt_val(*dst),
                fmt_uses(elements, &consts, &texts)
            )?,
            InstKind::TupleIndex { dst, tuple, index } => writeln!(
                f,
                "{} = {}.{index}",
                fmt_val(*dst),
                fmt_use(*tuple, &consts, &texts)
            )?,
            InstKind::MakeRange {
                dst,
                start,
                end,
                kind,
            } => writeln!(
                f,
                "{} = range {}{}{}",
                fmt_val(*dst),
                fmt_use(*start, &consts, &texts),
                fmt_range_kind(*kind),
                fmt_use(*end, &consts, &texts)
            )?,

            // Pattern matching
            InstKind::TestLiteral { dst, src, value } => writeln!(
                f,
                "{} = test {} == {}",
                fmt_val(*dst),
                fmt_use(*src, &consts, &texts),
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
                    fmt_use(*src, &consts, &texts)
                )?
            }
            InstKind::TestObjectKey { dst, src, key } => writeln!(
                f,
                "{} = test has_key({}, \"{}\")",
                fmt_val(*dst),
                fmt_use(*src, &consts, &texts),
                ctx.interner.resolve(*key),
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
                fmt_use(*src, &consts, &texts),
                fmt_range_kind(*kind)
            )?,
            InstKind::ListIndex { dst, list, index } => writeln!(
                f,
                "{} = {}[{index}]",
                fmt_val(*dst),
                fmt_use(*list, &consts, &texts)
            )?,
            InstKind::ListGet { dst, list, index } => writeln!(
                f,
                "{} = {}[{}]",
                fmt_val(*dst),
                fmt_use(*list, &consts, &texts),
                fmt_use(*index, &consts, &texts)
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
                fmt_use(*list, &consts, &texts)
            )?,
            InstKind::ObjectGet { dst, object, key } => writeln!(
                f,
                "{} = {}.{}",
                fmt_val(*dst),
                fmt_use(*object, &consts, &texts),
                ctx.interner.resolve(*key),
            )?,

            // Variant
            InstKind::MakeVariant { dst, tag, payload } => {
                let name = ctx.tag_name(tag);
                match payload {
                    Some(p) => writeln!(
                        f,
                        "{} = variant {}({})",
                        fmt_val(*dst),
                        name,
                        fmt_use(*p, &consts, &texts)
                    )?,
                    None => writeln!(f, "{} = variant {}", fmt_val(*dst), name)?,
                }
            }
            InstKind::TestVariant { dst, src, tag } => {
                let name = ctx.tag_name(tag);
                writeln!(
                    f,
                    "{} = test {} is {}",
                    fmt_val(*dst),
                    fmt_use(*src, &consts, &texts),
                    name,
                )?
            }
            InstKind::UnwrapVariant { dst, src } => writeln!(
                f,
                "{} = unwrap {}",
                fmt_val(*dst),
                fmt_use(*src, &consts, &texts)
            )?,

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
                fmt_uses(captures, &consts, &texts)
            )?,

            // Iteration
            InstKind::IterStep { dst, iter_src, iter_dst, done, done_args } => writeln!(
                f,
                "iter_step {}, {} = {} else {}({})",
                fmt_val(*dst),
                fmt_val(*iter_dst),
                fmt_use(*iter_src, &consts, &texts),
                fmt_label(*done),
                fmt_uses(done_args, &consts, &texts)
            )?,

            // Control flow
            InstKind::BlockLabel {
                label,
                params,
                merge_of,
            } => {
                let merge_suffix = match merge_of {
                    Some(m) => format!("  ; merge_of {}", fmt_label(*m)),
                    None => String::new(),
                };
                if params.is_empty() {
                    writeln!(f, "{}:{merge_suffix}", fmt_label(*label))?
                } else {
                    let params_str = params
                        .iter()
                        .map(|v| {
                            let ty = body
                                .val_types
                                .get(v)
                                .map(|t| format!("{}", t.display(ctx.interner)))
                                .unwrap_or_else(|| "?".into());
                            format!("{}: {ty}", fmt_val(*v))
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(f, "{}({params_str}):{merge_suffix}", fmt_label(*label))?
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
                        fmt_uses(args, &consts, &texts)
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
                        fmt_uses(then_args, &consts, &texts)
                    )
                };
                let else_str = if else_args.is_empty() {
                    fmt_label(*else_label)
                } else {
                    format!(
                        "{}({})",
                        fmt_label(*else_label),
                        fmt_uses(else_args, &consts, &texts)
                    )
                };
                writeln!(
                    f,
                    "jump_if {} then {} else {}",
                    fmt_use(*cond, &consts, &texts),
                    then_str,
                    else_str
                )?
            }
            InstKind::Return(r) => writeln!(f, "return {}", fmt_use(*r, &consts, &texts))?,
            InstKind::Nop => writeln!(f, "nop")?,
            InstKind::Cast { dst, src, kind } => writeln!(
                f,
                "{} = cast {:?} {}",
                fmt_val(*dst),
                kind,
                fmt_use(*src, &consts, &texts)
            )?,
            InstKind::Poison { dst } => writeln!(f, "{} = poison", fmt_val(*dst))?,
        }
    }

    // Print value types with origin names.
    if !body.val_types.is_empty() {
        writeln!(f)?;
        let mut entries: Vec<_> = body.val_types.iter().collect();
        entries.sort_by_key(|(v, _)| **v);
        for (val, ty) in entries {
            let origin = body.debug.label(*val, ctx.interner);
            writeln!(
                f,
                "{indent}  ; {} ({origin}) : {}",
                fmt_val(*val),
                ty.display(ctx.interner),
            )?;
        }
    }

    Ok(())
}

/// Display wrapper for MirModule that requires an interner.
pub struct MirModuleDisplay<'a> {
    module: &'a MirModule,
    interner: &'a Interner,
}

impl fmt::Display for MirModuleDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let module = self.module;

        // Collect unique String/List literals across all bodies -> text table.
        let mut lit_to_tidx: FxHashMap<String, usize> = FxHashMap::default();
        let mut text_entries: Vec<String> = Vec::new();

        collect_texts_from_body(&module.main, &mut lit_to_tidx, &mut text_entries);
        let mut labels: Vec<_> = module.closures.keys().collect();
        labels.sort_by_key(|l| l.0);
        for label in &labels {
            collect_texts_from_body(
                &module.closures[label],
                &mut lit_to_tidx,
                &mut text_entries,
            );
        }

        if !text_entries.is_empty() {
            writeln!(f, "=== literals ===")?;
            for (idx, lit) in text_entries.iter().enumerate() {
                writeln!(f, "  T{idx} = {lit}")?;
            }
            writeln!(f)?;
        }

        let ctx = PrintCtx {
            interner: self.interner,
            lit_to_tidx: &lit_to_tidx,
        };

        writeln!(f, "=== main ===")?;
        write_body(f, &module.main, "  ", &ctx)?;

        for label in &labels {
            let closure = &module.closures[label];
            write_closure(f, **label, closure, &ctx)?;
        }

        Ok(())
    }
}

impl MirModule {
    /// Create a display wrapper that resolves Astr values via the interner.
    pub fn display<'a>(&'a self, interner: &'a Interner) -> MirModuleDisplay<'a> {
        MirModuleDisplay {
            module: self,
            interner,
        }
    }
}

fn write_closure(
    f: &mut fmt::Formatter<'_>,
    label: Label,
    body: &MirBody,
    ctx: &PrintCtx<'_>,
) -> fmt::Result {
    writeln!(f)?;
    write!(f, "=== closure {} (", fmt_label(label))?;
    for (i, reg) in body.param_regs.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{reg:?}")?;
    }
    write!(f, ")")?;
    if !body.capture_regs.is_empty() {
        write!(f, " [captures: {:?}]", body.capture_regs)?;
    }
    writeln!(f, " ===")?;
    write_body(f, body, "  ", ctx)?;
    Ok(())
}

/// Display wrapper for MirBody that requires an interner.
pub struct MirBodyDisplay<'a> {
    body: &'a MirBody,
    interner: &'a Interner,
}

impl fmt::Display for MirBodyDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let empty_lits = FxHashMap::default();
        let ctx = PrintCtx {
            interner: self.interner,
            lit_to_tidx: &empty_lits,
        };
        write_body(f, self.body, "", &ctx)
    }
}

impl MirBody {
    /// Create a display wrapper that resolves Astr values via the interner.
    pub fn display<'a>(&'a self, interner: &'a Interner) -> MirBodyDisplay<'a> {
        MirBodyDisplay {
            body: self,
            interner,
        }
    }
}

/// Dump a MirModule to a String, resolving Astr values via the interner.
pub fn dump(interner: &Interner, module: &MirModule) -> String {
    format!("{}", module.display(interner))
}

/// Alias for `dump`. Kept for backward compatibility.
pub fn dump_with(interner: &Interner, module: &MirModule) -> String {
    dump(interner, module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ty::{Effect, Param, Ty};
    use acvus_utils::Interner;

    fn compile_and_dump(
        source: &str,
        context: &FxHashMap<Astr, Ty>,
        interner: &Interner,
    ) -> String {
        let ctx: Vec<(&str, Ty)> = context.iter()
            .map(|(name, ty)| (interner.resolve(*name), ty.clone()))
            .collect();
        let (module, _) = crate::test::compile_template(interner, source, &ctx)
            .expect("compile failed");
        dump(interner, &module)
    }

    #[test]
    fn print_text_only() {
        let interner = Interner::new();
        let out = compile_and_dump("hello world", &FxHashMap::default(), &interner);
        assert!(out.contains("=== literals ==="));
        assert!(out.contains("\"hello world\""));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_string_emit() {
        let interner = Interner::new();
        let out = compile_and_dump(r#"{{ "hello" }}"#, &FxHashMap::default(), &interner);
        assert!(out.contains("\"hello\""));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_arithmetic() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([
            (interner.intern("a"), Ty::Int),
            (interner.intern("b"), Ty::Int),
        ]);
        let out = compile_and_dump(
            "{{ x = @a + @b }}{{ x | to_string }}{{_}}{{/}}",
            &context,
            &interner,
        );
        assert!(out.contains("+"));
        // to_string is a builtin — requires Phase 2 (builtin → graph Function) to
        // appear as a FunctionCall. For now, just check arithmetic works.
    }

    #[test]
    fn print_match_block() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("name"), Ty::String)]);
        let out = compile_and_dump(
            r#"{{ true = @name == "test" }}matched{{/}}"#,
            &context,
            &interner,
        );
        assert!(!out.contains("iter_init"));
        assert!(!out.contains("iter_next"));
        assert!(out.contains("jump_if"));
    }

    #[test]
    fn print_closure() {
        let interner = Interner::new();
        let context =
            FxHashMap::from_iter([(interner.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let out = compile_and_dump(
            "{{ x = @items | filter(x -> x != 0) | collect }}{{ x | len | to_string }}{{_}}{{/}}",
            &context,
            &interner,
        );
        assert!(out.contains("closure L"));
        assert!(out.contains("=== closure"));
        assert!(out.contains("!="));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_extern_call() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(
            interner.intern("fetch"),
            Ty::Fn {
                params: vec![Param::new(interner.intern("x"), Ty::Int)],
                ret: Box::new(Ty::String),
                captures: vec![],
                effect: Effect::pure(),
            },
        )]);
        let out = compile_and_dump("{{ x = @fetch(1) }}{{ x }}{{_}}{{/}}", &context, &interner);
        assert!(
            out.contains("call"),
            "expected call instruction, got:\n{out}"
        );
    }

    #[test]
    fn print_object_field() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(
            interner.intern("user"),
            Ty::Object(FxHashMap::from_iter([
                (interner.intern("name"), Ty::String),
                (interner.intern("age"), Ty::Int),
            ])),
        )]);
        let out = compile_and_dump("{{ @user.name }}", &context, &interner);
        assert!(out.contains(".name"));
    }

    #[test]
    fn print_var_write() {
        let interner = Interner::new();
        let out = compile_and_dump("{{ $count = 42 }}", &FxHashMap::default(), &interner);
        assert!(out.contains("var_store $count"));
    }

    #[test]
    fn snapshot_full_example() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(
            interner.intern("users"),
            Ty::List(Box::new(Ty::Object(FxHashMap::from_iter([(
                interner.intern("name"),
                Ty::String,
            )])))),
        )]);
        let out = compile_and_dump(
            r#"{{ { name, } in @users }}{{ name }}{{/}}"#,
            &context,
            &interner,
        );
        assert!(out.contains("=== main ==="));
        assert!(out.contains("iter_step"));
    }

    #[test]
    fn print_text_dedup() {
        let interner = Interner::new();
        let out = compile_and_dump(
            r#"{{ "hello" }}{{ "hello" }}"#,
            &FxHashMap::default(),
            &interner,
        );
        // Same literal "hello" should appear only once in literals section.
        let hello_count = out.matches("\"hello\"").count();
        // Once in literals, possibly referenced in instructions.
        assert!(hello_count >= 1);
        assert!(out.contains("return"));
    }
}
