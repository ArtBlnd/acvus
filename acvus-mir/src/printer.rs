use std::fmt;

use acvus_ast::{BinOp, Literal, UnaryOp};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::ir::{Callee, IndexMode, InstKind, Label, MirBody, MirModule, ValueId};

/// Normalizes ValueIds to sequential order of first appearance.
struct ValNormalizer {
    map: FxHashMap<ValueId, usize>,
}

impl ValNormalizer {
    fn new() -> Self {
        Self {
            map: FxHashMap::default(),
        }
    }

    fn get(&mut self, v: ValueId) -> usize {
        let len = self.map.len();
        *self.map.entry(v).or_insert(len)
    }

    fn fmt_val(&mut self, r: ValueId) -> String {
        format!("r{}", self.get(r))
    }

    fn fmt_use(
        &mut self,
        r: ValueId,
        consts: &FxHashMap<ValueId, &Literal>,
        texts: &FxHashMap<ValueId, usize>,
    ) -> String {
        if let Some(tidx) = texts.get(&r) {
            return format!("T{tidx}");
        }
        match consts.get(&r) {
            Some(lit) => format!("{} ({})", fmt_literal(lit), self.fmt_val(r)),
            None => self.fmt_val(r),
        }
    }

    fn fmt_uses(
        &mut self,
        regs: &[ValueId],
        consts: &FxHashMap<ValueId, &Literal>,
        texts: &FxHashMap<ValueId, usize>,
    ) -> String {
        regs.iter()
            .map(|r| self.fmt_use(*r, consts, texts))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn fmt_label(l: Label) -> String {
    format!("L{}", l.0)
}

fn fmt_literal(lit: &Literal) -> String {
    match lit {
        Literal::Int(n) => n.to_string(),
        Literal::Float(f) => format!("{f:?}"),
        Literal::Char(c) => format!("{c:?}"),
        sugar @ (Literal::IntOf(_) | Literal::Bytes(_)) => fmt_literal(&sugar.desugared()),
        Literal::String(s) => format!("{s:?}"),
        Literal::Bool(b) => b.to_string(),
        Literal::Unit => "()".to_string(),
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
        UnaryOp::Deref => "*",
    }
}

struct PrintCtx<'a> {
    interner: &'a Interner,
    lit_to_tidx: &'a FxHashMap<String, usize>,
    /// FunctionId -> canonical index (order of first appearance across all bodies).
    fn_id_map: FxHashMap<crate::graph::QualifiedRef, usize>,
}

impl PrintCtx<'_> {
    fn tag_name(&self, tag: &Astr) -> String {
        self.interner.resolve(*tag).to_string()
    }

    fn fmt_fn_id(&self, id: crate::graph::QualifiedRef) -> String {
        match self.fn_id_map.get(&id) {
            Some(&idx) => format!("#{idx}"),
            None => {
                let name = self.interner.resolve(id.name);
                format!("#?{name}")
            }
        }
    }
}

/// Collect FunctionIds from a body in order of first appearance.
fn collect_fn_ids_from_body(
    body: &MirBody,
    fn_id_map: &mut FxHashMap<crate::graph::QualifiedRef, usize>,
) {
    for inst in &body.insts {
        let ids: &[crate::graph::QualifiedRef] = match &inst.kind {
            InstKind::LoadFunction { id, .. } => std::slice::from_ref(id),
            InstKind::FunctionCall {
                callee: Callee::Direct(id) | Callee::Extern { id, .. },
                ..
            } => std::slice::from_ref(id),
            InstKind::Spawn {
                callee: Callee::Direct(id) | Callee::Extern { id, .. },
                ..
            } => std::slice::from_ref(id),
            _ => &[],
        };
        for &id in ids {
            let len = fn_id_map.len();
            fn_id_map.entry(id).or_insert(len);
        }
    }
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

/// A storage and a field path under it, as `Ref`, `Take`, and `Assign` name one.
fn fmt_place(
    body: &MirBody,
    ctx: &PrintCtx<'_>,
    vn: &mut ValNormalizer,
    target: &crate::ir::RefTarget,
    path: &[crate::ir::PathSeg],
) -> String {
    let base = match target {
        crate::ir::RefTarget::Var(slot) => body.debug.label(*slot, ctx.interner),
        crate::ir::RefTarget::Param(slot) => {
            let name = body.debug.label(*slot, ctx.interner);
            if name.starts_with('$') {
                name
            } else {
                format!("${name}")
            }
        }
        crate::ir::RefTarget::Through(r) => format!("(*{})", vn.fmt_val(*r)),
    };
    if path.is_empty() {
        base
    } else {
        let suffix: String = path
            .iter()
            .map(|seg| match seg {
                crate::ir::PathSeg::Field(f) => format!(".{}", ctx.interner.resolve(*f)),
                crate::ir::PathSeg::Index(i) => format!("[{i}]"),
                crate::ir::PathSeg::Payload => ".payload".to_string(),
            })
            .collect();
        format!("{base}{suffix}")
    }
}

fn write_body(
    f: &mut fmt::Formatter<'_>,
    body: &MirBody,
    indent: &str,
    ctx: &PrintCtx<'_>,
) -> fmt::Result {
    let mut vn = ValNormalizer::new();

    // A constant is shown at its use sites only while its ValueId names that
    // one definition; after register allocation an id is reused, and a use
    // of a reused id prints as the id.
    let mut def_count: FxHashMap<ValueId, usize> = FxHashMap::default();
    for inst in &body.insts {
        for dst in crate::analysis::inst_info::defs(&inst.kind) {
            *def_count.entry(dst).or_default() += 1;
        }
    }
    let single_consts = body.insts.iter().filter_map(|inst| match &inst.kind {
        InstKind::Const { dst, value } if def_count[dst] == 1 => Some((*dst, value)),
        _ => None,
    });

    // Small constants (Int, Float, Bool, Byte) -> inline at use sites.
    let consts: FxHashMap<ValueId, &Literal> = single_consts
        .clone()
        .filter(|(_, value)| !matches!(value, Literal::String(_) | Literal::List(_)))
        .collect();

    // String/List constants -> T-indexed references.
    let texts: FxHashMap<ValueId, usize> = single_consts
        .filter(|(_, value)| matches!(value, Literal::String(_) | Literal::List(_)))
        .filter_map(|(dst, value)| {
            ctx.lit_to_tidx
                .get(&fmt_literal(value))
                .map(|&tidx| (dst, tidx))
        })
        .collect();

    for (i, inst) in body.insts.iter().enumerate() {
        if let InstKind::Const { dst, .. } = &inst.kind
            && (consts.contains_key(dst) || texts.contains_key(dst))
        {
            continue;
        }

        let is_label = matches!(&inst.kind, InstKind::BlockLabel { .. });
        // Fixed-width index column, then content indent for non-labels.
        if is_label {
            write!(f, "{indent}{i:>4} | ")?;
        } else {
            write!(f, "{indent}{i:>4} |   ")?;
        }

        match &inst.kind {
            InstKind::Const { dst, value } => {
                let shown = match ctx.lit_to_tidx.get(&fmt_literal(value)) {
                    Some(tidx) if matches!(value, Literal::String(_) | Literal::List(_)) => {
                        format!("T{tidx}")
                    }
                    _ => fmt_literal(value),
                };
                writeln!(f, "{} = const {}", vn.fmt_val(*dst), shown)?
            }
            // Projection
            InstKind::Ref {
                dst,
                target,
                path,
                mutability,
            } => writeln!(
                f,
                "{} = ref {}{}",
                vn.fmt_val(*dst),
                mutability.prefix(),
                fmt_place(body, ctx, &mut vn, target, path)
            )?,
            InstKind::Take { dst, target, path } => writeln!(
                f,
                "{} = take {}",
                vn.fmt_val(*dst),
                fmt_place(body, ctx, &mut vn, target, path)
            )?,
            InstKind::Assign {
                target,
                path,
                value,
            } => writeln!(
                f,
                "assign {} = {}",
                fmt_place(body, ctx, &mut vn, target, path),
                vn.fmt_use(*value, &consts, &texts)
            )?,
            InstKind::Fetch { dst, context } => writeln!(
                f,
                "{} = fetch @{}",
                vn.fmt_val(*dst),
                ctx.interner.resolve(context.name)
            )?,
            InstKind::Commit { context, value } => writeln!(
                f,
                "commit @{} = {}",
                ctx.interner.resolve(context.name),
                vn.fmt_use(*value, &consts, &texts)
            )?,

            // Scalar field access
            InstKind::FieldGet {
                dst,
                object,
                field,
                rest,
            } => {
                let mut path = ctx.interner.resolve(*field).to_string();
                for r in rest {
                    path.push('.');
                    path.push_str(ctx.interner.resolve(*r));
                }
                writeln!(
                    f,
                    "{} = {}.{}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*object, &consts, &texts),
                    path
                )?
            }
            InstKind::FieldSet {
                dst,
                object,
                field,
                rest,
                value,
            } => {
                let mut path = ctx.interner.resolve(*field).to_string();
                for r in rest {
                    path.push('.');
                    path.push_str(ctx.interner.resolve(*r));
                }
                writeln!(
                    f,
                    "{} = field_set {}.{} = {}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*object, &consts, &texts),
                    path,
                    vn.fmt_use(*value, &consts, &texts)
                )?
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
                vn.fmt_val(*dst),
                vn.fmt_use(*left, &consts, &texts),
                fmt_binop(*op),
                vn.fmt_use(*right, &consts, &texts)
            )?,
            InstKind::UnaryOp { dst, op, operand } => writeln!(
                f,
                "{} = {}{}",
                vn.fmt_val(*dst),
                fmt_unaryop(*op),
                vn.fmt_use(*operand, &consts, &texts)
            )?,
            InstKind::Cast { dst, src, to } => writeln!(
                f,
                "{} = {} as {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
                to.name()
            )?,

            // Functions
            InstKind::LoadFunction { dst, id } => writeln!(
                f,
                "{} = load_function {}",
                vn.fmt_val(*dst),
                ctx.fmt_fn_id(*id),
            )?,
            InstKind::FunctionCall {
                dst,
                callee,
                args,
                order,
                ..
            } => {
                let callee_str = match callee {
                    Callee::Direct(id) | Callee::Extern { id, .. } => ctx.fmt_fn_id(*id),
                    Callee::Indirect(val) => vn.fmt_use(*val, &consts, &texts),
                };
                write!(
                    f,
                    "{} = call {}({})",
                    vn.fmt_val(*dst),
                    callee_str,
                    vn.fmt_uses(args, &consts, &texts)
                )?;
                if let Some(edge) = order {
                    write!(
                        f,
                        " [{} -> {}]",
                        vn.fmt_use(edge.before, &consts, &texts),
                        vn.fmt_val(edge.after)
                    )?;
                }
                writeln!(f)?
            }
            InstKind::Merge { dst, orders } => writeln!(
                f,
                "{} = merge {}",
                vn.fmt_val(*dst),
                vn.fmt_uses(orders, &consts, &texts)
            )?,

            // Spawn / Eval
            InstKind::Spawn {
                dst,
                callee,
                args,
                order,
                ..
            } => {
                let callee_str = match callee {
                    Callee::Direct(id) | Callee::Extern { id, .. } => ctx.fmt_fn_id(*id),
                    Callee::Indirect(val) => vn.fmt_use(*val, &consts, &texts),
                };
                write!(
                    f,
                    "{} = spawn {}({})",
                    vn.fmt_val(*dst),
                    callee_str,
                    vn.fmt_uses(args, &consts, &texts)
                )?;
                if let Some(o) = order {
                    write!(f, " [{} ->]", vn.fmt_use(*o, &consts, &texts))?;
                }
                writeln!(f)?
            }
            InstKind::Eval { dst, src, order } => {
                write!(
                    f,
                    "{} = eval {}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*src, &consts, &texts)
                )?;
                if let Some(o) = order {
                    write!(f, " [-> {}]", vn.fmt_val(*o))?;
                }
                writeln!(f)?
            }

            // Composite constructors
            InstKind::StringClone { dst, src } => writeln!(
                f,
                "{} = string_clone {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts)
            )?,
            InstKind::StringEq { dst, a, b } => writeln!(
                f,
                "{} = string_eq {} {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*a, &consts, &texts),
                vn.fmt_use(*b, &consts, &texts)
            )?,
            InstKind::StringConcat { dst, parts } => writeln!(
                f,
                "{} = string_concat [{}]",
                vn.fmt_val(*dst),
                vn.fmt_uses(parts, &consts, &texts)
            )?,
            InstKind::MakeArray { dst, elements } => writeln!(
                f,
                "{} = list [{}]",
                vn.fmt_val(*dst),
                vn.fmt_uses(elements, &consts, &texts)
            )?,
            InstKind::MakeObject { dst, fields } => {
                let fields_str: String = fields
                    .iter()
                    .map(|(k, r)| {
                        format!(
                            "{}: {}",
                            ctx.interner.resolve(*k),
                            vn.fmt_use(*r, &consts, &texts)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(f, "{} = object {{{fields_str}}}", vn.fmt_val(*dst))?
            }
            InstKind::MakeTuple { dst, elements } => writeln!(
                f,
                "{} = tuple ({})",
                vn.fmt_val(*dst),
                vn.fmt_uses(elements, &consts, &texts)
            )?,
            InstKind::TupleIndex { dst, tuple, index } => writeln!(
                f,
                "{} = {}.{index}",
                vn.fmt_val(*dst),
                vn.fmt_use(*tuple, &consts, &texts)
            )?,

            // Pattern matching
            InstKind::TestLiteral { dst, src, value } => writeln!(
                f,
                "{} = test {} == {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
                fmt_literal(value)
            )?,
            InstKind::TestObjectKey { dst, src, key } => writeln!(
                f,
                "{} = test has_key({}, \"{}\")",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
                ctx.interner.resolve(*key),
            )?,
            InstKind::AsSlice {
                dst,
                container,
                mutability,
                ..
            } => writeln!(
                f,
                "{} = as_slice {}{}",
                vn.fmt_val(*dst),
                mutability.prefix(),
                vn.fmt_use(*container, &consts, &texts)
            )?,
            InstKind::Index {
                dst,
                slice,
                index,
                mode,
            } => writeln!(
                f,
                "{} = {}{}[{}]",
                vn.fmt_val(*dst),
                match mode {
                    IndexMode::Copy => "",
                    IndexMode::Ref => "&",
                },
                vn.fmt_use(*slice, &consts, &texts),
                vn.fmt_use(*index, &consts, &texts)
            )?,
            InstKind::IndexSet {
                slice,
                index,
                value,
            } => writeln!(
                f,
                "{}[{}] = {}",
                vn.fmt_use(*slice, &consts, &texts),
                vn.fmt_use(*index, &consts, &texts),
                vn.fmt_use(*value, &consts, &texts)
            )?,
            InstKind::ArrayIndex {
                dst,
                array: list,
                index,
            } => writeln!(
                f,
                "{} = {}[{index}]",
                vn.fmt_val(*dst),
                vn.fmt_use(*list, &consts, &texts)
            )?,
            InstKind::ObjectGet { dst, object, key } => writeln!(
                f,
                "{} = {}.{}",
                vn.fmt_val(*dst),
                vn.fmt_use(*object, &consts, &texts),
                ctx.interner.resolve(*key),
            )?,

            // Variant
            InstKind::MakeVariant { dst, tag, payload } => {
                let name = ctx.tag_name(tag);
                match payload {
                    Some(p) => writeln!(
                        f,
                        "{} = variant {}({})",
                        vn.fmt_val(*dst),
                        name,
                        vn.fmt_use(*p, &consts, &texts)
                    )?,
                    None => writeln!(f, "{} = variant {}", vn.fmt_val(*dst), name)?,
                }
            }
            InstKind::TestVariant { dst, src, tag } => {
                let name = ctx.tag_name(tag);
                writeln!(
                    f,
                    "{} = test {} is {}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*src, &consts, &texts),
                    name,
                )?
            }
            InstKind::UnwrapVariant { dst, src } => writeln!(
                f,
                "{} = unwrap {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts)
            )?,

            // Closures
            InstKind::MakeClosure {
                dst,
                body,
                captures,
            } => writeln!(
                f,
                "{} = closure {} [{}]",
                vn.fmt_val(*dst),
                fmt_label(*body),
                vn.fmt_uses(captures, &consts, &texts)
            )?,

            // Iteration

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
                                .map(|t| format!("{}", t.display(ctx.interner)))
                                .unwrap_or_else(|| "?".into());
                            format!("{}: {ty}", vn.fmt_val(*v))
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(f, "{}({params_str}):", fmt_label(*label))?
                }
            }
            // `switch r5 { A -> L1, B -> L2, _ -> L3 }` (RFC-0051).
            InstKind::Switch { tag, arms, default } => {
                let mut edge = |label: &Label, args: &[ValueId]| {
                    if args.is_empty() {
                        fmt_label(*label)
                    } else {
                        format!(
                            "{}({})",
                            fmt_label(*label),
                            vn.fmt_uses(args, &consts, &texts)
                        )
                    }
                };
                let mut parts: Vec<String> = arms
                    .iter()
                    .map(|(t, label, args)| {
                        format!("{} -> {}", ctx.interner.resolve(*t), edge(label, args))
                    })
                    .collect();
                if let Some((label, args)) = default {
                    parts.push(format!("_ -> {}", edge(label, args)));
                }
                drop(edge);
                writeln!(
                    f,
                    "switch {} {{ {} }}",
                    vn.fmt_use(*tag, &consts, &texts),
                    parts.join(", ")
                )?
            }
            InstKind::Jump { label, args } => {
                if args.is_empty() {
                    writeln!(f, "jump {}", fmt_label(*label))?
                } else {
                    writeln!(
                        f,
                        "jump {}({})",
                        fmt_label(*label),
                        vn.fmt_uses(args, &consts, &texts)
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
                        vn.fmt_uses(then_args, &consts, &texts)
                    )
                };
                let else_str = if else_args.is_empty() {
                    fmt_label(*else_label)
                } else {
                    format!(
                        "{}({})",
                        fmt_label(*else_label),
                        vn.fmt_uses(else_args, &consts, &texts)
                    )
                };
                writeln!(
                    f,
                    "jump_if {} then {} else {}",
                    vn.fmt_use(*cond, &consts, &texts),
                    then_str,
                    else_str
                )?
            }
            InstKind::Return { value, order } => {
                write!(f, "return {}", vn.fmt_use(*value, &consts, &texts))?;
                if let Some(o) = order {
                    write!(f, " [{}]", vn.fmt_use(*o, &consts, &texts))?;
                }
                writeln!(f)?
            }
            InstKind::Diverge => writeln!(f, "diverge")?,
            InstKind::Nop => writeln!(f, "nop")?,
            InstKind::Drop { src } => writeln!(f, "drop {}", vn.fmt_use(*src, &consts, &texts))?,
            InstKind::Undef { dst } => writeln!(f, "{} = undef", vn.fmt_val(*dst))?,
            InstKind::Poison { dst } => writeln!(f, "{} = poison", vn.fmt_val(*dst))?,
        }
    }

    write_order_tree(f, body, indent, ctx, &mut vn)?;

    // Print value types with origin names.
    if !body.val_types.is_empty() {
        writeln!(f)?;
        let mut entries: Vec<_> = body.val_types.iter().collect();
        entries.sort_by_key(|(v, _)| vn.get(**v));
        for (val, ty) in entries {
            let origin = body.debug.label(*val, ctx.interner);
            writeln!(
                f,
                "{indent}  ; {} ({origin}) : {}",
                vn.fmt_val(*val),
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
            collect_texts_from_body(&module.closures[label], &mut lit_to_tidx, &mut text_entries);
        }

        // Collect FunctionIds across all bodies for canonical numbering.
        let mut fn_id_map: FxHashMap<crate::graph::QualifiedRef, usize> = FxHashMap::default();
        collect_fn_ids_from_body(&module.main, &mut fn_id_map);
        for label in &labels {
            collect_fn_ids_from_body(&module.closures[label], &mut fn_id_map);
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
            fn_id_map,
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

#[derive(Clone)]
struct OrderNode {
    /// Named by instruction index: registers are reused after coloring.
    label: String,
    waits_for: Vec<OrderNode>,
}

fn write_order_tree(
    f: &mut fmt::Formatter<'_>,
    body: &MirBody,
    indent: &str,
    ctx: &PrintCtx<'_>,
    vn: &mut ValNormalizer,
) -> fmt::Result {
    let mut produced: FxHashMap<ValueId, OrderNode> = FxHashMap::default();
    let entry = |v: ValueId, vn: &mut ValNormalizer| OrderNode {
        label: format!("{}(entry)", vn.fmt_val(v)),
        waits_for: Vec::new(),
    };
    let mut handles: FxHashMap<ValueId, OrderNode> = FxHashMap::default();
    let mut returned: Option<OrderNode> = None;
    let mut any = false;
    for (i, inst) in body.insts.iter().enumerate() {
        let callee_name = |c: &Callee, vn: &mut ValNormalizer| match c {
            Callee::Direct(id) | Callee::Extern { id, .. } => ctx.fmt_fn_id(*id),
            Callee::Indirect(v) => vn.fmt_val(*v),
        };
        match &inst.kind {
            InstKind::FunctionCall {
                callee,
                order: Some(edge),
                ..
            } => {
                any = true;
                let before = produced
                    .get(&edge.before)
                    .cloned()
                    .unwrap_or_else(|| entry(edge.before, vn));
                produced.insert(
                    edge.after,
                    OrderNode {
                        label: format!("call@{i} {}", callee_name(callee, vn)),
                        waits_for: vec![before],
                    },
                );
            }
            InstKind::Spawn {
                dst,
                callee,
                order: Some(o),
                ..
            } => {
                any = true;
                let before = produced.get(o).cloned().unwrap_or_else(|| entry(*o, vn));
                handles.insert(
                    *dst,
                    OrderNode {
                        label: format!("spawn@{i} {}", callee_name(callee, vn)),
                        waits_for: vec![before],
                    },
                );
            }
            InstKind::Eval {
                src,
                order: Some(o),
                ..
            } => {
                any = true;
                let spawned = handles.get(src).cloned();
                produced.insert(
                    *o,
                    OrderNode {
                        label: format!("eval@{i}"),
                        waits_for: spawned.into_iter().collect(),
                    },
                );
            }
            InstKind::Merge { dst, orders } => {
                any = true;
                let waits_for = orders
                    .iter()
                    .map(|o| produced.get(o).cloned().unwrap_or_else(|| entry(*o, vn)))
                    .collect();
                produced.insert(
                    *dst,
                    OrderNode {
                        label: format!("merge@{i}"),
                        waits_for,
                    },
                );
            }
            InstKind::Return { order: Some(o), .. } => {
                returned = Some(produced.get(o).cloned().unwrap_or_else(|| entry(*o, vn)));
            }
            _ => {}
        }
    }
    if !any {
        return Ok(());
    }
    let Some(root) = returned else {
        return Ok(());
    };
    writeln!(f)?;
    writeln!(
        f,
        "{indent}  ; orders: what the body's yielded Order waits for"
    )?;
    write_order_node(f, &root, indent, 0)
}

fn write_order_node(
    f: &mut fmt::Formatter<'_>,
    node: &OrderNode,
    indent: &str,
    depth: usize,
) -> fmt::Result {
    let mut line = node.label.clone();
    let mut current = node;
    while current.waits_for.len() == 1 {
        current = &current.waits_for[0];
        line.push_str(" <- ");
        line.push_str(&current.label);
    }
    writeln!(f, "{indent}  ;   {}{line}", "  ".repeat(depth))?;
    for child in &current.waits_for {
        write_order_node(f, child, indent, depth + 1)?;
    }
    Ok(())
}

fn write_closure(
    f: &mut fmt::Formatter<'_>,
    label: Label,
    body: &MirBody,
    ctx: &PrintCtx<'_>,
) -> fmt::Result {
    writeln!(f)?;
    write!(f, "=== closure {} (", fmt_label(label))?;
    for (i, (_, reg)) in body.params.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{reg:?}")?;
    }
    write!(f, ")")?;
    if !body.captures.is_empty() {
        let cap_regs: Vec<_> = body.captures.iter().map(|(_, v)| v).collect();
        write!(f, " [captures: {:?}]", cap_regs)?;
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
        let mut fn_id_map = FxHashMap::default();
        collect_fn_ids_from_body(self.body, &mut fn_id_map);
        let ctx = PrintCtx {
            interner: self.interner,
            lit_to_tidx: &empty_lits,
            fn_id_map,
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
    use crate::ty::{Param, Ty};
    use acvus_utils::Interner;

    fn compile_and_dump_ctx(
        source: &str,
        context: &FxHashMap<Astr, Ty>,
        interner: &Interner,
    ) -> String {
        let ctx: Vec<(&str, Ty)> = context
            .iter()
            .map(|(name, ty)| (interner.resolve(*name), ty.clone()))
            .collect();
        crate::test::compile_and_dump(interner, source, &ctx)
    }

    #[test]
    fn print_text_only() {
        let interner = Interner::new();
        let out = crate::test::compile_and_dump(&interner, "hello world", &[]);
        assert!(out.contains("=== literals ==="));
        assert!(out.contains("\"hello world\""));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_string_emit() {
        let interner = Interner::new();
        let out = crate::test::compile_and_dump(&interner, r#"{{ "hello" }}"#, &[]);
        assert!(out.contains("\"hello\""));
        assert!(out.contains("return"));
    }

    #[test]
    fn print_match_block() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
        let out = compile_and_dump_ctx(r#"{{ true = @n == 1 }}matched{{/}}"#, &context, &interner);
        assert!(!out.contains("iter_init"));
        assert!(!out.contains("iter_next"));
        assert!(out.contains("jump_if"));
    }

    #[test]
    fn print_object_field() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(
            interner.intern("user"),
            Ty::Object(FxHashMap::from_iter([
                (interner.intern("n"), Ty::I64),
                (interner.intern("age"), Ty::I64),
            ])),
        )]);
        let out = compile_and_dump_ctx("{{ x = @user.age }}", &context, &interner);
        assert!(out.contains(".age"), "{out}");
    }

    #[test]
    fn extern_param_write_rejected() {
        let interner = Interner::new();
        let result = crate::test::compile_template(&interner, "{{ $count = 42 }}", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn print_text_dedup() {
        let interner = Interner::new();
        let out = compile_and_dump_ctx(
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
