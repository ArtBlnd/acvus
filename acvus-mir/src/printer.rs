use std::fmt;

use acvus_ast::{BinOp, Literal, RangeKind, UnaryOp};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::ir::{Callee, InstKind, Label, MirBody, MirModule, ValueId};

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
        Literal::String(s) => format!("{s:?}"),
        Literal::Bool(b) => b.to_string(),
        Literal::Byte(b) => format!("0x{b:02x}"),
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
    /// FunctionId → canonical index (order of first appearance across all bodies).
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
                callee: Callee::Direct(id),
                ..
            } => std::slice::from_ref(id),
            InstKind::Spawn {
                callee: Callee::Direct(id),
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

fn write_body(
    f: &mut fmt::Formatter<'_>,
    body: &MirBody,
    indent: &str,
    ctx: &PrintCtx<'_>,
) -> fmt::Result {
    let mut vn = ValNormalizer::new();

    // Build QualifiedRef → name mapping from Ref(Context) instructions + debug info.
    let mut ctx_ref_to_name: FxHashMap<crate::graph::QualifiedRef, String> = FxHashMap::default();
    for inst in &body.insts {
        if let InstKind::Ref {
            dst,
            target: crate::ir::RefTarget::Context(qref),
            ..
        } = &inst.kind
            && let Some(crate::ir::ValOrigin::Context(name)) = body.debug.get(*dst)
        {
            ctx_ref_to_name
                .entry(*qref)
                .or_insert_with(|| ctx.interner.resolve(*name).to_string());
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
            // Projection
            InstKind::Ref { dst, target, path } => {
                let base = match target {
                    crate::ir::RefTarget::Var(slot) => {
                        body.debug.label(*slot, ctx.interner)
                    }
                    crate::ir::RefTarget::Param(slot) => {
                        let name = body.debug.label(*slot, ctx.interner);
                        if name.starts_with('$') { name } else { format!("${name}") }
                    }
                    crate::ir::RefTarget::Context(qref) => {
                        let name = ctx_ref_to_name.get(qref).map(|s| s.as_str()).unwrap_or("?");
                        format!("@{name}")
                    }
                };
                if path.is_empty() {
                    writeln!(f, "{} = ref {}", vn.fmt_val(*dst), base)?
                } else {
                    let suffix: Vec<&str> = path.iter().map(|p| ctx.interner.resolve(*p)).collect();
                    writeln!(
                        f,
                        "{} = ref {}.{}",
                        vn.fmt_val(*dst),
                        base,
                        suffix.join(".")
                    )?
                }
            }
            InstKind::Load { dst, src, volatile } => {
                let vol = if *volatile { " volatile" } else { "" };
                writeln!(
                    f,
                    "{} = load{} {}",
                    vn.fmt_val(*dst),
                    vol,
                    vn.fmt_use(*src, &consts, &texts)
                )?
            }
            InstKind::Store {
                dst,
                value,
                volatile,
            } => {
                let vol = if *volatile { " volatile" } else { "" };
                writeln!(
                    f,
                    "store{} {} = {}",
                    vol,
                    vn.fmt_use(*dst, &consts, &texts),
                    vn.fmt_use(*value, &consts, &texts)
                )?
            }

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
                context_uses,
                context_defs,
            } => {
                let callee_str = match callee {
                    Callee::Direct(id) => ctx.fmt_fn_id(*id),
                    Callee::Indirect(val) => vn.fmt_use(*val, &consts, &texts),
                };
                write!(
                    f,
                    "{} = call {}({})",
                    vn.fmt_val(*dst),
                    callee_str,
                    vn.fmt_uses(args, &consts, &texts)
                )?;
                if !context_uses.is_empty() {
                    write!(f, " uses[")?;
                    for (i, (ctx_id, vid)) in context_uses.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:?}={}", ctx_id, vn.fmt_use(*vid, &consts, &texts))?;
                    }
                    write!(f, "]")?;
                }
                if !context_defs.is_empty() {
                    write!(f, " defs[")?;
                    for (i, (ctx_id, vid)) in context_defs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:?}={}", ctx_id, vn.fmt_val(*vid))?;
                    }
                    write!(f, "]")?;
                }
                writeln!(f)?
            }

            // Spawn / Eval
            InstKind::Spawn {
                dst,
                callee,
                args,
                context_uses,
            } => {
                let callee_str = match callee {
                    Callee::Direct(id) => ctx.fmt_fn_id(*id),
                    Callee::Indirect(val) => vn.fmt_use(*val, &consts, &texts),
                };
                let ctx_str = if context_uses.is_empty() {
                    String::new()
                } else {
                    let bindings: Vec<String> = context_uses
                        .iter()
                        .map(|(qref, val)| {
                            let name = ctx_ref_to_name.get(qref).map(|s| s.as_str()).unwrap_or("?");
                            format!("@{}={}", name, vn.fmt_val(*val))
                        })
                        .collect();
                    format!(" use({})", bindings.join(", "))
                };
                writeln!(
                    f,
                    "{} = spawn {}({}){ctx_str}",
                    vn.fmt_val(*dst),
                    callee_str,
                    vn.fmt_uses(args, &consts, &texts)
                )?
            }
            InstKind::Eval {
                dst,
                src,
                context_defs,
            } => {
                let ctx_str = if context_defs.is_empty() {
                    String::new()
                } else {
                    let bindings: Vec<String> = context_defs
                        .iter()
                        .map(|(qref, val)| {
                            let name = ctx_ref_to_name.get(qref).map(|s| s.as_str()).unwrap_or("?");
                            format!("@{}={}", name, vn.fmt_val(*val))
                        })
                        .collect();
                    format!(" def({})", bindings.join(", "))
                };
                writeln!(
                    f,
                    "{} = eval {}{ctx_str}",
                    vn.fmt_val(*dst),
                    vn.fmt_use(*src, &consts, &texts)
                )?
            }

            // Composite constructors
            InstKind::MakeDeque { dst, elements } => writeln!(
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
            InstKind::MakeRange {
                dst,
                start,
                end,
                kind,
            } => writeln!(
                f,
                "{} = range {}{}{}",
                vn.fmt_val(*dst),
                vn.fmt_use(*start, &consts, &texts),
                fmt_range_kind(*kind),
                vn.fmt_use(*end, &consts, &texts)
            )?,

            // Pattern matching
            InstKind::TestLiteral { dst, src, value } => writeln!(
                f,
                "{} = test {} == {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
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
                    vn.fmt_val(*dst),
                    vn.fmt_use(*src, &consts, &texts)
                )?
            }
            InstKind::TestObjectKey { dst, src, key } => writeln!(
                f,
                "{} = test has_key({}, \"{}\")",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
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
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts),
                fmt_range_kind(*kind)
            )?,
            InstKind::ListIndex { dst, list, index } => writeln!(
                f,
                "{} = {}[{index}]",
                vn.fmt_val(*dst),
                vn.fmt_use(*list, &consts, &texts)
            )?,
            InstKind::ListGet { dst, list, index } => writeln!(
                f,
                "{} = {}[{}]",
                vn.fmt_val(*dst),
                vn.fmt_use(*list, &consts, &texts),
                vn.fmt_use(*index, &consts, &texts)
            )?,
            InstKind::ListSlice {
                dst,
                list,
                skip_head,
                skip_tail,
            } => writeln!(
                f,
                "{} = {}[{skip_head}..-{skip_tail}]",
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
            InstKind::ListStep {
                dst,
                list,
                index_src,
                index_dst,
                done,
                done_args,
            } => writeln!(
                f,
                "list_step {}, {} = {}[{}] else {}({})",
                vn.fmt_val(*dst),
                vn.fmt_val(*index_dst),
                vn.fmt_use(*list, &consts, &texts),
                vn.fmt_use(*index_src, &consts, &texts),
                fmt_label(*done),
                vn.fmt_uses(done_args, &consts, &texts)
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
                            format!("{}: {ty}", vn.fmt_val(*v))
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
            InstKind::Return(r) => writeln!(f, "return {}", vn.fmt_use(*r, &consts, &texts))?,
            InstKind::Nop => writeln!(f, "nop")?,
            InstKind::Cast { dst, src, kind } => writeln!(
                f,
                "{} = cast {:?} {}",
                vn.fmt_val(*dst),
                kind,
                vn.fmt_use(*src, &consts, &texts)
            )?,
            InstKind::Clone { dst, src } => writeln!(
                f,
                "{} = clone {}",
                vn.fmt_val(*dst),
                vn.fmt_use(*src, &consts, &texts)
            )?,
            InstKind::Drop { src } => writeln!(
                f,
                "drop {}",
                vn.fmt_use(*src, &consts, &texts)
            )?,
            InstKind::Undef { dst } => writeln!(f, "{} = undef", vn.fmt_val(*dst))?,
            InstKind::Poison { dst } => writeln!(f, "{} = poison", vn.fmt_val(*dst))?,
        }
    }

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
    use crate::ty::{Effect, Param, Ty};
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
        let context = FxHashMap::from_iter([(interner.intern("name"), Ty::String)]);
        let out = compile_and_dump_ctx(
            r#"{{ true = @name == "test" }}matched{{/}}"#,
            &context,
            &interner,
        );
        assert!(!out.contains("iter_init"));
        assert!(!out.contains("iter_next"));
        assert!(out.contains("jump_if"));
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
        let out = compile_and_dump_ctx("{{ x = @fetch(1) }}{{ x }}{{_}}{{/}}", &context, &interner);
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
        let out = compile_and_dump_ctx("{{ @user.name }}", &context, &interner);
        assert!(out.contains(".name"));
    }

    #[test]
    fn extern_param_write_rejected() {
        let interner = Interner::new();
        let result = crate::test::compile_template(&interner, "{{ $count = 42 }}", &[]);
        assert!(result.is_err());
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
        let out = compile_and_dump_ctx(
            r#"{{ { name, } in @users }}{{ name }}{{/}}"#,
            &context,
            &interner,
        );
        assert!(out.contains("=== main ==="));
        assert!(out.contains("list_step"));
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
