//! Control Flow Flattening: split straight-line code into blocks and shuffle
//! their order. All transitions become explicit jumps, making the original
//! sequential structure invisible.
//!
//! Two modes (chosen randomly per segment):
//!   1. Simple shuffle: split + reorder blocks with explicit jumps.
//!   2. Dispatcher: all blocks dispatch through a central state-machine loop.
//!      v_state = Const(initial_block_id) → dispatcher label
//!      dispatcher: JumpIf(v_state == block_0_id, block_0_label, ...) chain
//!      each block ends: v_state = Const(next_block_id) → Jump dispatcher

use acvus_ast::{BinOp, Literal};
use acvus_mir::ir::{Inst, InstKind, Label};
use acvus_mir::ty::Ty;
use rand::rngs::StdRng;
use rand::Rng;

use super::rewriter::PassState;

/// Minimum number of instructions in a segment before we bother splitting it.
const MIN_SPLIT_SIZE: usize = 4;

/// Target chunk size range when splitting.
const CHUNK_MIN: usize = 2;
const CHUNK_MAX: usize = 5;

pub fn flatten(insts: Vec<Inst>, ctx: &mut PassState, rng: &mut StdRng) -> Vec<Inst> {
    let segments = parse_segments(insts);
    let mut out = Vec::new();

    for seg in segments {
        if seg.kind == SegmentKind::StraightLine && seg.insts.len() >= MIN_SPLIT_SIZE {
            // 50% chance of dispatcher pattern vs simple shuffle.
            if rng.random_bool(0.5) {
                out.extend(flatten_dispatcher(seg.insts, ctx, rng));
            } else {
                out.extend(flatten_shuffle(seg.insts, ctx, rng));
            }
        } else {
            out.extend(seg.insts);
        }
    }

    out
}

/// Simple shuffle: split into chunks, connect with jumps, shuffle order.
/// The last chunk gets an explicit jump to an exit label so that
/// fall-through semantics are preserved regardless of emission order.
fn flatten_shuffle(insts: Vec<Inst>, ctx: &mut PassState, rng: &mut StdRng) -> Vec<Inst> {
    let mut chunks = split_segment(insts, ctx, rng);

    if chunks.len() <= 1 {
        // Nothing to shuffle.
        let mut out = Vec::new();
        for chunk in chunks {
            emit_chunk(&mut out, chunk);
        }
        return out;
    }

    // Add an exit label after the last chunk so shuffled blocks can jump to it.
    let exit_label = ctx.alloc_label();
    if let Some(last) = chunks.last_mut() {
        let span = last.insts.last().map(|i| i.span).unwrap_or(acvus_ast::Span { start: 0, end: 0 });
        last.insts.push(Inst {
            span,
            kind: InstKind::Jump { label: exit_label, args: vec![] },
        });
    }

    let first = chunks.remove(0);
    shuffle(&mut chunks, rng);

    let mut out = Vec::new();
    emit_chunk(&mut out, first);
    for chunk in chunks {
        emit_chunk(&mut out, chunk);
    }

    // Exit label at the end — code after this segment continues here.
    let span = out.last().map(|i| i.span).unwrap_or(acvus_ast::Span { start: 0, end: 0 });
    out.push(Inst {
        span,
        kind: InstKind::BlockLabel { label: exit_label, params: vec![] },
    });

    out
}

/// Dispatcher pattern: all chunks routed through a central dispatcher.
fn flatten_dispatcher(insts: Vec<Inst>, ctx: &mut PassState, rng: &mut StdRng) -> Vec<Inst> {
    // Split into chunks.
    let chunk_size = rng.random_range(CHUNK_MIN..=CHUNK_MAX);
    let raw_chunks: Vec<Vec<Inst>> = insts.chunks(chunk_size).map(|c| c.to_vec()).collect();

    if raw_chunks.len() < 2 {
        return raw_chunks.into_iter().flatten().collect();
    }

    let num_chunks = raw_chunks.len();

    // Assign a unique integer ID to each chunk.
    let chunk_ids: Vec<i64> = (0..num_chunks as i64).collect();

    // Allocate labels for each chunk and the dispatcher.
    let dispatcher_label = ctx.alloc_label();
    let exit_label = ctx.alloc_label();
    let chunk_labels: Vec<Label> = (0..num_chunks).map(|_| ctx.alloc_label()).collect();

    // Shuffle chunk order for emission.
    let mut emission_order: Vec<usize> = (0..num_chunks).collect();
    for i in (1..emission_order.len()).rev() {
        let j = rng.random_range(0..=i);
        emission_order.swap(i, j);
    }

    let span = raw_chunks[0].first().map(|i| i.span).unwrap_or(acvus_ast::Span { start: 0, end: 0 });

    let mut out = Vec::new();

    // Initialize state to the first chunk's ID.
    let v_state = ctx.alloc_val(Ty::Int);
    out.push(Inst { span, kind: InstKind::Const {
        dst: v_state,
        value: Literal::Int(chunk_ids[0]),
    }});
    out.push(Inst { span, kind: InstKind::Jump { label: dispatcher_label, args: vec![] } });

    // Emit dispatcher: chain of JumpIf comparing v_state to each chunk ID.
    out.push(Inst { span, kind: InstKind::BlockLabel { label: dispatcher_label, params: vec![] } });

    for (idx, &chunk_id) in chunk_ids.iter().enumerate() {
        if idx == num_chunks - 1 {
            // Last chunk: unconditional jump (default case).
            out.push(Inst { span, kind: InstKind::Jump { label: chunk_labels[idx], args: vec![] } });
        } else {
            let v_id = ctx.alloc_val(Ty::Int);
            out.push(Inst { span, kind: InstKind::Const {
                dst: v_id,
                value: Literal::Int(chunk_id),
            }});
            let v_cmp = ctx.alloc_val(Ty::Bool);
            out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_cmp,
                op: BinOp::Eq,
                left: v_state,
                right: v_id,
            }});

            let next_check_label = ctx.alloc_label();
            out.push(Inst { span, kind: InstKind::JumpIf {
                cond: v_cmp,
                then_label: chunk_labels[idx],
                then_args: vec![],
                else_label: next_check_label,
                else_args: vec![],
            }});
            out.push(Inst { span, kind: InstKind::BlockLabel { label: next_check_label, params: vec![] } });
        }
    }

    // Emit chunks in shuffled order.
    for &idx in &emission_order {
        let chunk_span = raw_chunks[idx].first().map(|i| i.span).unwrap_or(span);
        out.push(Inst { span: chunk_span, kind: InstKind::BlockLabel { label: chunk_labels[idx], params: vec![] } });
        out.extend(raw_chunks[idx].iter().cloned());

        if idx + 1 < num_chunks {
            // Update state to next chunk ID and jump back to dispatcher.
            // Reuse v_state by creating a new val for the assignment.
            let v_next = ctx.alloc_val(Ty::Int);
            out.push(Inst { span: chunk_span, kind: InstKind::Const {
                dst: v_next,
                value: Literal::Int(chunk_ids[idx + 1]),
            }});
            // Store the new state. We use VarStore with a dedicated variable name
            // so the dispatcher can read it. But since we're using SSA, we need
            // to use VarStore/VarLoad to thread state.
            //
            // Simpler approach: emit a VarStore + VarLoad pair for the state variable.
            let state_var = "__cff_state".to_string();
            out.push(Inst { span: chunk_span, kind: InstKind::VarStore {
                name: state_var.clone(),
                src: v_next,
            }});
            out.push(Inst { span: chunk_span, kind: InstKind::Jump { label: dispatcher_label, args: vec![] } });
        } else {
            // Last chunk: jump to exit.
            out.push(Inst { span: chunk_span, kind: InstKind::Jump { label: exit_label, args: vec![] } });
        }
    }

    out.push(Inst { span, kind: InstKind::BlockLabel { label: exit_label, params: vec![] } });

    // Fix: the dispatcher needs to read the state variable instead of the initial const.
    // Re-emit the dispatcher to use VarLoad.
    // Actually, let's restructure: use VarStore for initial state too, then VarLoad in dispatcher.
    let mut fixed_out = Vec::new();
    let state_var = "__cff_state".to_string();

    // Initial state store.
    let v_init = ctx.alloc_val(Ty::Int);
    fixed_out.push(Inst { span, kind: InstKind::Const {
        dst: v_init,
        value: Literal::Int(chunk_ids[0]),
    }});
    fixed_out.push(Inst { span, kind: InstKind::VarStore {
        name: state_var.clone(),
        src: v_init,
    }});
    fixed_out.push(Inst { span, kind: InstKind::Jump { label: dispatcher_label, args: vec![] } });

    // Dispatcher with VarLoad.
    fixed_out.push(Inst { span, kind: InstKind::BlockLabel { label: dispatcher_label, params: vec![] } });
    let v_loaded_state = ctx.alloc_val(Ty::Int);
    fixed_out.push(Inst { span, kind: InstKind::VarLoad {
        dst: v_loaded_state,
        name: state_var,
    }});

    for (idx, &chunk_id) in chunk_ids.iter().enumerate() {
        if idx == num_chunks - 1 {
            fixed_out.push(Inst { span, kind: InstKind::Jump { label: chunk_labels[idx], args: vec![] } });
        } else {
            let v_id = ctx.alloc_val(Ty::Int);
            fixed_out.push(Inst { span, kind: InstKind::Const {
                dst: v_id,
                value: Literal::Int(chunk_id),
            }});
            let v_cmp = ctx.alloc_val(Ty::Bool);
            fixed_out.push(Inst { span, kind: InstKind::BinOp {
                dst: v_cmp,
                op: BinOp::Eq,
                left: v_loaded_state,
                right: v_id,
            }});

            let next_check_label = ctx.alloc_label();
            fixed_out.push(Inst { span, kind: InstKind::JumpIf {
                cond: v_cmp,
                then_label: chunk_labels[idx],
                then_args: vec![],
                else_label: next_check_label,
                else_args: vec![],
            }});
            fixed_out.push(Inst { span, kind: InstKind::BlockLabel { label: next_check_label, params: vec![] } });
        }
    }

    // Copy chunk emissions from `out` (skip the initial setup and old dispatcher).
    // Find where chunks start in `out`.
    for &idx in &emission_order {
        let chunk_span = raw_chunks[idx].first().map(|i| i.span).unwrap_or(span);
        fixed_out.push(Inst { span: chunk_span, kind: InstKind::BlockLabel { label: chunk_labels[idx], params: vec![] } });
        fixed_out.extend(raw_chunks[idx].iter().cloned());

        if idx + 1 < num_chunks {
            let v_next = ctx.alloc_val(Ty::Int);
            fixed_out.push(Inst { span: chunk_span, kind: InstKind::Const {
                dst: v_next,
                value: Literal::Int(chunk_ids[idx + 1]),
            }});
            fixed_out.push(Inst { span: chunk_span, kind: InstKind::VarStore {
                name: "__cff_state".to_string(),
                src: v_next,
            }});
            fixed_out.push(Inst { span: chunk_span, kind: InstKind::Jump { label: dispatcher_label, args: vec![] } });
        } else {
            fixed_out.push(Inst { span: chunk_span, kind: InstKind::Jump { label: exit_label, args: vec![] } });
        }
    }

    fixed_out.push(Inst { span, kind: InstKind::BlockLabel { label: exit_label, params: vec![] } });

    fixed_out
}

fn emit_chunk(out: &mut Vec<Inst>, chunk: Chunk) {
    if let Some(label) = chunk.label {
        let span = chunk.insts.first().map(|i| i.span).unwrap_or(acvus_ast::Span { start: 0, end: 0 });
        out.push(Inst {
            span,
            kind: InstKind::BlockLabel { label, params: vec![] },
        });
    }
    out.extend(chunk.insts);
}

#[derive(Clone, Debug)]
struct Chunk {
    label: Option<Label>,
    insts: Vec<Inst>,
}

#[derive(Clone, Debug, PartialEq)]
enum SegmentKind {
    StraightLine,
    ControlFlow,
}

struct Segment {
    kind: SegmentKind,
    insts: Vec<Inst>,
}

/// Parse instructions into segments.
fn parse_segments(insts: Vec<Inst>) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut current = Vec::new();

    for inst in insts {
        let is_cf = matches!(
            inst.kind,
            InstKind::BlockLabel { .. }
                | InstKind::Jump { .. }
                | InstKind::JumpIf { .. }
                | InstKind::Return(_)
        );

        if is_cf {
            if !current.is_empty() {
                segments.push(Segment {
                    kind: SegmentKind::StraightLine,
                    insts: std::mem::take(&mut current),
                });
            }
            segments.push(Segment {
                kind: SegmentKind::ControlFlow,
                insts: vec![inst],
            });
        } else {
            current.push(inst);
        }
    }

    if !current.is_empty() {
        segments.push(Segment {
            kind: SegmentKind::StraightLine,
            insts: current,
        });
    }

    segments
}

/// Split a straight-line segment into labeled chunks connected by jumps.
fn split_segment(
    insts: Vec<Inst>,
    ctx: &mut PassState,
    rng: &mut StdRng,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut remaining = &insts[..];

    while !remaining.is_empty() {
        let size = rng.random_range(CHUNK_MIN..=CHUNK_MAX).min(remaining.len());
        let (chunk_insts, rest) = remaining.split_at(size);
        remaining = rest;

        let label = if chunks.is_empty() {
            None
        } else {
            Some(ctx.alloc_label())
        };

        chunks.push(Chunk {
            label,
            insts: chunk_insts.to_vec(),
        });
    }

    for i in 1..chunks.len() {
        if chunks[i].label.is_none() {
            chunks[i].label = Some(ctx.alloc_label());
        }
    }

    for i in 0..chunks.len() {
        if i + 1 < chunks.len() {
            let next_label = chunks[i + 1].label.unwrap();
            let span = chunks[i].insts.last().map(|i| i.span).unwrap_or(acvus_ast::Span { start: 0, end: 0 });
            chunks[i].insts.push(Inst {
                span,
                kind: InstKind::Jump { label: next_label, args: vec![] },
            });
        }
    }

    chunks
}

fn shuffle(chunks: &mut Vec<Chunk>, rng: &mut StdRng) {
    let n = chunks.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        chunks.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::{Literal, Span};
    use acvus_mir::ir::{DebugInfo, ValueId};
    use rand::SeedableRng;
    use std::collections::HashMap;

    fn span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn make_ctx() -> PassState {
        PassState {
            insts: Vec::new(),
            val_types: HashMap::new(),
            debug: DebugInfo::new(),
            next_val: 100,
            next_label: 100,
        }
    }

    #[test]
    fn splits_long_straight_line() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);

        let insts: Vec<Inst> = (0..10)
            .map(|i| Inst {
                span: span(),
                kind: InstKind::Const { dst: ValueId(i), value: Literal::Int(i as i64) },
            })
            .collect();

        let result = flatten(insts, &mut ctx, &mut rng);

        let jump_count = result.iter().filter(|i| matches!(i.kind, InstKind::Jump { .. })).count();
        assert!(jump_count >= 1, "expected at least 1 Jump, got {jump_count}");

        let label_count = result.iter().filter(|i| matches!(i.kind, InstKind::BlockLabel { .. })).count();
        assert!(label_count >= 1, "expected at least 1 BlockLabel, got {label_count}");
    }

    #[test]
    fn preserves_short_blocks() {
        let mut ctx = make_ctx();
        let mut rng = StdRng::seed_from_u64(42);

        let insts = vec![
            Inst { span: span(), kind: InstKind::Const { dst: ValueId(0), value: Literal::Int(1) } },
            Inst { span: span(), kind: InstKind::Const { dst: ValueId(1), value: Literal::Int(2) } },
        ];

        let result = flatten(insts.clone(), &mut ctx, &mut rng);

        let jump_count = result.iter().filter(|i| matches!(i.kind, InstKind::Jump { .. })).count();
        assert_eq!(jump_count, 0);
    }

    #[test]
    fn dispatcher_produces_var_store_load() {
        // Use a seed that triggers dispatcher mode.
        for seed in 0..20 {
            let mut ctx = make_ctx();
            let mut rng = StdRng::seed_from_u64(seed);

            let insts: Vec<Inst> = (0..10)
                .map(|i| Inst {
                    span: span(),
                    kind: InstKind::Const { dst: ValueId(i), value: Literal::Int(i as i64) },
                })
                .collect();

            let result = flatten(insts, &mut ctx, &mut rng);

            let has_var_store = result.iter().any(|i| matches!(&i.kind, InstKind::VarStore { name, .. } if name == "__cff_state"));
            let has_var_load = result.iter().any(|i| matches!(&i.kind, InstKind::VarLoad { name, .. } if name == "__cff_state"));

            if has_var_store {
                // If dispatcher was used, VarLoad must also be present.
                assert!(has_var_load, "dispatcher should use VarLoad for state");
                return; // Found at least one dispatcher instance.
            }
        }
        // At least one seed out of 20 should trigger dispatcher.
        panic!("expected at least one dispatcher pattern across 20 seeds");
    }
}
