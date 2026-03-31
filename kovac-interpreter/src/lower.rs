//! MIR → kovac bytecode lowering.
//!
//! Linear lowerer: walks MIR instructions in order, assigns ValueIds to
//! kovac registers based on type, emits bytecode.
//!
//! Bank assignment by type:
//! - Int, Bool, Float → A bank (primary) or B bank (overflow)
//! - String, Object, List, Tuple, Deque, Variant, etc. → M bank
//!
//! Register allocation: simple linear scan with LRU eviction.
//! Not optimal — good enough to get real programs running and measure.

use acvus_mir::ir::{InstKind, MirBody, ValueId};
use acvus_mir::ty::Ty;
use acvus_utils::LocalIdOps;
use rustc_hash::FxHashMap;

use crate::encoding::*;

// ── Bank classification ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bank {
    A,
    B,
    M,
}

/// Classify a type into a bank.
fn classify_ty(ty: &Ty) -> Bank {
    match ty {
        Ty::Int | Ty::Bool | Ty::Float | Ty::Unit => Bank::A,
        _ => Bank::M,
    }
}

// ── Register allocator ───────────────────────────────────────────

/// Tracks which kovac register a ValueId lives in.
#[derive(Debug, Clone, Copy)]
struct RegSlot {
    value: ValueId,
    /// When this slot was last used (instruction index).
    last_use: usize,
}

/// Per-bank register allocator with 4 slots.
struct BankAlloc {
    slots: [Option<RegSlot>; 4],
}

impl BankAlloc {
    fn new() -> Self {
        Self { slots: [None; 4] }
    }

    /// Find which register holds this ValueId, if any.
    fn find(&self, val: ValueId) -> Option<u8> {
        self.slots
            .iter()
            .position(|s| s.map_or(false, |s| s.value == val))
            .map(|i| i as u8)
    }

    /// Allocate a register for a new ValueId. Returns the register index.
    /// If all slots are full, evicts the least recently used.
    fn alloc(&mut self, val: ValueId, time: usize) -> u8 {
        // First: find an empty slot.
        if let Some(i) = self.slots.iter().position(|s| s.is_none()) {
            self.slots[i] = Some(RegSlot {
                value: val,
                last_use: time,
            });
            return i as u8;
        }
        // All full: evict LRU.
        let lru = self
            .slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.unwrap().last_use)
            .unwrap()
            .0;
        self.slots[lru] = Some(RegSlot {
            value: val,
            last_use: time,
        });
        lru as u8
    }

    /// Mark a register as used at the given time.
    fn touch(&mut self, reg: u8, time: usize) {
        if let Some(slot) = &mut self.slots[reg as usize] {
            slot.last_use = time;
        }
    }
}

// ── Lowerer state ────────────────────────────────────────────────

struct Lowerer<'a> {
    body: &'a MirBody,
    /// ValueId → (bank, register index)
    reg_map: FxHashMap<ValueId, (Bank, u8)>,
    /// Next scalar register to assign (0-3 = A, 4-7 = B).
    scalar_count: usize,
    /// Next M register to assign (0-3).
    m_count: usize,
    pb: ProgramBuilder,
    /// Label → bytecode offset.
    label_offsets: FxHashMap<u32, u32>,
    /// Pending jump patches: (bytecode offset of the u32 target, label id).
    jump_patches: Vec<(u32, u32)>,
    /// Label → block params (pre-scanned from MIR).
    block_params: FxHashMap<u32, Vec<ValueId>>,
}

impl<'a> Lowerer<'a> {
    fn new(body: &'a MirBody) -> Self {
        // Pre-scan: collect BlockLabel → params.
        let mut block_params = FxHashMap::default();
        for inst in &body.insts {
            if let InstKind::BlockLabel { label, params, .. } = &inst.kind {
                block_params.insert(label.0, params.clone());
            }
        }

        Self {
            body,
            reg_map: FxHashMap::default(),
            scalar_count: 0,
            m_count: 0,
            pb: ProgramBuilder::new(),
            label_offsets: FxHashMap::default(),
            jump_patches: Vec::new(),
            block_params,
        }
    }

    /// Map a ValueId to a kovac register.
    ///
    /// After reg_color, ValueIds are compact (0, 1, 2, ...).
    /// Scalar types: ids 0-3 → A bank, 4-7 → B bank.
    /// M types: ids 0-3 → M bank (tracked separately).
    fn reg(&mut self, val: ValueId) -> (Bank, u8) {
        if let Some(&r) = self.reg_map.get(&val) {
            return r;
        }
        let ty = self.body.val_types.get(&val);
        let bank = ty.map_or(Bank::A, classify_ty);
        let res = self.assign_reg(val, bank);
        self.reg_map.insert(val, res);
        res
    }

    /// Assign a register based on ValueId and bank.
    ///
    /// After reg_color, ValueIds are reused via liveness analysis —
    /// the raw ValueId.0 IS the physical slot index.
    /// Scalar: slot 0-3 → A bank, 4-7 → B bank.
    /// M: separate counter.
    fn assign_reg(&mut self, val: ValueId, bank: Bank) -> (Bank, u8) {
        match bank {
            Bank::A | Bank::B => {
                let slot = val.to_raw();
                if slot < 4 {
                    (Bank::A, slot as u8)
                } else if slot < 8 {
                    (Bank::B, (slot - 4) as u8)
                } else {
                    // reg_color should keep slot count low, but if not,
                    // wrap around. This may clobber — better than panic for now.
                    let wrapped = slot % 8;
                    if wrapped < 4 {
                        (Bank::A, wrapped as u8)
                    } else {
                        (Bank::B, (wrapped - 4) as u8)
                    }
                }
            }
            Bank::M => {
                let m_idx = self.m_count;
                self.m_count += 1;
                if m_idx < 4 {
                    (Bank::M, m_idx as u8)
                } else {
                    panic!("too many M ValueIds ({}) for kovac 4 M registers", m_idx + 1);
                }
            }
        }
    }

    /// Ensure a scalar value is in bank A. If it's in B, emit MOV_B2A.
    /// Returns the A register index.
    fn ensure_a(&mut self, val: ValueId) -> u8 {
        let (bank, reg) = self.reg(val);
        match bank {
            Bank::A => reg,
            Bank::B => {
                // Cross-move to a0 temporarily (JUMP_IF always reads A).
                self.pb.emit(encode2(MOV_B2A, 0, reg));
                0
            }
            Bank::M => panic!("expected scalar, got M bank value"),
        }
    }

    /// Get the opcode base for a binary op in the given bank.
    fn binop_base(bank: Bank, op: &acvus_ast::BinOp) -> u16 {
        use acvus_ast::BinOp::*;
        let (a, b) = match op {
            Add => (ADD_A, ADD_B),
            Sub => (SUB_A, SUB_B),
            Mul => (MUL_A, MUL_B),
            Div => (DIV_A, DIV_B),
            Mod => (MOD_A, MOD_B),
            Eq => (EQ_A, EQ_B),
            Lt => (LT_A, LT_B),
            Gt => (GT_A, GT_B),
            // For ops we don't have dedicated kovac instructions yet,
            // we'll need to expand the instruction set later.
            _ => {
                todo!("BinOp {:?} not yet implemented in kovac", op)
            }
        };
        match bank {
            Bank::A => a,
            Bank::B => b,
            Bank::M => panic!("BinOp on M bank"),
        }
    }

    /// Lower all instructions in the body.
    fn lower(&mut self) {
        for inst in &self.body.insts {
            // (time tracking removed — using direct ValueId→register mapping)
            match &inst.kind {
                InstKind::Const { dst, value } => {
                    self.lower_const(*dst, value);
                }
                InstKind::BinOp {
                    dst,
                    op,
                    left,
                    right,
                } => {
                    self.lower_binop(*dst, op, *left, *right);
                }
                InstKind::UnaryOp { dst, op, operand } => {
                    self.lower_unaryop(*dst, op, *operand);
                }
                InstKind::BlockLabel { label, params, .. } => {
                    self.label_offsets.insert(label.0, self.pb.offset());
                    // Ensure params have registers allocated.
                    for p in params {
                        let _ = self.reg(*p);
                    }
                }
                InstKind::Jump { label, args } => {
                    self.lower_jump(*label, args);
                }
                InstKind::JumpIf {
                    cond,
                    then_label,
                    then_args,
                    else_label,
                    else_args,
                } => {
                    self.lower_jump_if(*cond, *then_label, then_args, *else_label, else_args);
                }
                InstKind::Return(val) => {
                    // For now, just ensure the return value is somewhere accessible.
                    let _ = self.reg(*val);
                    // Emit HALT — single function, no call stack yet.
                    self.pb.emit(encode(HALT, 0, 0, 0));
                }
                InstKind::FieldGet {
                    dst,
                    object,
                    field,
                    ..
                } => {
                    // TODO: implement properly with M bank
                    let _ = (dst, object, field);
                }
                InstKind::FunctionCall { dst, callee, args, .. } => {
                    // TODO: implement function calls
                    let _ = (dst, callee, args);
                }
                InstKind::Nop | InstKind::Undef { .. } | InstKind::Poison { .. } => {}

                // Skip instructions we haven't implemented yet.
                _ => {}
            }
        }
        // Emit HALT at the end if not already emitted.
        self.pb.emit(encode(HALT, 0, 0, 0));

        // Patch jump targets.
        self.patch_jumps();
    }

    fn lower_const(&mut self, dst: ValueId, value: &acvus_ast::Literal) {
        use acvus_ast::Literal;
        match value {
            Literal::Int(n) => {
                let (bank, rd) = self.reg(dst);
                let base = match bank {
                    Bank::A => CONST_A,
                    Bank::B => CONST_B,
                    _ => panic!("Int const should go to scalar bank"),
                };
                self.pb.emit_const(encode1(base, rd), *n as u64);
            }
            Literal::Bool(b) => {
                let (bank, rd) = self.reg(dst);
                let base = match bank {
                    Bank::A => CONST_A,
                    Bank::B => CONST_B,
                    _ => panic!("Bool const should go to scalar bank"),
                };
                self.pb.emit_const(encode1(base, rd), *b as u64);
            }
            Literal::Float(f) => {
                let (bank, rd) = self.reg(dst);
                let base = match bank {
                    Bank::A => CONST_A,
                    Bank::B => CONST_B,
                    _ => panic!("Float const should go to scalar bank"),
                };
                self.pb.emit_const(encode1(base, rd), f.to_bits());
            }
            Literal::String(_) => {
                // TODO: string constants go to M bank
            }
            Literal::Byte(b) => {
                let (bank, rd) = self.reg(dst);
                let base = match bank {
                    Bank::A => CONST_A,
                    Bank::B => CONST_B,
                    _ => panic!("Byte const should go to scalar bank"),
                };
                self.pb.emit_const(encode1(base, rd), *b as u64);
            }
            Literal::List(_) => {
                // TODO: list constants go to M bank
            }
            Literal::Unit => {
                // Unit is zero-sized; no register allocation needed.
            }
        }
    }

    fn lower_binop(
        &mut self,
        dst: ValueId,
        op: &acvus_ast::BinOp,
        left: ValueId,
        right: ValueId,
    ) {
        let (l_bank, l_reg) = self.reg(left);
        let (r_bank, r_reg) = self.reg(right);
        let (d_bank, d_reg) = self.reg(dst);

        // All three must be in the same scalar bank for a single instruction.
        // If cross-bank, emit MOV instructions to bring them together.
        // For now: require all in same bank (reg_color should make this common).
        let bank = match (l_bank, r_bank, d_bank) {
            (Bank::A, Bank::A, Bank::A) => Bank::A,
            (Bank::B, Bank::B, Bank::B) => Bank::B,
            // Cross-bank: move operands to destination bank.
            _ => {
                let target = d_bank;
                let l_reg = if l_bank != target {
                    self.emit_cross_move(l_bank, l_reg, target, d_reg);
                    d_reg // temporary reuse — not perfect but works for simple cases
                } else {
                    l_reg
                };
                let r_reg = if r_bank != target {
                    // Use a different temp register.
                    let tmp = (d_reg + 1) % 4;
                    self.emit_cross_move(r_bank, r_reg, target, tmp);
                    tmp
                } else {
                    r_reg
                };
                let base = Self::binop_base(target, op);
                self.pb.emit(encode(base, d_reg, l_reg, r_reg));
                return;
            }
        };

        let base = Self::binop_base(bank, op);
        self.pb.emit(encode(base, d_reg, l_reg, r_reg));
    }

    /// Emit MOV instructions to copy jump args into block params.
    fn emit_arg_to_param(&mut self, label: acvus_mir::ir::Label, args: &[ValueId]) {
        let params = match self.block_params.get(&label.0) {
            Some(p) => p.clone(),
            None => return,
        };
        assert_eq!(
            args.len(),
            params.len(),
            "arg count ({}) != param count ({}) for label {:?}",
            args.len(),
            params.len(),
            label
        );
        for (arg, param) in args.iter().zip(params.iter()) {
            if arg == param {
                continue; // same ValueId, reg_color may have unified them
            }
            let (a_bank, a_reg) = self.reg(*arg);
            let (p_bank, p_reg) = self.reg(*param);
            if a_bank == p_bank && a_reg == p_reg {
                continue; // already in the right register
            }
            // Emit MOV.
            match (a_bank, p_bank) {
                (Bank::A, Bank::A) => { self.pb.emit(encode2(MOV_A, p_reg, a_reg)); }
                (Bank::B, Bank::B) => { self.pb.emit(encode2(MOV_B, p_reg, a_reg)); }
                (Bank::A, Bank::B) => { self.pb.emit(encode2(MOV_A2B, p_reg, a_reg)); }
                (Bank::B, Bank::A) => { self.pb.emit(encode2(MOV_B2A, p_reg, a_reg)); }
                _ => {} // M bank moves — todo
            }
        }
    }

    fn emit_cross_move(&mut self, from: Bank, from_reg: u8, to: Bank, to_reg: u8) {
        match (from, to) {
            (Bank::A, Bank::B) => self.pb.emit(encode2(MOV_A2B, to_reg, from_reg)),
            (Bank::B, Bank::A) => self.pb.emit(encode2(MOV_B2A, to_reg, from_reg)),
            _ => panic!("cross move between {:?} and {:?} not supported", from, to),
        };
    }

    fn lower_unaryop(&mut self, dst: ValueId, op: &acvus_ast::UnaryOp, operand: ValueId) {
        use acvus_ast::UnaryOp;
        let (src_bank, src_reg) = self.reg(operand);
        let (d_bank, d_reg) = self.reg(dst);
        let bank = d_bank; // prefer destination bank
        let base = match op {
            UnaryOp::Neg => match bank {
                Bank::A => NEG_A,
                Bank::B => NEG_B,
                _ => panic!("Neg on M bank"),
            },
            UnaryOp::Not => {
                todo!("Not not yet implemented in kovac")
            }
        };
        // Cross-move if needed.
        let src_reg = if src_bank != bank {
            self.emit_cross_move(src_bank, src_reg, bank, d_reg);
            d_reg
        } else {
            src_reg
        };
        self.pb.emit(encode2(base, d_reg, src_reg));
    }

    fn lower_jump(&mut self, label: acvus_mir::ir::Label, args: &[ValueId]) {
        self.emit_arg_to_param(label, args);
        let patch_offset = self.pb.offset() + 2;
        self.pb.emit_jump(encode(JUMP, 0, 0, 0), 0);
        self.jump_patches.push((patch_offset, label.0));
    }

    fn lower_jump_if(
        &mut self,
        cond: ValueId,
        then_label: acvus_mir::ir::Label,
        then_args: &[ValueId],
        else_label: acvus_mir::ir::Label,
        else_args: &[ValueId],
    ) {
        let cond_reg = self.ensure_a(cond);

        // JUMP_IF → then_trampoline (where then-args MOVs live)
        let then_tramp_patch = self.pb.offset() + 2;
        self.pb.emit_jump(encode2(JUMP_IF, 0, cond_reg), 0);

        // Fall-through = else path: else-args MOVs, then JUMP to else_label.
        self.emit_arg_to_param(else_label, else_args);
        let else_patch = self.pb.offset() + 2;
        self.pb.emit_jump(encode(JUMP, 0, 0, 0), 0);
        self.jump_patches.push((else_patch, else_label.0));

        // Then trampoline: then-args MOVs, then JUMP to then_label.
        let then_tramp_offset = self.pb.offset();
        self.emit_arg_to_param(then_label, then_args);
        let then_patch = self.pb.offset() + 2;
        self.pb.emit_jump(encode(JUMP, 0, 0, 0), 0);
        self.jump_patches.push((then_patch, then_label.0));

        // Patch JUMP_IF target → then_trampoline.
        let code = self.pb.code_mut();
        let off = then_tramp_patch as usize;
        code[off..off + 4].copy_from_slice(&(then_tramp_offset as u32).to_le_bytes());
    }

    fn patch_jumps(&mut self) {
        let code = self.pb.code_mut();
        for (offset, label_id) in &self.jump_patches {
            if let Some(&target) = self.label_offsets.get(label_id) {
                let off = *offset as usize;
                let target_bytes = (target as u32).to_le_bytes();
                code[off..off + 4].copy_from_slice(&target_bytes);
            }
        }
    }
}

// ── Public API ───────────────────────────────────────────────────

/// Result of lowering a MIR body to kovac bytecode.
pub struct LowerResult {
    pub code: Vec<u8>,
    pub reg_map: FxHashMap<ValueId, (Bank, u8)>,
}

/// Lower a MIR body to kovac bytecode.
pub fn lower_body(body: &MirBody) -> LowerResult {
    let mut lowerer = Lowerer::new(body);
    lowerer.lower();
    LowerResult {
        code: lowerer.pb.build(),
        reg_map: lowerer.reg_map,
    }
}
