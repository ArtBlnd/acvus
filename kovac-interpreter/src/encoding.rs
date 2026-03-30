//! Instruction encoding for the kovac bytecode VM.
//!
//! # Layout
//!
//! All instructions are 2 bytes. Fused cross-bank operations are encoded
//! as two adjacent 2-byte instructions that the dispatch loop matches as
//! a single u32.
//!
//! ## 2-byte instruction format
//!
//! ```text
//! 15        6  5  4  3  2  1  0
//! ┌─────────┬─────┬─────┬─────┐
//! │ opcode  │ rd  │ rs1 │ rs2 │
//! │ (10bit) │(2b) │(2b) │(2b) │
//! └─────────┴─────┴─────┴─────┘
//! ```
//!
//! - `rd`: destination register (0-3)
//! - `rs1`: source register 1 (0-3)
//! - `rs2`: source register 2 (0-3)
//!
//! ## Banks
//!
//! Which bank an instruction operates on is encoded in the opcode, not
//! in the register bits. `ADD_A` and `ADD_B` are different opcodes that
//! operate on bank A and bank B respectively.
//!
//! ## Fused cross-bank (4 bytes)
//!
//! Two 2-byte instructions placed adjacently. The dispatch loop reads
//! a u32 and matches known fusion patterns before falling back to
//! single u16 dispatch.
//!
//! ```text
//! ┌──────────────────┬──────────────────┐
//! │  instruction 1   │  instruction 2   │
//! │     (2 bytes)    │     (2 bytes)    │
//! └──────────────────┴──────────────────┘
//! ```
//!
//! Example: `ADD_A a0,a1,a2` followed by `MOV_A2B a0->b1` can be fused.

// ── Opcode bases (upper 10 bits, shifted left by 6) ──────────────

// Bank A arithmetic
pub const ADD_A: u16 = 0x01 << 6;
pub const SUB_A: u16 = 0x02 << 6;
pub const MUL_A: u16 = 0x03 << 6;
pub const DIV_A: u16 = 0x04 << 6;
pub const MOD_A: u16 = 0x05 << 6;
pub const NEG_A: u16 = 0x06 << 6; // rd = -rs1, rs2 ignored

// Bank A comparison (result: 0 or 1 in rd)
pub const EQ_A: u16 = 0x08 << 6;
pub const LT_A: u16 = 0x09 << 6;
pub const GT_A: u16 = 0x0A << 6;

// Bank A register moves
pub const MOV_A: u16 = 0x0C << 6; // rd = rs1, rs2 ignored
pub const CONST_A: u16 = 0x0D << 6; // rd = next_u64 (reads 8 bytes after instruction)

// Bank B arithmetic (same ops, different opcode base)
pub const ADD_B: u16 = 0x11 << 6;
pub const SUB_B: u16 = 0x12 << 6;
pub const MUL_B: u16 = 0x13 << 6;
pub const DIV_B: u16 = 0x14 << 6;
pub const MOD_B: u16 = 0x15 << 6;
pub const NEG_B: u16 = 0x16 << 6;

// Bank B comparison
pub const EQ_B: u16 = 0x18 << 6;
pub const LT_B: u16 = 0x19 << 6;
pub const GT_B: u16 = 0x1A << 6;

// Bank B register moves
pub const MOV_B: u16 = 0x1C << 6;
pub const CONST_B: u16 = 0x1D << 6;

// Cross-bank moves
pub const MOV_A2B: u16 = 0x20 << 6; // b[rd] = a[rs1], rs2 ignored
pub const MOV_B2A: u16 = 0x21 << 6; // a[rd] = b[rs1], rs2 ignored

// Bank M operations (rd/rs are M registers, operands may come from A/B)
pub const LOAD_M: u16 = 0x30 << 6; // m[rd] = heap[constant_index], rs1/rs2 ignored
pub const MOV_M: u16 = 0x31 << 6; // m[rd] = m[rs1], rs2 ignored
pub const SWAP_M: u16 = 0x32 << 6; // swap m[rd] and m[rs1], rs2 ignored

// Control flow
pub const JUMP: u16 = 0x3E << 6; // pc = next_u32 (reads 4 bytes after instruction)
pub const JUMP_IF: u16 = 0x3F << 6; // if a[rs1] != 0: pc = next_u32, else pc += 6

// System
pub const HALT: u16 = 0x00 << 6; // stop execution
pub const NOP: u16 = 0x3FF << 6; // no operation (all 1s in opcode)

// ── Register encoding helpers ────────────────────────────────────

/// Encode a 2-byte instruction: opcode_base | (rd << 4) | (rs1 << 2) | rs2
#[inline(always)]
pub const fn encode(opcode: u16, rd: u8, rs1: u8, rs2: u8) -> u16 {
    opcode | ((rd as u16 & 0x3) << 4) | ((rs1 as u16 & 0x3) << 2) | (rs2 as u16 & 0x3)
}

/// Encode with only rd and rs1 (rs2 = 0).
#[inline(always)]
pub const fn encode2(opcode: u16, rd: u8, rs1: u8) -> u16 {
    encode(opcode, rd, rs1, 0)
}

/// Encode with only rd (rs1 = 0, rs2 = 0).
#[inline(always)]
pub const fn encode1(opcode: u16, rd: u8) -> u16 {
    encode(opcode, rd, 0, 0)
}

// ── Decode helpers ───────────────────────────────────────────────

#[inline(always)]
pub const fn decode_opcode(inst: u16) -> u16 {
    inst & 0xFFC0 // upper 10 bits
}

#[inline(always)]
pub const fn decode_rd(inst: u16) -> u8 {
    ((inst >> 4) & 0x3) as u8
}

#[inline(always)]
pub const fn decode_rs1(inst: u16) -> u8 {
    ((inst >> 2) & 0x3) as u8
}

#[inline(always)]
pub const fn decode_rs2(inst: u16) -> u8 {
    (inst & 0x3) as u8
}

// ── Program builder ──────────────────────────────────────────────

/// Builder for assembling bytecode programs.
pub struct ProgramBuilder {
    code: Vec<u8>,
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self { code: Vec::new() }
    }

    /// Emit a 2-byte instruction.
    pub fn emit(&mut self, inst: u16) -> &mut Self {
        self.code.extend_from_slice(&inst.to_le_bytes());
        self
    }

    /// Emit a 2-byte instruction followed by an 8-byte constant (for CONST_A/CONST_B).
    pub fn emit_const(&mut self, inst: u16, value: u64) -> &mut Self {
        self.code.extend_from_slice(&inst.to_le_bytes());
        self.code.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Emit a 2-byte instruction followed by a 4-byte jump target (for JUMP/JUMP_IF).
    pub fn emit_jump(&mut self, inst: u16, target: u32) -> &mut Self {
        self.code.extend_from_slice(&inst.to_le_bytes());
        self.code.extend_from_slice(&target.to_le_bytes());
        self
    }

    /// Current offset in bytes (for jump targets).
    pub fn offset(&self) -> u32 {
        self.code.len() as u32
    }

    /// Mutable access to the underlying byte buffer.
    pub fn code_mut(&mut self) -> &mut Vec<u8> {
        &mut self.code
    }

    /// Finish and return the bytecode.
    pub fn build(self) -> Vec<u8> {
        self.code
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}
