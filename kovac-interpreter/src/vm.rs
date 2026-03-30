//! The kovac bytecode VM.
//!
//! 3-bank register architecture:
//! - Bank A (a0-a3): u64 scalars, arithmetic
//! - Bank B (b0-b3): u64 scalars, arithmetic
//! - Bank M (m0-m3): heap objects (String, Vec, etc.)
//!
//! Registers are local variables, not array elements.
//! This is critical — LLVM promotes locals to physical registers.

use crate::encoding::*;

// ── Heap value ───────────────────────────────────────────────────

/// A heap-allocated value in Bank M.
#[derive(Debug, Clone)]
pub enum MValue {
    None,
    String(String),
    List(Vec<u64>),
}

impl Default for MValue {
    fn default() -> Self {
        MValue::None
    }
}

// ── VM state (exposed for testing) ───────────────────────────────

#[derive(Debug)]
pub struct VmState {
    pub a: [u64; 4],
    pub b: [u64; 4],
    pub m: [MValue; 4],
}

// ── Read helpers ─────────────────────────────────────────────────

#[inline(always)]
fn read_u16(code: &[u8], pc: usize) -> u16 {
    u16::from_le_bytes([code[pc], code[pc + 1]])
}

#[inline(always)]
fn read_u32(code: &[u8], pc: usize) -> u32 {
    u32::from_le_bytes([code[pc], code[pc + 1], code[pc + 2], code[pc + 3]])
}

#[inline(always)]
fn read_u64(code: &[u8], pc: usize) -> u64 {
    u64::from_le_bytes([
        code[pc], code[pc + 1], code[pc + 2], code[pc + 3],
        code[pc + 4], code[pc + 5], code[pc + 6], code[pc + 7],
    ])
}

// ── Register access helpers ──────────────────────────────────────

#[inline(always)]
fn get_a(idx: u8, a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
    match idx { 0 => a0, 1 => a1, 2 => a2, 3 => a3, _ => unreachable!() }
}

#[inline(always)]
fn get_b(idx: u8, b0: u64, b1: u64, b2: u64, b3: u64) -> u64 {
    match idx { 0 => b0, 1 => b1, 2 => b2, 3 => b3, _ => unreachable!() }
}

macro_rules! set_reg {
    ($idx:expr, $val:expr, $r0:ident, $r1:ident, $r2:ident, $r3:ident) => {
        match $idx { 0 => $r0 = $val, 1 => $r1 = $val, 2 => $r2 = $val, 3 => $r3 = $val, _ => unreachable!() }
    };
}

// ── Execute ──────────────────────────────────────────────────────

/// Execute a bytecode program. Returns the final register state.
pub fn execute(code: &[u8]) -> VmState {
    let mut pc: usize = 0;

    let mut a0: u64 = 0;
    let mut a1: u64 = 0;
    let mut a2: u64 = 0;
    let mut a3: u64 = 0;

    let mut b0: u64 = 0;
    let mut b1: u64 = 0;
    let mut b2: u64 = 0;
    let mut b3: u64 = 0;

    let mut m0 = MValue::None;
    let mut m1 = MValue::None;
    let mut m2 = MValue::None;
    let mut m3 = MValue::None;

    loop {
        let op = read_u16(code, pc);
        let base = decode_opcode(op);
        let rd = decode_rd(op);
        let rs1 = decode_rs1(op);
        let rs2 = decode_rs2(op);

        match base {
            HALT => break,

            // ── Bank A: ALU ──────────────────────────────
            ADD_A => { let v = get_a(rs1, a0, a1, a2, a3).wrapping_add(get_a(rs2, a0, a1, a2, a3)); set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }
            SUB_A => { let v = get_a(rs1, a0, a1, a2, a3).wrapping_sub(get_a(rs2, a0, a1, a2, a3)); set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }
            MUL_A => { let v = get_a(rs1, a0, a1, a2, a3).wrapping_mul(get_a(rs2, a0, a1, a2, a3)); set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }
            DIV_A => { let v = get_a(rs1, a0, a1, a2, a3) / get_a(rs2, a0, a1, a2, a3); set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }
            MOD_A => { let v = get_a(rs1, a0, a1, a2, a3) % get_a(rs2, a0, a1, a2, a3); set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }
            NEG_A => { let v = (get_a(rs1, a0, a1, a2, a3) as i64).wrapping_neg() as u64; set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }

            // ── Bank A: CMP ──────────────────────────────
            EQ_A => { let v = (get_a(rs1, a0, a1, a2, a3) == get_a(rs2, a0, a1, a2, a3)) as u64; set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }
            LT_A => { let v = (get_a(rs1, a0, a1, a2, a3) <  get_a(rs2, a0, a1, a2, a3)) as u64; set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }
            GT_A => { let v = (get_a(rs1, a0, a1, a2, a3) >  get_a(rs2, a0, a1, a2, a3)) as u64; set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }

            // ── Bank A: MOV/CONST ────────────────────────
            MOV_A   => { let v = get_a(rs1, a0, a1, a2, a3); set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }
            CONST_A => { let v = read_u64(code, pc + 2); set_reg!(rd, v, a0, a1, a2, a3); pc += 10; }

            // ── Bank B: ALU ──────────────────────────────
            ADD_B => { let v = get_b(rs1, b0, b1, b2, b3).wrapping_add(get_b(rs2, b0, b1, b2, b3)); set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            SUB_B => { let v = get_b(rs1, b0, b1, b2, b3).wrapping_sub(get_b(rs2, b0, b1, b2, b3)); set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            MUL_B => { let v = get_b(rs1, b0, b1, b2, b3).wrapping_mul(get_b(rs2, b0, b1, b2, b3)); set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            DIV_B => { let v = get_b(rs1, b0, b1, b2, b3) / get_b(rs2, b0, b1, b2, b3); set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            MOD_B => { let v = get_b(rs1, b0, b1, b2, b3) % get_b(rs2, b0, b1, b2, b3); set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            NEG_B => { let v = (get_b(rs1, b0, b1, b2, b3) as i64).wrapping_neg() as u64; set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }

            // ── Bank B: CMP ──────────────────────────────
            EQ_B => { let v = (get_b(rs1, b0, b1, b2, b3) == get_b(rs2, b0, b1, b2, b3)) as u64; set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            LT_B => { let v = (get_b(rs1, b0, b1, b2, b3) <  get_b(rs2, b0, b1, b2, b3)) as u64; set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            GT_B => { let v = (get_b(rs1, b0, b1, b2, b3) >  get_b(rs2, b0, b1, b2, b3)) as u64; set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }

            // ── Bank B: MOV/CONST ────────────────────────
            MOV_B   => { let v = get_b(rs1, b0, b1, b2, b3); set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            CONST_B => { let v = read_u64(code, pc + 2); set_reg!(rd, v, b0, b1, b2, b3); pc += 10; }

            // ── Cross-bank ───────────────────────────────
            MOV_A2B => { let v = get_a(rs1, a0, a1, a2, a3); set_reg!(rd, v, b0, b1, b2, b3); pc += 2; }
            MOV_B2A => { let v = get_b(rs1, b0, b1, b2, b3); set_reg!(rd, v, a0, a1, a2, a3); pc += 2; }

            // ── Control flow ─────────────────────────────
            JUMP => { pc = read_u32(code, pc + 2) as usize; }
            JUMP_IF => {
                if get_a(rs1, a0, a1, a2, a3) != 0 {
                    pc = read_u32(code, pc + 2) as usize;
                } else {
                    pc += 6;
                }
            }

            // ── Bank M ──────────────────────────────────
            MOV_M => {
                let v = match rs1 { 0 => m0.clone(), 1 => m1.clone(), 2 => m2.clone(), 3 => m3.clone(), _ => unreachable!() };
                match rd { 0 => m0 = v, 1 => m1 = v, 2 => m2 = v, 3 => m3 = v, _ => unreachable!() }
                pc += 2;
            }
            SWAP_M => {
                match (rd, rs1) {
                    (0, 1) | (1, 0) => std::mem::swap(&mut m0, &mut m1),
                    (0, 2) | (2, 0) => std::mem::swap(&mut m0, &mut m2),
                    (0, 3) | (3, 0) => std::mem::swap(&mut m0, &mut m3),
                    (1, 2) | (2, 1) => std::mem::swap(&mut m1, &mut m2),
                    (1, 3) | (3, 1) => std::mem::swap(&mut m1, &mut m3),
                    (2, 3) | (3, 2) => std::mem::swap(&mut m2, &mut m3),
                    _ => {}
                }
                pc += 2;
            }

            NOP => { pc += 2; }
            _ => panic!("illegal opcode: {base:#06X} at pc={pc}"),
        }
    }

    VmState {
        a: [a0, a1, a2, a3],
        b: [b0, b1, b2, b3],
        m: [m0, m1, m2, m3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::*;

    fn run(build: impl FnOnce(&mut ProgramBuilder)) -> VmState {
        let mut pb = ProgramBuilder::new();
        build(&mut pb);
        pb.emit(encode(HALT, 0, 0, 0));
        execute(&pb.build())
    }

    #[test]
    fn add_two_constants() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 10);
            pb.emit_const(encode1(CONST_A, 1), 20);
            pb.emit(encode(ADD_A, 2, 0, 1));
        });
        assert_eq!(state.a[2], 30);
    }

    #[test]
    fn sub_mul() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 100);
            pb.emit_const(encode1(CONST_A, 1), 30);
            pb.emit(encode(SUB_A, 2, 0, 1));
            pb.emit(encode(MUL_A, 3, 2, 1));
        });
        assert_eq!(state.a[2], 70);
        assert_eq!(state.a[3], 2100);
    }

    #[test]
    fn cross_bank_move() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 42);
            pb.emit(encode2(MOV_A2B, 2, 0));
        });
        assert_eq!(state.a[0], 42);
        assert_eq!(state.b[2], 42);
    }

    #[test]
    fn cross_bank_arithmetic() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 10);
            pb.emit_const(encode1(CONST_A, 1), 20);
            pb.emit(encode(ADD_A, 2, 0, 1));       // a2 = 30
            pb.emit(encode2(MOV_A2B, 0, 2));        // b0 = 30
            pb.emit_const(encode1(CONST_B, 1), 5);
            pb.emit(encode(SUB_B, 2, 0, 1));        // b2 = 25
        });
        assert_eq!(state.a[2], 30);
        assert_eq!(state.b[0], 30);
        assert_eq!(state.b[2], 25);
    }

    #[test]
    fn comparison() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 5);
            pb.emit_const(encode1(CONST_A, 1), 10);
            pb.emit(encode(LT_A, 2, 0, 1)); // 5 < 10 = 1
            pb.emit(encode(GT_A, 3, 0, 1)); // 5 > 10 = 0
        });
        assert_eq!(state.a[2], 1);
        assert_eq!(state.a[3], 0);
    }

    #[test]
    fn simple_loop() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 0);  // counter
            pb.emit_const(encode1(CONST_A, 1), 10); // limit
            pb.emit_const(encode1(CONST_A, 2), 1);  // increment

            let loop_top = pb.offset();
            pb.emit(encode(ADD_A, 0, 0, 2));
            pb.emit(encode(LT_A, 3, 0, 1));
            pb.emit_jump(encode2(JUMP_IF, 0, 3), loop_top);
        });
        assert_eq!(state.a[0], 10);
    }

    #[test]
    fn negation() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 42);
            pb.emit(encode2(NEG_A, 1, 0));
        });
        assert_eq!(state.a[1] as i64, -42);
    }

    #[test]
    fn bank_b_arithmetic() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_B, 0), 7);
            pb.emit_const(encode1(CONST_B, 1), 3);
            pb.emit(encode(MUL_B, 2, 0, 1));
            pb.emit(encode(MOD_B, 3, 2, 0));
        });
        assert_eq!(state.b[2], 21);
        assert_eq!(state.b[3], 0);
    }

    #[test]
    fn fibonacci_10() {
        // a0=prev(0), a1=curr(1), a2=temp
        // b0=limit(10), b1=1, b2=counter
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 0);
            pb.emit_const(encode1(CONST_A, 1), 1);
            pb.emit_const(encode1(CONST_B, 0), 10);
            pb.emit_const(encode1(CONST_B, 1), 1);
            pb.emit_const(encode1(CONST_B, 2), 1);  // start at 1 (already have fib(1))

            let loop_top = pb.offset();
            pb.emit(encode(ADD_A, 2, 0, 1));          // temp = prev + curr
            pb.emit(encode2(MOV_A, 0, 1));             // prev = curr
            pb.emit(encode2(MOV_A, 1, 2));             // curr = temp
            pb.emit(encode(ADD_B, 2, 2, 1));           // counter++
            pb.emit(encode(LT_B, 3, 2, 0));            // b3 = counter < limit
            pb.emit(encode2(MOV_B2A, 3, 3));           // a3 = b3
            pb.emit_jump(encode2(JUMP_IF, 0, 3), loop_top);
        });
        assert_eq!(state.a[1], 55);
    }

    #[test]
    fn mov_within_bank() {
        let state = run(|pb| {
            pb.emit_const(encode1(CONST_A, 0), 99);
            pb.emit(encode2(MOV_A, 3, 0));
        });
        assert_eq!(state.a[3], 99);
    }

    #[test]
    fn swap_m() {
        // Just verify SWAP_M doesn't panic
        let _state = run(|pb| {
            pb.emit(encode2(SWAP_M, 0, 1));
        });
    }
}
