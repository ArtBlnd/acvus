//! This module reads no law, by decision. Whether a carried state combines
//! through a law, and which one, is read in `analysis::loop_deps` from what
//! the cycle over the state computes, so that a law has one reader
//! (RFC-0066 rule 5, RFC-0089 rule 4) and no second recognizer here can
//! disagree with it.

use crate::analysis::affine::{AffineValues, Derivation};
use crate::analysis::loops::Loop;
use crate::cfg::CfgBody;
use crate::ir::ValueId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carried {
    Iv,
    State,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarriedParam {
    pub param: ValueId,
    pub carried: Carried,
}

pub struct CarriedState {
    pub params: Vec<CarriedParam>,
}

impl CarriedState {
    pub fn of(cfg: &CfgBody, loop_: &Loop, affine: &AffineValues) -> Self {
        let params = cfg.blocks[loop_.natural.header.0]
            .params
            .iter()
            .map(|&param| CarriedParam {
                param,
                carried: match affine.get(param).map(|a| &a.derivation) {
                    Some(Derivation::Carried { .. }) => Carried::Iv,
                    _ => Carried::State,
                },
            })
            .collect();
        Self { params }
    }
}
