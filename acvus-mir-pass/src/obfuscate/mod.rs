mod cff;
mod config;
mod const_obf;
mod mba;
mod opaque;
mod rewriter;
mod scheduler;
mod text_obf;


pub use config::ObfConfig;

use acvus_mir::ir::MirModule;

use crate::TransformPass;

pub struct ObfuscatePass {
    pub config: ObfConfig,
}

impl TransformPass for ObfuscatePass {
    type Required<'a> = ();

    fn transform(&self, module: MirModule, _deps: ()) -> MirModule {
        rewriter::obfuscate(module, &self.config)
    }
}
