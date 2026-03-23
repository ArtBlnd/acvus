use acvus_mir::ty::Ty;

/// Plain node spec — renders a single template, no model call.
#[derive(Debug, Clone)]
pub struct PlainSpec {
    pub source: String,
}

impl PlainSpec {
    pub fn output_ty(&self) -> Ty {
        Ty::String
    }
}
