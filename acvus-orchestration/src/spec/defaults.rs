/// A named set of default values for context fields.
/// Users pick one set - each field is a (name, expr) pair.
/// The expr must be isolated and materializable.
pub struct Defaults {
    pub name: String,
    pub fields: Vec<DefaultField>,
}

/// A single default field: name + acvus expression source.
/// Type is inferred from the expression.
pub struct DefaultField {
    pub name: String,
    pub expr: String,
}
