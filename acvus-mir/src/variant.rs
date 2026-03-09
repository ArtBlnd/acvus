/// Payload specification for a builtin variant constructor (Option).
#[derive(Debug, Clone)]
pub enum VariantPayload {
    /// No payload (e.g. `None`).
    None,
    /// Payload type is the i-th type parameter of the parent enum.
    TypeParam(usize),
}
