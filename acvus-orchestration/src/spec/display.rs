/// Display spec — the mandatory output root of a pipeline.
pub enum DisplaySpec {
    /// Renders a single value.
    Static {
        name: String,
        source: String,
    },
    /// Renders items from history (List<T>, indexed) and/or live (Iterator<T>, lazy).
    Iterator {
        name: String,
        history: Option<String>,
        live: Option<String>,
        bind: String,
        template: String,
    },
}
