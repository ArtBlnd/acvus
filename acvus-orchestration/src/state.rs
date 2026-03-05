#[derive(Debug)]
pub struct State<S> {
    pub storage: S,
    pub turn: usize,
}

impl<S> State<S> {
    pub fn new(storage: S, turn: usize) -> Self {
        Self { storage, turn }
    }
}
