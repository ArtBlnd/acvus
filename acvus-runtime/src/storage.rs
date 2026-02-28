use serde::{Deserialize, Serialize};

#[trait_variant::make(Send)]
pub trait Storage: SemiLattice {
    const NAME: &'static str;

    type Config;
    type Output;
    type Error;
    type Command: Serialize + for<'de> Deserialize<'de>;

    fn new(config: Self::Config) -> Self;
    async fn query(&mut self, cmd: Self::Command) -> Result<Self::Output, Self::Error>;
}

pub trait SemiLattice: Sized {
    fn join(lhs: Self, rhs: Self) -> Self;
}
