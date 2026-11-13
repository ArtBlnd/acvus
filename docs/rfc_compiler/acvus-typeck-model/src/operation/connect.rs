//! Connecting: making two classes one type.

use crate::primitive::ItemId;
use crate::primitive::class::Side;
use crate::primitive::constraint::ConstraintOf;
use crate::store::Store;

use super::Outcome;
use super::law::{Law, Pair};
use super::take_in::take_in;

/// Makes the two classes of `pair` one type, by `at`, and says whether they were
/// not so already.
///
/// | The two classes | What being one type comes to |
/// |---|---|
/// | neither decided | They are one class: what enters either enters it, and the parts of one name on the two are one type in turn. |
/// | one decided by solving or by the text | The other is that class. What was recorded on it stays to be judged against the head, and each part named on it is one type with the part in its place. |
/// | one a declaration writes | Nothing is merged with it. The law's bound says the other has its head, and is judged as any bound. |
/// | both decided | They have one head, or the program is refused, and their parts are one type in turn. |
///
/// This is the only place two classes are merged. A part named on a class
/// decided as a head with no place of that name is asked of the law
/// ([`Law::place`]), and so is the head a literal's class takes
/// ([`Law::admits`]).
pub fn connect<C, L>(store: &mut Store<C>, law: &L, outcome: &mut Outcome, pair: Pair, at: ItemId) -> bool
where
    C: ConstraintOf,
    L: Law<C>,
{
    let Store { classes, scopes, .. } = store;
    let mut pairs = vec![(pair, at)];
    let mut connected = false;
    while let Some((pair, at)) = pairs.pop() {
        let (a, b) = match pair {
            Pair::Classes(a, b) => (a, b),
            Pair::Part { part, of, key } => {
                let Some(named) = classes.part(of, key, Side::Carried, at) else {
                    continue;
                };
                (named, part)
            }
        };
        let (a, b) = (classes.find(a).id(), classes.find(b).id());
        if a == b {
            continue;
        }
        let compared = match (classes.is_external(a), classes.is_external(b)) {
            (_, true) => Some(law.same(a, b)),
            (true, false) => Some(law.same(b, a)),
            (false, false) => None,
        };
        match compared {
            Some(constraint) => connected |= classes.constrain(constraint, at),
            None => {
                let unified = classes.unify(scopes, a, b, at);
                connected = true;
                pairs.extend(take_in(classes, law, outcome, unified));
            }
        }
    }
    connected
}
