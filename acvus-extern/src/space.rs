//! A journaled extension type (RFC-0033): how a value of it is laid out as
//! canonical bytes, which ops it records, how an op is applied on replay,
//! and where its nested journaled values are. A space interprets these;
//! the type never sees the space.

use std::fmt;

use acvus_mir::ty::Ty;

use crate::canonical::Canonical;
use crate::runtime::Runtime;
use crate::ty_arg::kind;

/// A content address: the BLAKE3 hash of a node's canonical bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeHash(pub [u8; 32]);

impl NodeHash {
    pub const LEN: usize = 32;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceError(pub String);

impl fmt::Display for SpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SpaceError {}

impl SpaceError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

pub type SpaceResult<T> = Result<T, SpaceError>;

/// Writes an element of type `Ty` as canonical bytes; the space provides
/// it, and it turns a nested journaled value into its head hash.
pub type Encode<'a, Rt> =
    dyn Fn(&Ty, &<Rt as Runtime>::Value, &mut Vec<u8>) -> SpaceResult<()> + 'a;
/// Reads an element of type `Ty` back.
pub type Decode<'a, Rt> = dyn Fn(&Ty, &mut &[u8]) -> SpaceResult<<Rt as Runtime>::Value> + 'a;
/// Visits a nested value the space commits before its parent.
pub type Visit<'a, Rt> = dyn FnMut(&Ty, &mut <Rt as Runtime>::Value) -> SpaceResult<()> + 'a;

/// An extension type a space can hold. `type_args` are the arguments of
/// the declared type (`Deque<Int>` gives `[Int]`); every element passes
/// through the space's own encoder, so nesting recurses.
pub trait Journaled<Rt>: Sized + Send + Sync + 'static
where
    Rt: Runtime,
{
    /// The whole value: what a checkpoint holds.
    fn encode_state(
        &self,
        rt: &Rt,
        type_args: &[Ty],
        elem: &Encode<'_, Rt>,
        out: &mut Vec<u8>,
    ) -> SpaceResult<()>;
    fn decode_state(
        rt: &Rt,
        type_args: &[Ty],
        elem: &Decode<'_, Rt>,
        input: &mut &[u8],
    ) -> SpaceResult<Self>;

    /// The ops recorded since the last take, each as canonical bytes, in
    /// order; the record is cleared.
    fn take_ops(
        &mut self,
        rt: &Rt,
        type_args: &[Ty],
        elem: &Encode<'_, Rt>,
    ) -> SpaceResult<Vec<Vec<u8>>>;
    fn apply_op(
        &mut self,
        rt: &Rt,
        type_args: &[Ty],
        elem: &Decode<'_, Rt>,
        op: &mut &[u8],
    ) -> SpaceResult<()>;

    /// Every nested value whose type is an extension type, so the space
    /// commits it first and the parent's encoding names its head.
    fn children(&mut self, type_args: &[Ty], visit: &mut Visit<'_, Rt>) -> SpaceResult<()>;

    /// The node this value was loaded from or last committed as.
    fn head(&self) -> Option<NodeHash>;
    fn set_head(&mut self, head: NodeHash);
}

type EncodeStateFn<Rt> = dyn Fn(&Rt, &<Rt as Runtime>::Value, &[Ty], &Encode<'_, Rt>, &mut Vec<u8>) -> SpaceResult<()>
    + Send
    + Sync;
type DecodeStateFn<Rt> = dyn Fn(&Rt, &[Ty], &Decode<'_, Rt>, &mut &[u8]) -> SpaceResult<<Rt as Runtime>::Value>
    + Send
    + Sync;
type TakeOpsFn<Rt> = dyn Fn(&Rt, &mut <Rt as Runtime>::Value, &[Ty], &Encode<'_, Rt>) -> SpaceResult<Vec<Vec<u8>>>
    + Send
    + Sync;
type ApplyOpFn<Rt> = dyn Fn(&Rt, &mut <Rt as Runtime>::Value, &[Ty], &Decode<'_, Rt>, &mut &[u8]) -> SpaceResult<()>
    + Send
    + Sync;
type ChildrenFn<Rt> = dyn Fn(&Rt, &mut <Rt as Runtime>::Value, &[Ty], &mut Visit<'_, Rt>) -> SpaceResult<()>
    + Send
    + Sync;
type HeadFn<Rt> = dyn Fn(&Rt, &<Rt as Runtime>::Value) -> Option<NodeHash> + Send + Sync;
type SetHeadFn<Rt> = dyn Fn(&Rt, &mut <Rt as Runtime>::Value, NodeHash) + Send + Sync;

/// A `Journaled` impl over the runtime's erased value: what a registry
/// hands the space for one type.
pub struct SpaceHooks<Rt>
where
    Rt: Runtime,
{
    pub encode_state: Box<EncodeStateFn<Rt>>,
    pub decode_state: Box<DecodeStateFn<Rt>>,
    pub take_ops: Box<TakeOpsFn<Rt>>,
    pub apply_op: Box<ApplyOpFn<Rt>>,
    pub children: Box<ChildrenFn<Rt>>,
    pub head: Box<HeadFn<Rt>>,
    pub set_head: Box<SetHeadFn<Rt>>,
}

impl<Rt> SpaceHooks<Rt>
where
    Rt: Runtime,
{
    /// `decode_state` erases through `J`'s own crossing, so its box is the
    /// one `Borrowable::deref` reads; a derived type's box is keyed by its
    /// payload, not by `J`. `Canon = J` pins the registered `J` to the form
    /// the runtime holds.
    pub fn of<J>() -> Self
    where
        J: Journaled<Rt>
            + crate::Borrowable<Rt>
            + crate::OneValue<Rt>
            + Canonical<kind::Type, Canon = J>,
    {
        // SAFETY (each hook that reads a value): the hooks are looked up by
        // the value's declared type, whose canonical form is `J`, so its box
        // is the one `J`'s crossing writes, as `decode_state`'s is; the space
        // alone holds it while a hook runs.
        Self {
            encode_state: Box::new(|rt, value, args, elem, out| {
                let reference = unsafe { rt.reference(value) };
                let j: &J = unsafe { J::deref(rt, &reference) };
                j.encode_state(rt, args, elem, out)
            }),
            decode_state: Box::new(|rt, args, elem, input| {
                let j = J::decode_state(rt, args, elem, input)?;
                Ok(<J as crate::OneValue<Rt>>::erase(j, rt))
            }),
            take_ops: Box::new(|rt, value, args, elem| {
                let reference = unsafe { rt.reference(value) };
                let j: &mut J = unsafe { J::deref_mut(rt, &reference) };
                j.take_ops(rt, args, elem)
            }),
            apply_op: Box::new(|rt, value, args, elem, op| {
                let reference = unsafe { rt.reference(value) };
                let j: &mut J = unsafe { J::deref_mut(rt, &reference) };
                j.apply_op(rt, args, elem, op)
            }),
            children: Box::new(|rt, value, args, visit| {
                let reference = unsafe { rt.reference(value) };
                let j: &mut J = unsafe { J::deref_mut(rt, &reference) };
                j.children(args, visit)
            }),
            head: Box::new(|rt, value| {
                let reference = unsafe { rt.reference(value) };
                let j: &J = unsafe { J::deref(rt, &reference) };
                j.head()
            }),
            set_head: Box::new(|rt, value, head| {
                let reference = unsafe { rt.reference(value) };
                let j: &mut J = unsafe { J::deref_mut(rt, &reference) };
                j.set_head(head)
            }),
        }
    }
}
