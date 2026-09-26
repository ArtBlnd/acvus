//! RFC-0099: a pipeline consumed where it is built is one pull loop over its
//! source, with the steps its adaptors and its consumer state written into
//! the loop's body.

mod scheme;

use acvus_ast::{Literal, Span};
use acvus_utils::{Interner, QualifiedRef};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::inst_info;
use crate::cfg::{self, Block, BlockIdx, CfgBody, ENTRY_LABEL, Terminator};
use crate::ir::{
    BinOp, Callee, ExitTrip, ExternInstance, ForSource, Inst, InstKind, Label, MirBody,
    MirModule, Overflow, PathSeg, RefTarget, Stages, SwitchKey, ValOrigin, ValueId,
};
use crate::laws::LawTable;
use crate::step::{
    AdaptorFlow, Comparison, ConsumerBlock, ConsumerStmt, RunCall, Step, StreamParam, Term,
    closures_called,
};
use crate::ty::{Mutability, ObjectTy, PolyTy, Ty, TypeArg};
use scheme::Bindings;

#[derive(Debug, Clone, PartialEq)]
pub enum Declined {
    /// The fused loop threads no `Order`, so a pipeline one of whose calls,
    /// the source's `next` included, carries an effect stays with its
    /// handlers.
    Effectful,
    NoFittingInstance(QualifiedRef),
    UntypedTerm,
    Mismatch { wanted: Ty, found: Ty },
    ConstantOutOfType { constant: Literal, ty: Ty },
    NestOverNoWordSlice(Ty),
    AdaptorYieldsNothing,
    LastBlockFallsThrough,
    /// A comparison of other than two numbers of one type.
    ComparesNoNumbers(Ty),
    /// A field the record a local holds does not have.
    NoField { ty: Ty, field: acvus_utils::Astr },
}

pub fn run(interner: &Interner, laws: &LawTable, module: &mut MirModule) -> Vec<Declined> {
    let mut declined = Vec::new();
    module.main = fused(interner, laws, std::mem::take(&mut module.main), &mut declined);
    for closure in module.closures.values_mut() {
        *closure = fused(interner, laws, std::mem::take(closure), &mut declined);
    }
    declined
}

fn fused(
    interner: &Interner,
    laws: &LawTable,
    body: MirBody,
    declined: &mut Vec<Declined>,
) -> MirBody {
    let mut cfg = cfg::promote(body);
    let mut left: FxHashSet<ValueId> = FxHashSet::default();
    while let Some(pipeline) = Pipeline::next_in(&cfg, laws, &left) {
        match Fusion::of(interner, laws, cfg.clone(), &pipeline).written() {
            Ok(fused) => cfg = fused,
            Err(why) => {
                left.insert(pipeline.result);
                declined.push(why);
            }
        }
    }
    cfg::demote(cfg)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Located {
    block: BlockIdx,
    index: usize,
}

#[derive(Debug, Clone)]
struct Adaptor {
    at: Located,
    stream: StreamParam,
    flow: AdaptorFlow,
    callee: Callee,
    args: Vec<ValueId>,
}

#[derive(Debug, Clone)]
struct Consumer {
    at: Located,
    stream: StreamParam,
    state: Option<Term>,
    body: ConsumerBlock,
    finish: Term,
    callee: Callee,
    args: Vec<ValueId>,
}

impl Consumer {
    fn terms(&self) -> Vec<&Term> {
        self.state
            .iter()
            .chain(self.body.terms())
            .chain([&self.finish])
            .collect()
    }
}

#[derive(Debug, Clone)]
struct Pipeline {
    source: ValueId,
    adaptors_from_source: Vec<Adaptor>,
    consumer: Consumer,
    result: ValueId,
    span: Span,
}

#[derive(Debug, Clone, Copy)]
struct CallSite {
    result: ValueId,
    at: Located,
}

struct Pulling<'c> {
    callee: &'c Callee,
    stream: StreamParam,
}

struct PureCall<'c> {
    callee: &'c Callee,
    args: &'c [ValueId],
}

impl Pipeline {
    fn consumer_link(&self) -> usize {
        self.adaptors_from_source.len()
    }

    fn args(&self, link: usize) -> &[ValueId] {
        match self.adaptors_from_source.get(link) {
            Some(adaptor) => &adaptor.args,
            None => &self.consumer.args,
        }
    }

    fn next_in(cfg: &CfgBody, laws: &LawTable, left: &FxHashSet<ValueId>) -> Option<Pipeline> {
        let mut calls: Vec<CallSite> = Vec::new();
        let mut readers: FxHashMap<ValueId, usize> = FxHashMap::default();
        for (b, block) in cfg.blocks.iter().enumerate() {
            for (index, inst) in block.insts.iter().enumerate() {
                if let InstKind::FunctionCall { dst, .. } = &inst.kind {
                    let block = BlockIdx(b);
                    calls.push(CallSite {
                        result: *dst,
                        at: Located { block, index },
                    });
                }
                for used in inst_info::uses(&inst.kind) {
                    *readers.entry(used).or_default() += 1;
                }
            }
            for used in inst_info::terminator_uses(&block.terminator) {
                *readers.entry(used).or_default() += 1;
            }
        }
        let defined_at: FxHashMap<ValueId, Located> =
            calls.iter().map(|call| (call.result, call.at)).collect();
        let one_reader = |value: ValueId| readers.get(&value) == Some(&1);
        let pure_call = |at: Located| match &cfg.blocks[at.block.0].insts[at.index].kind {
            InstKind::FunctionCall {
                callee,
                args,
                order: None,
                ..
            } => Some(PureCall { callee, args }),
            _ => None,
        };
        for &CallSite { result, at } in &calls {
            if left.contains(&result) {
                continue;
            }
            let Some(call) = pure_call(at) else {
                continue;
            };
            let Some(Step::Consumer {
                stream,
                state,
                body,
                finish,
            }) = laws.step_of(call.callee)
            else {
                continue;
            };
            let consumer = Consumer {
                at,
                stream: *stream,
                state: state.clone(),
                body: body.clone(),
                finish: finish.clone(),
                callee: call.callee.clone(),
                args: call.args.to_vec(),
            };
            let mut adaptors: Vec<Adaptor> = Vec::new();
            let mut pulled = consumer.args[stream.param];
            let source = loop {
                if !one_reader(pulled) {
                    break None;
                }
                let Some(&def) = defined_at.get(&pulled) else {
                    break None;
                };
                let InstKind::FunctionCall {
                    callee, args, order, ..
                } = &cfg.blocks[def.block.0].insts[def.index].kind
                else {
                    break None;
                };
                match laws.step_of(callee) {
                    None => break Some(pulled),
                    Some(Step::Consumer { .. }) => break None,
                    Some(Step::Adaptor { .. }) if order.is_some() => break None,
                    Some(Step::Adaptor { stream, flow }) => {
                        adaptors.push(Adaptor {
                            at: def,
                            stream: *stream,
                            flow: flow.clone(),
                            callee: callee.clone(),
                            args: args.clone(),
                        });
                        pulled = args[stream.param];
                    }
                }
            };
            let Some(source) = source else {
                continue;
            };
            let closures_read_once = closures_called(consumer.terms())
                .into_iter()
                .map(|param| consumer.args[param])
                .chain(adaptors.iter().flat_map(|adaptor| {
                    closures_called(adaptor.flow.terms())
                        .into_iter()
                        .map(|param| adaptor.args[param])
                }))
                .all(one_reader);
            if !closures_read_once {
                continue;
            }
            adaptors.reverse();
            return Some(Pipeline {
                source,
                adaptors_from_source: adaptors,
                consumer,
                result,
                span: cfg.blocks[at.block.0].insts[at.index].span,
            });
        }
        None
    }
}

#[derive(Debug, Clone)]
struct Slot {
    slot: ValueId,
    ty: Ty,
}

#[derive(Debug, Clone)]
struct Typed {
    value: ValueId,
    ty: Ty,
}

struct Resolved {
    instance: ExternInstance,
    callee_ty: Ty,
    params: Vec<Ty>,
    ret: Ty,
}

struct Fitting {
    instance: usize,
    bindings: Bindings,
}

struct NextCall {
    callee: Callee,
    callee_ty: Ty,
    element: Ty,
}

struct Nested {
    view: Resolved,
    element: Ty,
}

#[derive(Debug, Clone)]
enum Element {
    Typed(Ty),
    /// A step that lends the element reads it from a slot; one that only
    /// moves it moves the value itself, as `flat_map`'s and `push`'s
    /// handlers do, and no `String` copy is written for it.
    Held(Slot),
    Moved(Typed),
}

impl Element {
    fn ty(&self) -> &Ty {
        match self {
            Element::Typed(ty)
            | Element::Held(Slot { ty, .. })
            | Element::Moved(Typed { ty, .. }) => ty,
        }
    }
}

/// A local a `let` or a `match` arm binds, held in a slot as the lowering
/// holds a `let` and a pattern's binding; typed only while the state's type
/// is found.
#[derive(Debug, Clone)]
enum Local {
    Typed(Ty),
    Held(Slot),
}

impl Local {
    fn ty(&self) -> &Ty {
        match self {
            Local::Typed(ty) | Local::Held(Slot { ty, .. }) => ty,
        }
    }
}

#[derive(Debug, Clone)]
struct Scope {
    link: usize,
    element: Option<Element>,
    locals: Vec<(usize, Local)>,
}

impl Scope {
    fn without_element(link: usize) -> Self {
        Scope {
            link,
            element: None,
            locals: Vec::new(),
        }
    }

    fn with_local(&self, local: usize, bound: Local) -> Scope {
        let mut scope = self.clone();
        scope.locals.push((local, bound));
        scope
    }

    fn local(&self, local: usize) -> Result<&Local, Declined> {
        self.locals
            .iter()
            .rev()
            .find(|(bound, _)| *bound == local)
            .map(|(_, bound)| bound)
            .ok_or(Declined::UntypedTerm)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockEnd {
    Broke,
    FellThrough,
}

struct Open {
    label: Label,
    params: Vec<ValueId>,
    insts: Vec<Inst>,
    replaces: Option<BlockIdx>,
}

struct Fusion<'p> {
    interner: &'p Interner,
    laws: &'p LawTable,
    cfg: CfgBody,
    pipeline: &'p Pipeline,
    next_label: u32,
    arg_types: Vec<Vec<Ty>>,
    element_type_per_link: Vec<Ty>,
    result_ty: Ty,
    open: Option<Open>,
    /// The loop's blocks in the order they close, and the block it leaves
    /// to: both stand where the consumer's block stood, before the blocks
    /// that followed it, since `prepare` reads a `for`'s preheader as the one
    /// edge above its header and the exit as the block below its back edge.
    written_blocks: Vec<Block>,
    after_consumer: Option<Block>,
    continue_headers: Vec<Label>,
    finish: Label,
    result: Label,
    state: Option<Slot>,
    /// Each closure a step calls, by link and parameter, held in a slot and
    /// called through a shared reference to it, as the lowering calls a
    /// closure. The loop first called the closure value itself once per
    /// element, and a capturing closure whose body held a loop aborted the
    /// machine on a code pointer that was no longer valid.
    held_closures: FxHashMap<ClosureParam, Slot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ClosureParam {
    link: usize,
    param: usize,
}

impl<'p> Fusion<'p> {
    fn of(interner: &'p Interner, laws: &'p LawTable, cfg: CfgBody, pipeline: &'p Pipeline) -> Self {
        let first_free = cfg
            .blocks
            .iter()
            .map(|block| block.label)
            .filter(|label| *label != ENTRY_LABEL)
            .map(|label| label.0 + 1)
            .max()
            .unwrap_or(0);
        let arg_types = (0..=pipeline.consumer_link())
            .map(|link| {
                pipeline
                    .args(link)
                    .iter()
                    .map(|arg| cfg.val_types[arg].clone())
                    .collect()
            })
            .collect();
        let result_ty = cfg.val_types[&pipeline.result].clone();
        Fusion {
            interner,
            laws,
            cfg,
            pipeline,
            next_label: first_free + 2,
            arg_types,
            element_type_per_link: Vec::new(),
            result_ty,
            open: None,
            written_blocks: Vec::new(),
            after_consumer: None,
            continue_headers: Vec::new(),
            finish: Label(first_free),
            result: Label(first_free + 1),
            state: None,
            held_closures: FxHashMap::default(),
        }
    }

    fn written(mut self) -> Result<CfgBody, Declined> {
        self.explicit_fallthroughs()?;
        let next = self.source_next_call()?;
        self.element_type_per_link = self.element_type_per_link(next.element.clone())?;
        self.remove_adaptor_calls();
        self.split_at_consumer();

        let consumer = &self.pipeline.consumer;
        let consumer_link = self.pipeline.consumer_link();
        let source_ty = self.cfg.val_types[&self.pipeline.source].clone();
        let pulled = self.slot(source_ty.clone(), "pulled");
        self.assign(&pulled, self.pipeline.source);
        self.hold_closures();
        if let Some(start) = &consumer.state {
            let ty = self.state_ty()?;
            let slot = self.slot(ty.clone(), "state");
            let start = self.term(start, Some(&ty), &Scope::without_element(consumer_link))?;
            self.assign(&slot, start.value);
            self.state = Some(slot);
        }
        let header = self.fresh_label();
        self.jump(header, Vec::new());

        self.start(header, Vec::new());
        let lent = self.value(Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(source_ty))));
        self.push(InstKind::Ref {
            dst: lent,
            target: RefTarget::Var(pulled.slot),
            path: Vec::new(),
            mutability: Mutability::Mut,
        });
        let pulled_option = self.value(Ty::Option(Box::new(next.element.clone())));
        self.push(InstKind::FunctionCall {
            dst: pulled_option,
            callee: next.callee,
            callee_ty: next.callee_ty,
            args: vec![lent],
            order: None,
        });
        let is_some = self.value(Ty::Bool);
        self.push(InstKind::TestVariant {
            dst: is_some,
            src: pulled_option,
            tag: self.interner.intern("Some"),
        });
        let first_step = self.fresh_label();
        let finish = self.finish;
        self.close(Terminator::JumpIf {
            cond: is_some,
            then_label: first_step,
            then_args: Vec::new(),
            else_label: finish,
            else_args: Vec::new(),
        });

        self.start(first_step, Vec::new());
        let element = self.value(next.element);
        self.push(InstKind::UnwrapVariant {
            dst: element,
            src: pulled_option,
        });
        self.continue_headers.push(header);
        self.link(0, element)?;
        self.continue_headers.pop();

        self.start(finish, Vec::new());
        let result_ty = self.result_ty.clone();
        let finished = self.term(
            &consumer.finish,
            Some(&result_ty),
            &Scope::without_element(consumer_link),
        )?;
        let result = self.result;
        self.jump(result, vec![finished.value]);
        Ok(self.laid_out())
    }

    fn laid_out(mut self) -> CfgBody {
        let at = self.pipeline.consumer.at.block.0;
        let after = self
            .after_consumer
            .take()
            .expect("the consumer's block was split before the loop was written");
        let below = self.cfg.blocks.split_off(at + 1);
        self.cfg.blocks.append(&mut self.written_blocks);
        self.cfg.blocks.push(after);
        self.cfg.blocks.extend(below);
        self.cfg.label_to_block = self
            .cfg
            .blocks
            .iter()
            .enumerate()
            .map(|(at, block)| (block.label, BlockIdx(at)))
            .collect();
        self.cfg
    }

    // -- The body around the loop ---------------------------------------

    fn hold_closures(&mut self) {
        let consumer_link = self.pipeline.consumer_link();
        let called: Vec<ClosureParam> = (0..=consumer_link)
            .flat_map(|link| {
                let terms = match self.pipeline.adaptors_from_source.get(link) {
                    Some(adaptor) => adaptor.flow.terms(),
                    None => self.pipeline.consumer.terms(),
                };
                closures_called(terms)
                    .into_iter()
                    .map(move |param| ClosureParam { link, param })
            })
            .collect();
        for closure in called {
            let value = self.pipeline.args(closure.link)[closure.param];
            let ty = self.arg_types[closure.link][closure.param].clone();
            let held = self.slot(ty, "closure");
            self.assign(&held, value);
            self.held_closures.insert(closure, held);
        }
    }

    fn explicit_fallthroughs(&mut self) -> Result<(), Declined> {
        let labels: Vec<Label> = self.cfg.blocks.iter().map(|block| block.label).collect();
        for (at, block) in self.cfg.blocks.iter_mut().enumerate() {
            if !matches!(block.terminator, Terminator::Fallthrough) {
                continue;
            }
            let Some(&below) = labels.get(at + 1) else {
                return Err(Declined::LastBlockFallsThrough);
            };
            block.terminator = Terminator::Jump {
                label: below,
                args: Vec::new(),
            };
        }
        Ok(())
    }

    fn remove_adaptor_calls(&mut self) {
        let mut removed: Vec<Located> = self
            .pipeline
            .adaptors_from_source
            .iter()
            .map(|adaptor| adaptor.at)
            .collect();
        removed.sort_unstable();
        for at in removed.into_iter().rev() {
            self.cfg.blocks[at.block.0].insts.remove(at.index);
        }
    }

    fn split_at_consumer(&mut self) {
        let at = self.pipeline.consumer.at.block;
        let block = &self.cfg.blocks[at.0];
        let index = block
            .insts
            .iter()
            .position(|inst| {
                matches!(&inst.kind, InstKind::FunctionCall { dst, .. } if *dst == self.pipeline.result)
            })
            .expect("the consumer's call stands in the block it was found in");
        let after = Block {
            label: self.result,
            params: vec![self.pipeline.result],
            insts: block.insts[index + 1..].to_vec(),
            terminator: block.terminator.clone(),
        };
        let open = Open {
            label: block.label,
            params: block.params.clone(),
            insts: block.insts[..index].to_vec(),
            replaces: Some(at),
        };
        if self.cfg.demoted_diamonds.remove(&block.label) {
            self.cfg.demoted_diamonds.insert(self.result);
        }
        self.after_consumer = Some(after);
        self.open = Some(open);
    }

    // -- The stream -----------------------------------------------------

    /// `required` stands one to one against the declaration's requirements
    /// (RFC-0070 rule 3), and `#[extern_fn(step(..))]` numbered the stream's
    /// requirement in that order.
    fn source_next_call(&self) -> Result<NextCall, Declined> {
        let Pulling { callee, stream } = match self.pipeline.adaptors_from_source.first() {
            Some(innermost) => Pulling {
                callee: &innermost.callee,
                stream: innermost.stream,
            },
            None => Pulling {
                callee: &self.pipeline.consumer.callee,
                stream: self.pipeline.consumer.stream,
            },
        };
        let Callee::Extern { required, .. } = callee else {
            panic!("a step is a declaration's, and only an extern call names a declaration")
        };
        let chosen = required.get(stream.requirement).unwrap_or_else(|| {
            panic!(
                "a step's stream is requirement {} of its declaration, and the call settled {}",
                stream.requirement,
                required.len()
            )
        });
        let instance = self
            .laws
            .instance_types(chosen.signature)
            .get(chosen.instance)
            .unwrap_or_else(|| {
                panic!(
                    "the checker settled instance {} of {:?}, which the law table lacks",
                    chosen.instance, chosen.signature
                )
            });
        let source_ty = &self.cfg.val_types[&self.pipeline.source];
        let lent = Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(source_ty.clone())));
        let unfit = Declined::NoFittingInstance(chosen.signature);
        let PolyTy::Fn { params, .. } = &instance.ty else {
            return Err(unfit);
        };
        let [pulled] = params.as_slice() else {
            return Err(unfit);
        };
        let mut bindings = Bindings::default();
        if !bindings.bind(&pulled.ty, &lent) {
            return Err(unfit);
        }
        let callee_ty = bindings.apply(&instance.ty).ok_or(Declined::UntypedTerm)?;
        let Ty::Fn { ret, effect, .. } = &callee_ty else {
            return Err(unfit);
        };
        if !effect.get().is_pure() {
            return Err(Declined::Effectful);
        }
        let Ty::Option(element) = &**ret else {
            return Err(unfit);
        };
        Ok(NextCall {
            callee: Callee::Extern {
                id: chosen.signature,
                instance: chosen.instance,
                required: chosen.required.clone(),
            },
            element: (**element).clone(),
            callee_ty,
        })
    }

    fn element_type_per_link(&self, source: Ty) -> Result<Vec<Ty>, Declined> {
        let mut types = vec![source];
        for (link, adaptor) in self.pipeline.adaptors_from_source.iter().enumerate() {
            let scope = Scope {
                link,
                element: Some(Element::Typed(types[link].clone())),
                locals: Vec::new(),
            };
            let out = self
                .flow_output_ty(&adaptor.flow, &scope)?
                .ok_or(Declined::AdaptorYieldsNothing)?;
            types.push(out);
        }
        Ok(types)
    }

    fn flow_output_ty(&self, flow: &AdaptorFlow, scope: &Scope) -> Result<Option<Ty>, Declined> {
        acvus_utils::grow(|| self.flow_output_ty_level(flow, scope))
    }

    fn flow_output_ty_level(
        &self,
        flow: &AdaptorFlow,
        scope: &Scope,
    ) -> Result<Option<Ty>, Declined> {
        match flow {
            AdaptorFlow::Yield(term) => self.type_of(term, None, scope).map(Some),
            AdaptorFlow::Nest(term) => {
                let container = self.type_of(term, None, scope)?;
                Ok(Some(self.nested_element(&container)?.element))
            }
            AdaptorFlow::Skip | AdaptorFlow::Done => Ok(None),
            AdaptorFlow::If {
                then, otherwise, ..
            } => {
                let then = self.flow_output_ty(then, scope)?;
                let otherwise = self.flow_output_ty(otherwise, scope)?;
                match (then, otherwise) {
                    (Some(wanted), Some(found)) if wanted != found => {
                        Err(Declined::Mismatch { wanted, found })
                    }
                    (Some(ty), _) | (None, Some(ty)) => Ok(Some(ty)),
                    (None, None) => Ok(None),
                }
            }
        }
    }

    fn state_ty(&self) -> Result<Ty, Declined> {
        let consumer = &self.pipeline.consumer;
        let consumer_link = self.pipeline.consumer_link();
        if consumer.finish == Term::State {
            return Ok(self.result_ty.clone());
        }
        let outside = Scope::without_element(consumer_link);
        if let Some(start) = &consumer.state
            && let Ok(ty) = self.type_of(start, None, &outside)
        {
            return Ok(ty);
        }
        let scope = Scope {
            link: consumer_link,
            element: Some(Element::Typed(
                self.element_type_per_link[consumer_link].clone(),
            )),
            locals: Vec::new(),
        };
        if let Some(ty) = self.set_ty(&consumer.body, &scope) {
            return Ok(ty);
        }
        let mut calls: Vec<RunCall<'_>> = consumer.body.runs();
        for term in consumer.terms() {
            term.visit(&mut |term| {
                if let Term::CallExtern { name, args } = term {
                    calls.push(RunCall { name: *name, args });
                }
            });
        }
        for call in calls {
            let Some(at) = call.args.iter().position(reads_the_state) else {
                continue;
            };
            let known: Vec<Option<Ty>> = call
                .args
                .iter()
                .enumerate()
                .map(|(i, arg)| match i == at {
                    true => None,
                    false => self.type_of(arg, None, &scope).ok(),
                })
                .collect();
            let resolved = self.resolve(call.name, &known, None)?;
            return Ok(match (&call.args[at], &resolved.params[at]) {
                (Term::Lend(..), Ty::Ref(_, lent)) => lent.ty().into_owned(),
                (_, param) => param.clone(),
            });
        }
        Err(Declined::UntypedTerm)
    }

    /// The type of the first update `s = e` of the block whose `e` types
    /// without the state's own type: a `match s` whose `None` arm builds
    /// the state it starts.
    fn set_ty(&self, block: &ConsumerBlock, scope: &Scope) -> Option<Ty> {
        acvus_utils::grow(|| self.set_ty_level(block, scope))
    }

    fn set_ty_level(&self, block: &ConsumerBlock, scope: &Scope) -> Option<Ty> {
        let mut scope = scope.clone();
        for stmt in &block.stmts {
            match stmt {
                ConsumerStmt::Let { local, value } => {
                    let ty = self.type_of(value, None, &scope).ok()?;
                    scope = scope.with_local(*local, Local::Typed(ty));
                }
                ConsumerStmt::Set(term) => {
                    if let Ok(ty) = self.type_of(term, None, &scope) {
                        return Some(ty);
                    }
                }
                ConsumerStmt::Run { .. } => {}
                ConsumerStmt::If {
                    then, otherwise, ..
                } => {
                    if let Some(ty) = self
                        .set_ty(then, &scope)
                        .or_else(|| self.set_ty(otherwise, &scope))
                    {
                        return Some(ty);
                    }
                }
            }
        }
        None
    }

    // -- Writing the loop -------------------------------------------------

    fn link(&mut self, link: usize, element: ValueId) -> Result<(), Declined> {
        acvus_utils::grow(|| self.link_level(link, element))
    }

    fn link_level(&mut self, link: usize, element: ValueId) -> Result<(), Declined> {
        let ty = self.element_type_per_link[link].clone();
        let terms = match self.pipeline.adaptors_from_source.get(link) {
            Some(adaptor) => adaptor.flow.terms(),
            None => self.pipeline.consumer.terms(),
        };
        let lent = terms.iter().any(|term| {
            let mut lends = false;
            term.visit(&mut |term| lends |= matches!(term, Term::Lend(_, lent) if **lent == Term::Elem));
            lends
        });
        let element = match lent {
            true => {
                let held = self.slot(ty, "element");
                self.assign(&held, element);
                Element::Held(held)
            }
            false => Element::Moved(Typed { value: element, ty }),
        };
        let scope = Scope {
            link,
            element: Some(element),
            locals: Vec::new(),
        };
        match self.pipeline.adaptors_from_source.get(link) {
            Some(adaptor) => self.flow(link, &adaptor.flow, &scope),
            None => {
                if self.block(&self.pipeline.consumer.body, &scope)? == BlockEnd::FellThrough {
                    self.jump(self.innermost_header(), Vec::new());
                }
                Ok(())
            }
        }
    }

    fn flow(&mut self, link: usize, flow: &'p AdaptorFlow, scope: &Scope) -> Result<(), Declined> {
        match flow {
            AdaptorFlow::Yield(term) => {
                let ty = self.element_type_per_link[link + 1].clone();
                let passed = self.term(term, Some(&ty), scope)?;
                self.link(link + 1, passed.value)
            }
            AdaptorFlow::Skip => {
                self.jump(self.innermost_header(), Vec::new());
                Ok(())
            }
            AdaptorFlow::Done => {
                self.jump(self.finish, Vec::new());
                Ok(())
            }
            AdaptorFlow::Nest(term) => {
                let container = self.term(term, None, scope)?;
                self.nest(link + 1, container)
            }
            AdaptorFlow::If {
                cond,
                then,
                otherwise,
            } => {
                let cond = self.term(cond, Some(&Ty::Bool), scope)?;
                let then_label = self.fresh_label();
                let otherwise_label = self.fresh_label();
                self.close(Terminator::JumpIf {
                    cond: cond.value,
                    then_label,
                    then_args: Vec::new(),
                    else_label: otherwise_label,
                    else_args: Vec::new(),
                });
                self.start(then_label, Vec::new());
                self.flow(link, then, scope)?;
                self.start(otherwise_label, Vec::new());
                self.flow(link, otherwise, scope)
            }
        }
    }

    /// `flat_map`'s handler draws its source, calls the closure, and hands
    /// out the result's elements in order before it draws again; the `for`
    /// below copies the same elements out of the result's slice in that
    /// order, and is left for the enclosing header when it runs out.
    fn nest(&mut self, link: usize, container: Typed) -> Result<(), Declined> {
        let nested = self.nested_element(&container.ty)?;
        let held = self.slot(container.ty.clone(), "nested");
        self.assign(&held, container.value);
        let lent = self.value(Ty::Ref(
            Mutability::Shared,
            Box::new(TypeArg::uniform(container.ty)),
        ));
        self.push(InstKind::Ref {
            dst: lent,
            target: RefTarget::Var(held.slot),
            path: Vec::new(),
            mutability: Mutability::Shared,
        });
        let slice = self.value(nested.view.ret.clone());
        self.push(InstKind::AsSlice {
            dst: slice,
            container: lent,
            mutability: Mutability::Shared,
            instance: nested.view.instance,
        });
        let header = self.fresh_label();
        let body = self.fresh_label();
        let exit = self.fresh_label();
        self.jump(header, Vec::new());

        self.start(header, Vec::new());
        self.close(Terminator::For {
            source: ForSource::Slice(slice),
            stages: Stages::lowered(body),
            exit,
            exit_trip: ExitTrip::Absent,
            exit_args: Vec::new(),
        });

        let at = self.value(Ty::Ref(
            Mutability::Shared,
            Box::new(TypeArg::uniform(nested.element.clone())),
        ));
        let counter = self.value(Ty::U64);
        self.start(body, vec![at, counter]);
        let element = self.value(nested.element.clone());
        self.push(InstKind::Take {
            dst: element,
            target: RefTarget::Through(at),
            path: Vec::new(),
            taken_out: false,
        });
        self.continue_headers.push(header);
        self.link(link, element)?;
        self.continue_headers.pop();

        self.start(exit, Vec::new());
        self.jump(self.innermost_header(), Vec::new());
        Ok(())
    }

    fn nested_element(&self, container: &Ty) -> Result<Nested, Declined> {
        let lent = Ty::Ref(
            Mutability::Shared,
            Box::new(TypeArg::uniform(container.clone())),
        );
        let unfit = || Declined::NestOverNoWordSlice(container.clone());
        let view = self
            .laws
            .shared_slice_views()
            .iter()
            .find_map(|view| self.resolve(*view, &[Some(lent.clone())], None).ok())
            .ok_or_else(unfit)?;
        let Ty::Ref(_, slice) = &view.ret else {
            return Err(unfit());
        };
        let Ty::Slice(element) = slice.ty().into_owned() else {
            return Err(unfit());
        };
        if element.is_word() != Some(true) {
            return Err(unfit());
        }
        Ok(Nested {
            view,
            element: *element,
        })
    }

    fn block(&mut self, block: &'p ConsumerBlock, scope: &Scope) -> Result<BlockEnd, Declined> {
        let mut scope = scope.clone();
        let scope = &mut scope;
        for stmt in &block.stmts {
            match stmt {
                ConsumerStmt::Let { local, value } => {
                    let value = self.term(value, None, scope)?;
                    let held = self.slot(value.ty.clone(), "local");
                    self.assign(&held, value.value);
                    *scope = scope.with_local(*local, Local::Held(held));
                }
                ConsumerStmt::Set(term) => {
                    let state = self.state.clone().ok_or(Declined::UntypedTerm)?;
                    let next = self.term(term, Some(&state.ty), scope)?;
                    self.assign(&state, next.value);
                }
                ConsumerStmt::Run { name, args } => {
                    self.call_extern(*name, args, Some(&Ty::Unit), scope)?;
                }
                ConsumerStmt::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    let cond = self.term(cond, Some(&Ty::Bool), scope)?;
                    let then_label = self.fresh_label();
                    let otherwise_label = self.fresh_label();
                    let join = self.fresh_label();
                    self.close(Terminator::JumpIf {
                        cond: cond.value,
                        then_label,
                        then_args: Vec::new(),
                        else_label: otherwise_label,
                        else_args: Vec::new(),
                    });
                    self.start(then_label, Vec::new());
                    let then_end = self.block(then, scope)?;
                    if then_end == BlockEnd::FellThrough {
                        self.jump(join, Vec::new());
                    }
                    self.start(otherwise_label, Vec::new());
                    let otherwise_end = self.block(otherwise, scope)?;
                    if otherwise_end == BlockEnd::FellThrough {
                        self.jump(join, Vec::new());
                    }
                    if then_end == BlockEnd::Broke && otherwise_end == BlockEnd::Broke {
                        return Ok(BlockEnd::Broke);
                    }
                    self.start(join, Vec::new());
                }
            }
        }
        let Some(breaks) = &block.breaks else {
            return Ok(BlockEnd::FellThrough);
        };
        let result_ty = self.result_ty.clone();
        let result = self.term(breaks, Some(&result_ty), scope)?;
        self.jump(self.result, vec![result.value]);
        Ok(BlockEnd::Broke)
    }

    // -- Terms ------------------------------------------------------------

    fn term(&mut self, term: &Term, expected: Option<&Ty>, scope: &Scope) -> Result<Typed, Declined> {
        let typed = match term {
            Term::Elem => match &scope.element {
                Some(Element::Held(held)) => self.take(held),
                Some(Element::Moved(moved)) => moved.clone(),
                Some(Element::Typed(_)) | None => return Err(Declined::UntypedTerm),
            },
            Term::State => {
                let held = self.state.clone().ok_or(Declined::UntypedTerm)?;
                self.take(&held)
            }
            Term::ValueParam(param) => Typed {
                value: self.pipeline.args(scope.link)[*param],
                ty: self.arg_types[scope.link][*param].clone(),
            },
            Term::CallClosure { param, args } => {
                let held = self
                    .held_closures
                    .get(&ClosureParam {
                        link: scope.link,
                        param: *param,
                    })
                    .cloned()
                    .expect("`hold_closures` holds every closure a step calls");
                let closure_ty = held.ty.clone();
                let Ty::Fn {
                    params,
                    ret,
                    effect,
                    ..
                } = &closure_ty
                else {
                    return Err(Declined::UntypedTerm);
                };
                if !effect.get().is_pure() {
                    return Err(Declined::Effectful);
                }
                if params.len() != args.len() {
                    return Err(Declined::UntypedTerm);
                }
                let mut values = Vec::with_capacity(args.len());
                for (arg, param) in args.iter().zip(params) {
                    values.push(self.term(arg, Some(&param.ty), scope)?.value);
                }
                let lent = self.value(Ty::Ref(
                    Mutability::Shared,
                    Box::new(TypeArg::uniform(closure_ty.clone())),
                ));
                self.push(InstKind::Ref {
                    dst: lent,
                    target: RefTarget::Var(held.slot),
                    path: Vec::new(),
                    mutability: Mutability::Shared,
                });
                let dst = self.value((**ret).clone());
                self.push(InstKind::FunctionCall {
                    dst,
                    callee: Callee::Indirect(lent),
                    callee_ty: closure_ty.clone(),
                    args: values,
                    order: None,
                });
                Typed {
                    value: dst,
                    ty: (**ret).clone(),
                }
            }
            Term::CallExtern { name, args } => self.call_extern(*name, args, expected, scope)?,
            Term::Const(constant) => {
                let ty = expected.ok_or(Declined::UntypedTerm)?.clone();
                let value = constant_at(constant, &ty)?;
                let dst = self.value(ty.clone());
                self.push(InstKind::Const { dst, value });
                Typed { value: dst, ty }
            }
            Term::Lend(mutability, lent) => {
                let held = match &**lent {
                    Term::Elem => match &scope.element {
                        Some(Element::Held(held)) => held.clone(),
                        Some(Element::Typed(_) | Element::Moved(_)) | None => {
                            return Err(Declined::UntypedTerm);
                        }
                    },
                    Term::State => self.state.clone().ok_or(Declined::UntypedTerm)?,
                    _ => return Err(Declined::UntypedTerm),
                };
                let ty = lent_ty(*mutability, &held.ty, expected);
                let dst = self.value(ty.clone());
                self.push(InstKind::Ref {
                    dst,
                    target: RefTarget::Var(held.slot),
                    path: Vec::new(),
                    mutability: *mutability,
                });
                Typed { value: dst, ty }
            }
            Term::WrappingAdd { left, right } => {
                let ty = self.shared_operand_ty(left, right, expected, scope)?;
                let left = self.term(left, Some(&ty), scope)?;
                let right = self.term(right, Some(&ty), scope)?;
                let dst = self.value(ty.clone());
                self.push(InstKind::BinOp {
                    dst,
                    op: BinOp::Add(Overflow::Wrap),
                    left: left.value,
                    right: right.value,
                });
                Typed { value: dst, ty }
            }
            Term::Local(local) => match scope.local(*local)? {
                Local::Held(held) => {
                    let held = held.clone();
                    self.take(&held)
                }
                Local::Typed(_) => return Err(Declined::UntypedTerm),
            },
            Term::Field { local, field } => {
                let Local::Held(held) = scope.local(*local)?.clone() else {
                    return Err(Declined::UntypedTerm);
                };
                let ty = field_ty(&held.ty, *field)?;
                let dst = self.value(ty.clone());
                self.push(InstKind::Take {
                    dst,
                    target: RefTarget::Var(held.slot),
                    path: vec![PathSeg::Field(*field)],
                    taken_out: false,
                });
                Typed { value: dst, ty }
            }
            Term::Record(fields) => {
                let ty = self.type_of(term, expected, scope)?;
                let mut values = Vec::with_capacity(fields.len());
                for (name, field) in fields {
                    let field_ty = field_ty(&ty, *name)?;
                    values.push((*name, self.term(field, Some(&field_ty), scope)?.value));
                }
                let dst = self.value(ty.clone());
                self.push(InstKind::MakeObject {
                    dst,
                    fields: values,
                });
                Typed { value: dst, ty }
            }
            Term::Some(payload) => {
                let wanted = match expected {
                    Some(Ty::Option(payload)) => Some(&**payload),
                    _ => None,
                };
                let payload = self.term(payload, wanted, scope)?;
                let ty = Ty::Option(Box::new(payload.ty));
                let dst = self.value(ty.clone());
                self.push(InstKind::MakeVariant {
                    dst,
                    tag: self.interner.intern("Some"),
                    payload: Some(payload.value),
                });
                Typed { value: dst, ty }
            }
            Term::None => {
                let ty = match expected {
                    Some(ty @ Ty::Option(_)) => ty.clone(),
                    _ => return Err(Declined::UntypedTerm),
                };
                let dst = self.value(ty.clone());
                self.push(InstKind::MakeVariant {
                    dst,
                    tag: self.interner.intern("None"),
                    payload: None,
                });
                Typed { value: dst, ty }
            }
            Term::Compare { op, left, right } => {
                let ty = self.compared_ty(left, right, scope)?;
                let left = self.term(left, Some(&ty), scope)?;
                let right = self.term(right, Some(&ty), scope)?;
                let dst = self.value(Ty::Bool);
                self.push(InstKind::BinOp {
                    dst,
                    op: match op {
                        Comparison::Lt => BinOp::Lt,
                        Comparison::Gt => BinOp::Gt,
                        Comparison::Lte => BinOp::Lte,
                        Comparison::Gte => BinOp::Gte,
                        Comparison::Eq => BinOp::Eq,
                        Comparison::Neq => BinOp::Neq,
                    },
                    left: left.value,
                    right: right.value,
                });
                Typed {
                    value: dst,
                    ty: Ty::Bool,
                }
            }
            Term::If {
                cond,
                then,
                otherwise,
            } => {
                let ty = self.type_of(term, expected, scope)?;
                let cond = self.term(cond, Some(&Ty::Bool), scope)?;
                let then_label = self.fresh_label();
                let else_label = self.fresh_label();
                let join = self.fresh_label();
                self.close(Terminator::Diamond {
                    cond: cond.value,
                    then_label,
                    then_args: Vec::new(),
                    else_label,
                    else_args: Vec::new(),
                    join,
                });
                self.start(then_label, Vec::new());
                let then = self.term(then, Some(&ty), scope)?;
                self.jump(join, vec![then.value]);
                self.start(else_label, Vec::new());
                let otherwise = self.term(otherwise, Some(&ty), scope)?;
                self.jump(join, vec![otherwise.value]);
                let joined = self.value(ty.clone());
                self.start(join, vec![joined]);
                Typed { value: joined, ty }
            }
            Term::Match {
                scrutinee,
                some,
                then,
                none,
            } => self.match_option(scrutinee, *some, then, none, expected, scope)?,
        };
        match expected {
            Some(wanted) if *wanted != typed.ty => Err(Declined::Mismatch {
                wanted: wanted.clone(),
                found: typed.ty,
            }),
            _ => Ok(typed),
        }
    }

    fn type_of(&self, term: &Term, expected: Option<&Ty>, scope: &Scope) -> Result<Ty, Declined> {
        let ty = match term {
            Term::Elem => scope
                .element
                .as_ref()
                .ok_or(Declined::UntypedTerm)?
                .ty()
                .clone(),
            Term::State => self.state.as_ref().ok_or(Declined::UntypedTerm)?.ty.clone(),
            Term::ValueParam(param) => self.arg_types[scope.link][*param].clone(),
            Term::CallClosure { param, .. } => match &self.arg_types[scope.link][*param] {
                Ty::Fn { ret, .. } => (**ret).clone(),
                _ => return Err(Declined::UntypedTerm),
            },
            Term::CallExtern { name, args } => {
                let known = self.self_typed_args(args, scope);
                self.resolve(*name, &known, expected)?.ret
            }
            Term::Const(constant) => {
                let ty = expected.ok_or(Declined::UntypedTerm)?.clone();
                constant_at(constant, &ty)?;
                ty
            }
            Term::Lend(mutability, lent) => {
                let ty = self.type_of(lent, None, scope)?;
                lent_ty(*mutability, &ty, expected)
            }
            Term::WrappingAdd { left, right } => {
                self.shared_operand_ty(left, right, expected, scope)?
            }
            Term::Local(local) => scope.local(*local)?.ty().clone(),
            Term::Field { local, field } => field_ty(scope.local(*local)?.ty(), *field)?,
            Term::Record(fields) => {
                let mut typed = FxHashMap::default();
                for (name, field) in fields {
                    let wanted = match expected {
                        Some(Ty::Object(object)) => object.get(name),
                        _ => None,
                    };
                    typed.insert(*name, self.type_of(field, wanted, scope)?);
                }
                Ty::Object(ObjectTy::written(typed))
            }
            Term::Some(payload) => {
                let wanted = match expected {
                    Some(Ty::Option(payload)) => Some(&**payload),
                    _ => None,
                };
                Ty::Option(Box::new(self.type_of(payload, wanted, scope)?))
            }
            Term::None => match expected {
                Some(ty @ Ty::Option(_)) => ty.clone(),
                _ => return Err(Declined::UntypedTerm),
            },
            Term::Compare { left, right, .. } => {
                self.compared_ty(left, right, scope)?;
                Ty::Bool
            }
            Term::If {
                then, otherwise, ..
            } => self
                .type_of(then, expected, scope)
                .or_else(|_| self.type_of(otherwise, expected, scope))?,
            Term::Match {
                scrutinee,
                some,
                then,
                none,
            } => {
                let payload = match self.type_of(scrutinee, None, scope) {
                    Ok(Ty::Option(payload)) => Some(*payload),
                    Ok(other) => {
                        return Err(Declined::Mismatch {
                            wanted: Ty::Option(Box::new(other.clone())),
                            found: other,
                        });
                    }
                    Err(_) => None,
                };
                let typed_then = payload.and_then(|payload| {
                    let arm = match some {
                        Some(local) => scope.with_local(*local, Local::Typed(payload)),
                        None => scope.clone(),
                    };
                    self.type_of(then, expected, &arm).ok()
                });
                match typed_then {
                    Some(ty) => ty,
                    None => self.type_of(none, expected, scope)?,
                }
            }
        };
        match expected {
            Some(wanted) if *wanted != ty => Err(Declined::Mismatch {
                wanted: wanted.clone(),
                found: ty,
            }),
            _ => Ok(ty),
        }
    }

    /// Each argument's type where the argument types itself, a constant's
    /// being `None`: `resolve` binds those by the instance the others choose,
    /// and the call types every argument again at that instance's parameter.
    fn self_typed_args(&self, args: &[Term], scope: &Scope) -> Vec<Option<Ty>> {
        args.iter()
            .map(|arg| self.type_of(arg, None, scope).ok())
            .collect()
    }

    fn shared_operand_ty(
        &self,
        left: &Term,
        right: &Term,
        expected: Option<&Ty>,
        scope: &Scope,
    ) -> Result<Ty, Declined> {
        let ty = match (
            self.type_of(left, None, scope),
            self.type_of(right, None, scope),
        ) {
            (Ok(ty), _) | (Err(_), Ok(ty)) => ty,
            (Err(why), Err(_)) => expected.cloned().ok_or(why)?,
        };
        match ty {
            Ty::Int(_) | Ty::Float => Ok(ty),
            other => Err(Declined::Mismatch {
                wanted: Ty::Float,
                found: other,
            }),
        }
    }

    /// The one number type both operands of a comparison have.
    fn compared_ty(&self, left: &Term, right: &Term, scope: &Scope) -> Result<Ty, Declined> {
        let left_ty = self.type_of(left, None, scope);
        let right_ty = self.type_of(right, None, scope);
        let ty = match (left_ty, right_ty) {
            (Ok(left), Ok(right)) if left != right => {
                return Err(Declined::Mismatch {
                    wanted: left,
                    found: right,
                });
            }
            (Ok(ty), _) | (Err(_), Ok(ty)) => ty,
            (Err(why), Err(_)) => return Err(why),
        };
        match ty {
            Ty::Int(_) | Ty::Float => Ok(ty),
            other => Err(Declined::ComparesNoNumbers(other)),
        }
    }

    /// `match t { None => a, Some(b) => c }` as the lowering writes a
    /// `match` on an `Option` it holds: a switch on a shared reference to
    /// the holding slot (the state's own, or one `t` is put in), the `Some`
    /// arm taking the payload into the slot `b` names.
    fn match_option(
        &mut self,
        scrutinee: &Term,
        some: Option<usize>,
        then: &Term,
        none: &Term,
        expected: Option<&Ty>,
        scope: &Scope,
    ) -> Result<Typed, Declined> {
        let ty = self.type_of(
            &Term::Match {
                scrutinee: Box::new(scrutinee.clone()),
                some,
                then: Box::new(then.clone()),
                none: Box::new(none.clone()),
            },
            expected,
            scope,
        )?;
        let holder = match scrutinee {
            Term::State => self.state.clone().ok_or(Declined::UntypedTerm)?,
            other => {
                let value = self.term(other, None, scope)?;
                let held = self.slot(value.ty.clone(), "matched");
                self.assign(&held, value.value);
                held
            }
        };
        let Ty::Option(payload_ty) = holder.ty.clone() else {
            return Err(Declined::Mismatch {
                wanted: Ty::Option(Box::new(holder.ty.clone())),
                found: holder.ty,
            });
        };
        let lent = self.value(Ty::Ref(
            Mutability::Shared,
            Box::new(TypeArg::uniform(holder.ty.clone())),
        ));
        self.push(InstKind::Ref {
            dst: lent,
            target: RefTarget::Var(holder.slot),
            path: Vec::new(),
            mutability: Mutability::Shared,
        });
        let none_label = self.fresh_label();
        let some_label = self.fresh_label();
        let join = self.fresh_label();
        self.close(Terminator::Switch {
            tag: lent,
            arms: vec![
                (
                    SwitchKey::Tag(self.interner.intern("None")),
                    none_label,
                    Vec::new(),
                ),
                (
                    SwitchKey::Tag(self.interner.intern("Some")),
                    some_label,
                    Vec::new(),
                ),
            ],
            default: None,
        });
        self.start(none_label, Vec::new());
        let otherwise = self.term(none, Some(&ty), scope)?;
        self.jump(join, vec![otherwise.value]);
        self.start(some_label, Vec::new());
        let arm = match some {
            Some(local) => {
                let payload = self.value((*payload_ty).clone());
                self.push(InstKind::Take {
                    dst: payload,
                    target: RefTarget::Var(holder.slot),
                    path: vec![PathSeg::Payload],
                    taken_out: false,
                });
                let held = self.slot(*payload_ty, "payload");
                self.assign(&held, payload);
                scope.with_local(local, Local::Held(held))
            }
            None => scope.clone(),
        };
        let then = self.term(then, Some(&ty), &arm)?;
        self.jump(join, vec![then.value]);
        let joined = self.value(ty.clone());
        self.start(join, vec![joined]);
        Ok(Typed { value: joined, ty })
    }

    fn call_extern(
        &mut self,
        name: QualifiedRef,
        args: &[Term],
        expected: Option<&Ty>,
        scope: &Scope,
    ) -> Result<Typed, Declined> {
        let known = self.self_typed_args(args, scope);
        let resolved = self.resolve(name, &known, expected)?;
        let mut values = Vec::with_capacity(args.len());
        for (arg, param) in args.iter().zip(&resolved.params) {
            values.push(self.term(arg, Some(param), scope)?.value);
        }
        let dst = self.value(resolved.ret.clone());
        self.push(InstKind::FunctionCall {
            dst,
            callee: Callee::Extern {
                id: resolved.instance.id,
                instance: resolved.instance.instance,
                required: Vec::new(),
            },
            callee_ty: resolved.callee_ty,
            args: values,
            order: None,
        });
        Ok(Typed {
            value: dst,
            ty: resolved.ret,
        })
    }

    /// The instance `laws::resolve` would choose for a law naming `name` at
    /// these types: the one concrete instance they have the shape of, and
    /// otherwise the generic one.
    fn resolve(
        &self,
        name: QualifiedRef,
        args: &[Option<Ty>],
        ret: Option<&Ty>,
    ) -> Result<Resolved, Declined> {
        let unfit = Declined::NoFittingInstance(name);
        let fits = |ty: &PolyTy| -> Option<Bindings> {
            let PolyTy::Fn {
                params, ret: to, ..
            } = ty
            else {
                return None;
            };
            if params.len() != args.len() {
                return None;
            }
            let mut bindings = Bindings::default();
            for (param, arg) in params.iter().zip(args) {
                if let Some(arg) = arg
                    && !bindings.bind(&param.ty, arg)
                {
                    return None;
                }
            }
            if let Some(ret) = ret
                && !bindings.bind(to, ret)
            {
                return None;
            }
            Some(bindings)
        };
        let instances = self.laws.instance_types(name);
        let fitting = |generic: bool| -> Vec<Fitting> {
            instances
                .iter()
                .enumerate()
                .filter(|(_, instance)| instance.generic == generic)
                .filter_map(|(at, instance)| {
                    Some(Fitting {
                        instance: at,
                        bindings: fits(&instance.ty)?,
                    })
                })
                .collect()
        };
        let mut concrete = fitting(false);
        let chosen = match concrete.len() {
            1 => concrete.remove(0),
            0 => fitting(true).into_iter().next().ok_or(unfit.clone())?,
            _ => return Err(unfit),
        };
        let instance = &instances[chosen.instance];
        if instance.requires {
            return Err(unfit);
        }
        let callee_ty = chosen
            .bindings
            .apply(&instance.ty)
            .ok_or(Declined::UntypedTerm)?;
        let Ty::Fn {
            params,
            ret,
            effect,
            ..
        } = &callee_ty
        else {
            return Err(unfit);
        };
        if !effect.get().is_pure() {
            return Err(Declined::Effectful);
        }
        Ok(Resolved {
            instance: ExternInstance {
                id: name,
                instance: chosen.instance,
            },
            params: params.iter().map(|param| param.ty.clone()).collect(),
            ret: (**ret).clone(),
            callee_ty: callee_ty.clone(),
        })
    }

    // -- Blocks and values ------------------------------------------------

    fn innermost_header(&self) -> Label {
        *self
            .continue_headers
            .last()
            .expect("a step is written inside the loop it continues")
    }

    fn fresh_label(&mut self) -> Label {
        let label = Label(self.next_label);
        self.next_label += 1;
        label
    }

    fn value(&mut self, ty: Ty) -> ValueId {
        let value = self.cfg.val_factory.next();
        self.cfg.val_types.insert(value, ty);
        value
    }

    fn slot(&mut self, ty: Ty, name: &str) -> Slot {
        let slot = self.value(ty.clone());
        let name = self.interner.intern(name);
        self.cfg.debug.set(slot, ValOrigin::Named(name));
        Slot { slot, ty }
    }

    fn assign(&mut self, held: &Slot, value: ValueId) {
        self.push(InstKind::Assign {
            target: RefTarget::Var(held.slot),
            path: Vec::new(),
            value,
            restores: false,
        });
    }

    fn take(&mut self, held: &Slot) -> Typed {
        let dst = self.value(held.ty.clone());
        self.push(InstKind::Take {
            dst,
            target: RefTarget::Var(held.slot),
            path: Vec::new(),
            taken_out: false,
        });
        Typed {
            value: dst,
            ty: held.ty.clone(),
        }
    }

    fn push(&mut self, kind: InstKind) {
        let span = self.pipeline.span;
        self.open
            .as_mut()
            .expect("an instruction is written into an open block")
            .insts
            .push(Inst { span, kind });
    }

    fn start(&mut self, label: Label, params: Vec<ValueId>) {
        assert!(self.open.is_none(), "a block opens after the last one closed");
        self.open = Some(Open {
            label,
            params,
            insts: Vec::new(),
            replaces: None,
        });
    }

    fn close(&mut self, terminator: Terminator) {
        let Open {
            label,
            params,
            insts,
            replaces,
        } = self.open.take().expect("a block closes once, after it opened");
        let block = Block {
            label,
            params,
            insts,
            terminator,
        };
        match replaces {
            Some(at) => self.cfg.blocks[at.0] = block,
            None => self.written_blocks.push(block),
        }
    }

    fn jump(&mut self, label: Label, args: Vec<ValueId>) {
        self.close(Terminator::Jump { label, args });
    }
}

fn field_ty(record: &Ty, field: acvus_utils::Astr) -> Result<Ty, Declined> {
    let no_field = || Declined::NoField {
        ty: record.clone(),
        field,
    };
    match record {
        Ty::Object(object) => object.get(&field).cloned().ok_or_else(no_field),
        _ => Err(no_field()),
    }
}

fn reads_the_state(term: &Term) -> bool {
    match term {
        Term::State => true,
        Term::Lend(_, lent) => **lent == Term::State,
        _ => false,
    }
}

/// The lowering types a reference to a local at the uniform
/// representation (`inliner::Emit::lend_local`, `lower_for`), and a
/// parameter that lends the same type takes the reference at its own.
fn lent_ty(mutability: Mutability, ty: &Ty, expected: Option<&Ty>) -> Ty {
    match expected {
        Some(wanted @ Ty::Ref(m, lent)) if *m == mutability && *lent.ty() == *ty => wanted.clone(),
        _ => Ty::Ref(mutability, Box::new(TypeArg::uniform(ty.clone()))),
    }
}

fn constant_at(constant: &Literal, ty: &Ty) -> Result<Literal, Declined> {
    let out_of_type = || Declined::ConstantOutOfType {
        constant: constant.clone(),
        ty: ty.clone(),
    };
    match (constant, ty) {
        (Literal::Int(value), Ty::Int(width)) if width.holds(*value) => Ok(Literal::Int(*value)),
        (Literal::Int(value), Ty::Float) => {
            let float = *value as f64;
            match float as i128 == *value {
                true => Ok(Literal::Float(float)),
                false => Err(out_of_type()),
            }
        }
        (Literal::Float(value), Ty::Float) => Ok(Literal::Float(*value)),
        (Literal::Bool(value), Ty::Bool) => Ok(Literal::Bool(*value)),
        _ => Err(out_of_type()),
    }
}
