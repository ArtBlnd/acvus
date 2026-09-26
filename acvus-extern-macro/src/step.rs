//! `#[extern_fn(step(..))]`: RFC-0099 rule 1's step, parsed into
//! `acvus_extern::step`.

use std::cell::RefCell;

use quote::quote;
use syn::parse::ParseStream;
use syn::{Ident, Lit, Path, Token, Type};

use crate::{ExternParam, Returning};

pub(crate) struct StepAttr {
    keyword: Ident,
    state: Option<StateDecl>,
    shape: Shape,
}

struct StateDecl {
    name: Ident,
    init: Expr,
}

enum Shape {
    Adaptor(Flow),
    Consumer { body: Block, finish: Expr },
}

enum Flow {
    Yield(Expr),
    Skip,
    Done,
    Nest(Expr),
    If {
        cond: Expr,
        then: Box<Flow>,
        otherwise: Box<Flow>,
    },
}

struct Block {
    stmts: Vec<Stmt>,
    breaks: Option<Expr>,
}

enum Stmt {
    Set { state: Ident, value: Expr },
    Run { callee: Path, args: Vec<Expr> },
    If {
        cond: Expr,
        then: Block,
        otherwise: Block,
    },
}

enum Expr {
    Name(Ident),
    Call { callee: Path, args: Vec<Expr> },
    Lit { negative: bool, lit: Lit },
    Lend { mutable: bool, of: Box<Expr> },
    WrappingAdd { left: Box<Expr>, right: Box<Expr> },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockCloser {
    Finish,
    Brace,
}

const ELEMENT: &str = "x";
const ADAPTOR_ENDS: [&str; 3] = ["skip", "done", "nest"];

fn next_word_is(input: ParseStream, words: &[&str]) -> bool {
    input.peek(Ident)
        && input
            .fork()
            .parse::<Ident>()
            .is_ok_and(|word| words.iter().any(|w| word == w))
}

impl StepAttr {
    pub(crate) fn parse_after(keyword: Ident, input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::parenthesized!(content in input);
        let state = match next_word_is(&content, &["state"]) {
            true => {
                content.parse::<Ident>()?;
                let name: Ident = content.parse()?;
                content.parse::<Token![=]>()?;
                let init = parse_expr(&content)?;
                content.parse::<Token![;]>()?;
                Some(StateDecl { name, init })
            }
            false => None,
        };
        let shape = match opens_an_adaptor_flow(&content)? {
            true => Shape::Adaptor(parse_flow(&content)?),
            false => {
                let body = parse_block(&content, BlockCloser::Finish)?;
                if !next_word_is(&content, &["finish"]) {
                    return Err(syn::Error::new(
                        content.span(),
                        "a consumer's step ends in `finish e`",
                    ));
                }
                content.parse::<Ident>()?;
                Shape::Consumer {
                    body,
                    finish: parse_expr(&content)?,
                }
            }
        };
        if !content.is_empty() {
            return Err(syn::Error::new(
                content.span(),
                "unexpected tokens after the step: an adaptor states one flow, and a consumer \
                 ends in `finish e`",
            ));
        }
        if let (Some(state), Shape::Adaptor(_)) = (&state, &shape) {
            return Err(syn::Error::new(
                state.name.span(),
                "an adaptor's step keeps no state: `state` is a consumer's (RFC-0099 rule 1)",
            ));
        }
        Ok(StepAttr {
            keyword,
            state,
            shape,
        })
    }
}

fn opens_an_adaptor_flow(input: ParseStream) -> syn::Result<bool> {
    if input.peek(Token![yield]) || next_word_is(input, &ADAPTOR_ENDS) {
        return Ok(true);
    }
    if !input.peek(Token![if]) {
        return Ok(false);
    }
    let fork = input.fork();
    fork.parse::<Token![if]>()?;
    parse_expr(&fork)?;
    let arm;
    syn::braced!(arm in fork);
    opens_an_adaptor_flow(&arm)
}

fn parse_flow(input: ParseStream) -> syn::Result<Flow> {
    if input.peek(Token![yield]) {
        input.parse::<Token![yield]>()?;
        return Ok(Flow::Yield(parse_expr(input)?));
    }
    if input.peek(Token![if]) {
        input.parse::<Token![if]>()?;
        let cond = parse_expr(input)?;
        let then;
        syn::braced!(then in input);
        let then_flow = parse_flow(&then)?;
        flow_ended(&then)?;
        if !input.peek(Token![else]) {
            return Err(syn::Error::new(
                input.span(),
                "an adaptor's `if` has an `else`: every path of its flow ends in `yield`, \
                 `skip`, `done` or `nest`",
            ));
        }
        input.parse::<Token![else]>()?;
        let otherwise;
        syn::braced!(otherwise in input);
        let otherwise_flow = parse_flow(&otherwise)?;
        flow_ended(&otherwise)?;
        return Ok(Flow::If {
            cond,
            then: Box::new(then_flow),
            otherwise: Box::new(otherwise_flow),
        });
    }
    let word: Ident = input.parse()?;
    match word.to_string().as_str() {
        "skip" => Ok(Flow::Skip),
        "done" => Ok(Flow::Done),
        "nest" => Ok(Flow::Nest(parse_expr(input)?)),
        other => Err(syn::Error::new(
            word.span(),
            format!(
                "`{other}` is no end of an adaptor's flow: a flow ends in `yield e`, `skip`, \
                 `done` or `nest e` (RFC-0099 rule 1)"
            ),
        )),
    }
}

fn flow_ended(input: ParseStream) -> syn::Result<()> {
    if input.peek(Token![;]) {
        input.parse::<Token![;]>()?;
    }
    match input.is_empty() {
        true => Ok(()),
        false => Err(syn::Error::new(
            input.span(),
            "an adaptor's flow ends at its first `yield`, `skip`, `done` or `nest`",
        )),
    }
}

fn parse_block(input: ParseStream, closer: BlockCloser) -> syn::Result<Block> {
    let closed = |input: ParseStream| {
        input.is_empty() || (closer == BlockCloser::Finish && next_word_is(input, &["finish"]))
    };
    let mut stmts = Vec::new();
    while !closed(input) {
        if input.peek(Token![break]) {
            input.parse::<Token![break]>()?;
            let breaks = parse_expr(input)?;
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
            }
            if !closed(input) {
                return Err(syn::Error::new(
                    input.span(),
                    "`break e` ends its block: nothing follows it",
                ));
            }
            return Ok(Block {
                stmts,
                breaks: Some(breaks),
            });
        }
        stmts.push(parse_stmt(input)?);
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
        } else if !closed(input) {
            return Err(syn::Error::new(
                input.span(),
                "a term is a call of a closure or a registered extern, a constant, `x`, the \
                 state, a parameter, `&x`, `&s`, or `a +% b` (RFC-0099 rule 1)",
            ));
        }
    }
    Ok(Block {
        stmts,
        breaks: None,
    })
}

fn parse_stmt(input: ParseStream) -> syn::Result<Stmt> {
    if input.peek(Token![if]) {
        input.parse::<Token![if]>()?;
        let cond = parse_expr(input)?;
        let then;
        syn::braced!(then in input);
        let then = parse_block(&then, BlockCloser::Brace)?;
        let otherwise = match input.peek(Token![else]) {
            true => {
                input.parse::<Token![else]>()?;
                let otherwise;
                syn::braced!(otherwise in input);
                parse_block(&otherwise, BlockCloser::Brace)?
            }
            false => Block {
                stmts: Vec::new(),
                breaks: None,
            },
        };
        return Ok(Stmt::If {
            cond,
            then,
            otherwise,
        });
    }
    if input.peek(Token![yield]) || next_word_is(input, &ADAPTOR_ENDS) {
        return Err(syn::Error::new(
            input.span(),
            "`yield`, `skip`, `done` and `nest` are an adaptor's, and this step is a \
             consumer's: it ends in `finish e`",
        ));
    }
    let callee: Path = input.parse()?;
    if input.peek(Token![=]) {
        input.parse::<Token![=]>()?;
        let Some(state) = callee.get_ident() else {
            return Err(syn::Error::new_spanned(callee, "a state update is written `s = e`"));
        };
        return Ok(Stmt::Set {
            state: state.clone(),
            value: parse_expr(input)?,
        });
    }
    if input.peek(syn::token::Paren) {
        let args;
        syn::parenthesized!(args in input);
        return Ok(Stmt::Run {
            callee,
            args: parse_args(&args)?,
        });
    }
    Err(syn::Error::new_spanned(
        callee,
        "a consumer's statement is `s = e`, a call `g(&mut s, ..)`, an `if`, or a closing \
         `break e` (RFC-0099 rule 1)",
    ))
}

fn parse_expr(input: ParseStream) -> syn::Result<Expr> {
    let mut expr = parse_primary(input)?;
    while input.peek(Token![+]) && input.peek2(Token![%]) {
        input.parse::<Token![+]>()?;
        input.parse::<Token![%]>()?;
        let right = parse_primary(input)?;
        expr = Expr::WrappingAdd {
            left: Box::new(expr),
            right: Box::new(right),
        };
    }
    Ok(expr)
}

fn parse_primary(input: ParseStream) -> syn::Result<Expr> {
    if input.peek(Token![&]) {
        input.parse::<Token![&]>()?;
        let mutable = input.peek(Token![mut]);
        if mutable {
            input.parse::<Token![mut]>()?;
        }
        return Ok(Expr::Lend {
            mutable,
            of: Box::new(parse_primary(input)?),
        });
    }
    if input.peek(syn::token::Paren) {
        let inner;
        syn::parenthesized!(inner in input);
        return parse_expr(&inner);
    }
    let negative = input.peek(Token![-]);
    if negative {
        input.parse::<Token![-]>()?;
    }
    if negative || input.peek(Lit) {
        return Ok(Expr::Lit {
            negative,
            lit: input.parse()?,
        });
    }
    let callee: Path = input.parse()?;
    if input.peek(syn::token::Paren) {
        let args;
        syn::parenthesized!(args in input);
        return Ok(Expr::Call {
            callee,
            args: parse_args(&args)?,
        });
    }
    match callee.get_ident() {
        Some(name) => Ok(Expr::Name(name.clone())),
        None => Err(syn::Error::new_spanned(
            callee,
            "a path names an extern, which a step calls: `ns::g(..)`",
        )),
    }
}

fn parse_args(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let mut args = Vec::new();
    while !input.is_empty() {
        args.push(parse_expr(input)?);
        if !input.is_empty() {
            input.parse::<Token![,]>()?;
        }
    }
    Ok(args)
}

#[derive(Clone, Copy)]
enum Named {
    Element,
    State,
    ValueParam(usize),
    Closure { param: usize, arity: usize },
    Stream,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Site {
    StateStartOrFinish,
    PerElement,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Use {
    Moved,
    Lent,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StateMoves {
    AtMostOnce,
    None,
}

pub(crate) struct PulledStream {
    pub(crate) acvus_param: usize,
    pub(crate) requirement: usize,
}

struct Scope<'a> {
    fn_ident: &'a Ident,
    params: &'a [&'a ExternParam],
    stream: usize,
    state: Option<&'a Ident>,
    moved_value_params: RefCell<Vec<usize>>,
}

/// How many times one path of the step moves the element and the state.
#[derive(Clone, Copy, Default)]
struct Moves {
    element: usize,
    state: usize,
}

impl StepAttr {
    pub(crate) fn checked(
        &self,
        fn_ident: &Ident,
        params: &[&ExternParam],
        pulled: &[PulledStream],
        returning: &Returning,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let [stream] = pulled else {
            return Err(syn::Error::new(
                self.keyword.span(),
                format!(
                    "a step is stated on a declaration that pulls one stream, one `Instance` \
                     parameter, and `{fn_ident}` pulls {} (RFC-0099 rule 1)",
                    pulled.len()
                ),
            ));
        };
        if !matches!(returning, Returning::Value) {
            return Err(syn::Error::new(
                self.keyword.span(),
                format!(
                    "a step is stated on a declaration that returns a value, and `{fn_ident}` \
                     returns a view or a place (RFC-0099 rule 1)"
                ),
            ));
        }
        let state_name = self.state.as_ref().map(|state| &state.name);
        if let Some(param) = params
            .iter()
            .find(|param| param.name == ELEMENT || state_name.is_some_and(|s| *s == param.name))
        {
            return Err(syn::Error::new(
                self.keyword.span(),
                format!(
                    "the parameter `{}` of `{fn_ident}` shadows a name the step gives: `x` is \
                     the element and the state is named by `state`",
                    param.name
                ),
            ));
        }
        let scope = Scope {
            fn_ident,
            params,
            stream: stream.acvus_param,
            state: state_name,
            moved_value_params: RefCell::new(Vec::new()),
        };
        let param_at = stream.acvus_param;
        let requirement = stream.requirement;
        let stream = quote! {
            ::acvus_extern::step::StreamParam {
                param: #param_at,
                requirement: #requirement,
            }
        };
        match &self.shape {
            Shape::Adaptor(flow) => {
                let flow = scope.flow(flow, Moves::default())?;
                Ok(quote! {
                    ::acvus_extern::Laws::Step(::acvus_extern::step::Step::Adaptor {
                        stream: #stream,
                        flow: #flow,
                    })
                })
            }
            Shape::Consumer { body, finish } => {
                let state = match &self.state {
                    Some(state) => {
                        let init = scope.whole_term(&state.init, StateMoves::None)?;
                        quote! { ::core::option::Option::Some(#init) }
                    }
                    None => quote! { ::core::option::Option::None },
                };
                let body = scope.block(body, &mut Moves::default())?;
                let finish = scope.whole_term(finish, StateMoves::AtMostOnce)?;
                Ok(quote! {
                    ::acvus_extern::Laws::Step(::acvus_extern::step::Step::Consumer {
                        stream: #stream,
                        state: #state,
                        body: #body,
                        finish: #finish,
                    })
                })
            }
        }
    }
}

impl Scope<'_> {
    fn lookup(&self, name: &Ident) -> Option<Named> {
        if name == ELEMENT {
            return Some(Named::Element);
        }
        if self.state.is_some_and(|state| state == name) {
            return Some(Named::State);
        }
        let at = self.params.iter().position(|param| *name == param.name)?;
        if at == self.stream {
            return Some(Named::Stream);
        }
        Some(match closure_arity(&self.params[at].ty) {
            Some(arity) => Named::Closure { param: at, arity },
            None => Named::ValueParam(at),
        })
    }

    fn named(&self, name: &Ident) -> syn::Result<Named> {
        self.lookup(name).ok_or_else(|| {
            syn::Error::new(
                name.span(),
                format!(
                    "`{name}` names no parameter of `{}`, the element `x`, or the state",
                    self.fn_ident
                ),
            )
        })
    }

    fn flow(&self, flow: &Flow, moves: Moves) -> syn::Result<proc_macro2::TokenStream> {
        let path = quote! { ::acvus_extern::step::AdaptorFlow };
        let mut moves = moves;
        Ok(match flow {
            Flow::Yield(e) => {
                let e = self.per_element(e, &mut moves, StateMoves::None)?;
                quote! { #path::Yield(#e) }
            }
            Flow::Skip => quote! { #path::Skip },
            Flow::Done => quote! { #path::Done },
            Flow::Nest(e) => {
                let e = self.per_element(e, &mut moves, StateMoves::None)?;
                quote! { #path::Nest(#e) }
            }
            Flow::If {
                cond,
                then,
                otherwise,
            } => {
                let cond = self.per_element(cond, &mut moves, StateMoves::None)?;
                let then = self.flow(then, moves)?;
                let otherwise = self.flow(otherwise, moves)?;
                quote! {
                    #path::If {
                        cond: #cond,
                        then: ::std::boxed::Box::new(#then),
                        otherwise: ::std::boxed::Box::new(#otherwise),
                    }
                }
            }
        })
    }

    fn block(&self, block: &Block, moves: &mut Moves) -> syn::Result<proc_macro2::TokenStream> {
        let mut stmts = Vec::new();
        for stmt in &block.stmts {
            stmts.push(match stmt {
                Stmt::Set { state, value } => {
                    if !matches!(self.named(state)?, Named::State) {
                        return Err(syn::Error::new(
                            state.span(),
                            format!("`{state} = e` updates the state, and `{state}` is not it"),
                        ));
                    }
                    let value = self.per_element(value, moves, StateMoves::AtMostOnce)?;
                    quote! { ::acvus_extern::step::ConsumerStmt::Set(#value) }
                }
                Stmt::Run { callee, args } => {
                    let lends_the_state = args.iter().any(|arg| self.lends_the_state_mutably(arg));
                    if !lends_the_state {
                        return Err(syn::Error::new_spanned(
                            callee,
                            "a call standing as a statement updates the state it lends: \
                             `g(&mut s, ..)`",
                        ));
                    }
                    let name = crate::law::qref_of(callee)?;
                    let args = args
                        .iter()
                        .map(|arg| match self.lends_the_state_mutably(arg) {
                            true => Ok(quote! {
                                ::acvus_extern::step::Term::Lend(
                                    ::acvus_extern::Mutability::Mut,
                                    ::std::boxed::Box::new(::acvus_extern::step::Term::State),
                                )
                            }),
                            false => self.per_element(arg, moves, StateMoves::None),
                        })
                        .collect::<syn::Result<Vec<_>>>()?;
                    quote! {
                        ::acvus_extern::step::ConsumerStmt::Run {
                            name: #name,
                            args: vec![#(#args),*],
                        }
                    }
                }
                Stmt::If {
                    cond,
                    then,
                    otherwise,
                } => {
                    let cond = self.per_element(cond, moves, StateMoves::None)?;
                    let mut then_moves = *moves;
                    let then = self.block(then, &mut then_moves)?;
                    let mut otherwise_moves = *moves;
                    let otherwise = self.block(otherwise, &mut otherwise_moves)?;
                    moves.element = then_moves.element.max(otherwise_moves.element);
                    quote! {
                        ::acvus_extern::step::ConsumerStmt::If {
                            cond: #cond,
                            then: #then,
                            otherwise: #otherwise,
                        }
                    }
                }
            });
        }
        let breaks = match &block.breaks {
            Some(e) => {
                let e = self.per_element(e, moves, StateMoves::AtMostOnce)?;
                quote! { ::core::option::Option::Some(#e) }
            }
            None => quote! { ::core::option::Option::None },
        };
        Ok(quote! {
            ::acvus_extern::step::ConsumerBlock {
                stmts: vec![#(#stmts),*],
                breaks: #breaks,
            }
        })
    }

    fn lends_the_state_mutably(&self, arg: &Expr) -> bool {
        let Expr::Lend { mutable: true, of } = arg else {
            return false;
        };
        let Expr::Name(name) = &**of else {
            return false;
        };
        matches!(self.lookup(name), Some(Named::State))
    }

    /// The element may be moved once on each path through the step, and
    /// the state once within the term whose value replaces it.
    fn per_element(
        &self,
        e: &Expr,
        moves: &mut Moves,
        state_moves: StateMoves,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let mut here = Moves {
            element: moves.element,
            state: 0,
        };
        let term = self.walk(e, Site::PerElement, &mut here, Use::Moved)?;
        self.state_moved_within(here.state, state_moves)?;
        if here.element > 1 {
            return Err(syn::Error::new(
                self.fn_ident.span(),
                "a step moves the element `x` once on each path: lend it with `&x` where it \
                 reads it again",
            ));
        }
        moves.element = here.element;
        Ok(term)
    }

    fn whole_term(&self, e: &Expr, state_moves: StateMoves) -> syn::Result<proc_macro2::TokenStream> {
        let mut here = Moves::default();
        let term = self.walk(e, Site::StateStartOrFinish, &mut here, Use::Moved)?;
        self.state_moved_within(here.state, state_moves)?;
        Ok(term)
    }

    fn state_moved_within(&self, moved: usize, allowed: StateMoves) -> syn::Result<()> {
        let most = match allowed {
            StateMoves::AtMostOnce => 1,
            StateMoves::None => 0,
        };
        match moved <= most {
            true => Ok(()),
            false => Err(syn::Error::new(
                self.fn_ident.span(),
                "a step moves the state only where its value becomes the state again or the \
                 result, `s = e`, `break e` or `finish e`, once",
            )),
        }
    }

    fn walk(
        &self,
        e: &Expr,
        site: Site,
        moves: &mut Moves,
        used: Use,
    ) -> syn::Result<proc_macro2::TokenStream> {
        let term = quote! { ::acvus_extern::step::Term };
        match e {
            Expr::Name(name) => match self.named(name)? {
                Named::Element => {
                    if site == Site::StateStartOrFinish {
                        return Err(syn::Error::new(
                            name.span(),
                            "`x` is an element, and the state's start and `finish` have none",
                        ));
                    }
                    if used == Use::Moved {
                        moves.element += 1;
                    }
                    Ok(quote! { #term::Elem })
                }
                Named::State => {
                    if used == Use::Moved {
                        moves.state += 1;
                    }
                    Ok(quote! { #term::State })
                }
                Named::ValueParam(at) => {
                    if used == Use::Moved {
                        self.move_value_param(name, at, site)?;
                    }
                    Ok(quote! { #term::ValueParam(#at) })
                }
                Named::Closure { .. } => Err(syn::Error::new(
                    name.span(),
                    format!("`{name}` is a closure, which a step calls: `{name}(..)`"),
                )),
                Named::Stream => Err(syn::Error::new(
                    name.span(),
                    format!("`{name}` is the stream the step pulls, which no term reads"),
                )),
            },
            Expr::Call { callee, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.walk(arg, site, moves, Use::Moved))
                    .collect::<syn::Result<Vec<_>>>()?;
                let Some(name) = callee.get_ident() else {
                    let name = crate::law::qref_of(callee)?;
                    return Ok(quote! { #term::CallExtern { name: #name, args: vec![#(#args),*] } });
                };
                match self.lookup(name) {
                    None => {
                        let name = crate::law::qref_of(callee)?;
                        Ok(quote! { #term::CallExtern { name: #name, args: vec![#(#args),*] } })
                    }
                    Some(Named::Closure { param, arity }) if arity == args.len() => {
                        Ok(quote! { #term::CallClosure { param: #param, args: vec![#(#args),*] } })
                    }
                    Some(Named::Closure { arity, .. }) => Err(syn::Error::new(
                        name.span(),
                        format!(
                            "`{name}` takes {arity} arguments, and is called with {}",
                            args.len()
                        ),
                    )),
                    Some(
                        Named::Element | Named::State | Named::ValueParam(_) | Named::Stream,
                    ) => Err(syn::Error::new(
                        name.span(),
                        format!(
                            "`{name}` is no closure, and a step calls only closures and \
                             registered externs"
                        ),
                    )),
                }
            }
            Expr::Lit { negative, lit } => {
                let literal = constant(*negative, lit)?;
                Ok(quote! { #term::Const(#literal) })
            }
            Expr::Lend { mutable: true, .. } => Err(syn::Error::new(
                self.fn_ident.span(),
                "`&mut s` is an argument of a call standing as a statement, `g(&mut s, ..)`",
            )),
            Expr::Lend { mutable: false, of } => {
                let lends_a_slot = match &**of {
                    Expr::Name(name) => {
                        matches!(self.lookup(name), Some(Named::Element | Named::State))
                    }
                    _ => false,
                };
                if !lends_a_slot {
                    return Err(syn::Error::new(
                        self.fn_ident.span(),
                        "`&` lends the element or the state",
                    ));
                }
                let of = self.walk(of, site, moves, Use::Lent)?;
                Ok(quote! {
                    #term::Lend(::acvus_extern::Mutability::Shared, ::std::boxed::Box::new(#of))
                })
            }
            Expr::WrappingAdd { left, right } => {
                let left = self.walk(left, site, moves, Use::Moved)?;
                let right = self.walk(right, site, moves, Use::Moved)?;
                Ok(quote! {
                    #term::WrappingAdd {
                        left: ::std::boxed::Box::new(#left),
                        right: ::std::boxed::Box::new(#right),
                    }
                })
            }
        }
    }

    fn move_value_param(&self, name: &Ident, at: usize, site: Site) -> syn::Result<()> {
        if site == Site::PerElement {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "`{name}` is moved once, and a per-element step runs many times: move it \
                     in the state's start or `finish`"
                ),
            ));
        }
        let mut moved = self.moved_value_params.borrow_mut();
        if moved.contains(&at) {
            return Err(syn::Error::new(name.span(), format!("`{name}` is moved twice")));
        }
        moved.push(at);
        Ok(())
    }
}

fn constant(negative: bool, lit: &Lit) -> syn::Result<proc_macro2::TokenStream> {
    match lit {
        Lit::Int(int) => {
            let magnitude: i128 = int.base10_parse()?;
            let value = match negative {
                true => -magnitude,
                false => magnitude,
            };
            Ok(quote! { ::acvus_extern::Literal::Int(#value) })
        }
        Lit::Float(float) => {
            let magnitude: f64 = float.base10_parse()?;
            let value = match negative {
                true => -magnitude,
                false => magnitude,
            };
            Ok(quote! { ::acvus_extern::Literal::Float(#value) })
        }
        Lit::Bool(b) if !negative => Ok(quote! { ::acvus_extern::Literal::Bool(#b) }),
        _ => Err(syn::Error::new_spanned(
            lit,
            "a step's constant is an integer, a float, or a bool",
        )),
    }
}

fn closure_arity(ty: &Type) -> Option<usize> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Closure" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        syn::GenericArgument::Type(Type::Tuple(tuple)) => Some(tuple.elems.len()),
        _ => None,
    })
}
