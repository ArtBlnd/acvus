//! Where the content a store overwrites ends (ownership §6: a value ends at
//! exactly one place). In a place the body owns, it ends where its
//! liveness ends, whether the store names the place or reaches it through
//! a reference to it alone. In a context or storage outside the body, it
//! ends at the last point it is known to be that content: just before the
//! store where it is there on every path, and on each edge into the paths'
//! meeting where it is there, named from a root, where it is there on some
//! paths only. Among the places a reference may reach, it ends at the
//! store, and must be there at each; a content that cannot end so refuses
//! the store.

use std::collections::HashMap;

use acvus_mir::ty::{IntTy, Mutability, Poly, TyId, TyKind};
use acvus_mir_test::Case;

fn string(case: &mut Case) -> TyId<Poly> {
    case.ty(TyKind::String)
}

fn exclusive(case: &mut Case, target: TyId<Poly>) -> TyId<Poly> {
    case.ty(TyKind::Ref(Mutability::Mut, target))
}

/// The instructions of the one body printed, one per line, trimmed.
fn lines(printed: &str) -> Vec<&str> {
    printed.lines().map(str::trim).collect()
}

/// For each store through a reference, whether the two instructions just
/// before it take out through that same reference and drop what they took.
fn dropped_before_each_store(printed: &str) -> Vec<bool> {
    let lines = lines(printed);
    let mut dropped = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        let Some(store) = line.split("Assign *").nth(1) else {
            continue;
        };
        let reference = store.split(|c: char| c == ',' || c == '.').next().expect("a store names its reference");
        let taken = at.checked_sub(2).map(|before| lines[before]);
        let drop = at.checked_sub(1).map(|before| lines[before]);
        let pair = match (taken, drop) {
            (Some(taken), Some(drop)) => match taken.split_once(" = Take *") {
                Some((result, from)) => {
                    let result = result.split(':').next().expect("a take names its result");
                    from.starts_with(reference) && drop.ends_with(&format!("Drop {result}"))
                }
                None => false,
            },
            _ => false,
        };
        dropped.push(pair);
    }
    dropped
}

// -- Where each content ends, by running the printed body ---------------------

/// What an entry input is: a `String`, whose content is named by the
/// input, a `&mut String`, whose target's content outside the body is named
/// `*` and the input, or a word; or a `String` context, whose content is
/// named `@` and the context and is no parameter. The body's parameters are
/// its other inputs in the order the source first names them, and a run
/// takes them so.
#[derive(Clone, Copy)]
enum Input<'a> {
    Text(&'a str),
    Lends(&'a str),
    Word,
    Context(&'a str),
}

/// What a value or a place holds while the body runs: a `String` content
/// by its name, a reference to a place by its name, or nothing tracked.
#[derive(Debug, Clone, PartialEq)]
enum Held {
    Text(String),
    Lends(String),
    Nothing,
}

struct Block<'p> {
    params: Vec<&'p str>,
    lines: Vec<&'p str>,
}

/// The blocks of the one body printed, by name.
fn blocks(printed: &str) -> HashMap<&str, Block<'_>> {
    let mut blocks: HashMap<&str, Block<'_>> = HashMap::new();
    let mut current: Option<&str> = None;
    for line in lines(printed) {
        if line.starts_with('b') && line.ends_with("):") && !line.starts_with("body") {
            let (name, params) = line.trim_end_matches("):").split_once('(').expect("a block header");
            let params = params.split(", ").filter(|param| !param.is_empty()).map(|param| param.split(':').next().expect("a parameter")).collect();
            blocks.insert(name, Block { params, lines: Vec::new() });
            current = Some(name);
        } else if let Some(name) = current
            && !line.is_empty()
        {
            blocks.get_mut(name).expect("the current block").lines.push(line);
        }
    }
    blocks
}

/// A target block and its arguments, as a terminator writes it.
fn edge(written: &str) -> (&str, Vec<&str>) {
    let (to, args) = written.trim().trim_end_matches(')').split_once('(').expect("an edge");
    (to, args.split(", ").filter(|arg| !arg.is_empty()).collect())
}

/// What one run down the arms `choices` picks leaves: the contents ended
/// by a drop or a return, in order; the contents a store replaced that no
/// one ended; the places a `String` was taken out of while they held none;
/// and what each place holds at the return. `None` where the run takes more
/// arms than `choices` names or diverges.
struct Ran {
    ended: Vec<String>,
    lost: Vec<String>,
    emptied: Vec<String>,
    places: HashMap<String, Held>,
}

fn run(printed: &str, inputs: &[Input<'_>], choices: &[bool]) -> Option<Ran> {
    let blocks = blocks(printed);
    let mut values: HashMap<String, Held> = HashMap::new();
    let mut places: HashMap<String, Held> = HashMap::new();
    let mut params = 0;
    for input in inputs {
        let held = match input {
            Input::Text(name) => Held::Text((*name).to_string()),
            Input::Lends(name) => {
                places.insert(format!("*{name}"), Held::Text(format!("*{name}")));
                Held::Lends(format!("*{name}"))
            }
            Input::Word => Held::Nothing,
            Input::Context(name) => {
                places.insert(format!("@{name}"), Held::Text(format!("@{name}")));
                continue;
            }
        };
        values.insert(format!("p{params}"), held);
        params += 1;
    }
    let (mut ended, mut lost, mut emptied) = (Vec::new(), Vec::new(), Vec::new());
    let mut choices = choices.iter();
    let mut block = "b0";
    loop {
        let record = &blocks[block];
        let mut next = None;
        for line in &record.lines {
            let place_of = |values: &HashMap<String, Held>, written: &str| match written.strip_prefix('*') {
                Some(reference) => match values.get(reference) {
                    Some(Held::Lends(place)) => place.clone(),
                    held => panic!("{reference} lends no place: {held:?}"),
                },
                None => written.to_string(),
            };
            if let Some(rest) = line.strip_prefix("Assign ") {
                let (target, value) = rest.split_once(", ").expect("a store names its place and value");
                let place = place_of(&values, target);
                let held = values.remove(value).unwrap_or(Held::Nothing);
                if let Some(Held::Text(old)) = places.insert(place, held) {
                    lost.push(old);
                }
            } else if let Some(value) = line.strip_prefix("Drop ").or_else(|| line.strip_prefix("Return ")) {
                if let Some(Held::Text(content)) = values.remove(value) {
                    ended.push(content);
                }
                if line.starts_with("Return") {
                    return Some(Ran { ended, lost, emptied, places });
                }
            } else if line.starts_with("Diverge") {
                return None;
            } else if let Some(rest) = line.strip_prefix("Jump ") {
                next = Some(edge(rest));
            } else if let Some(rest) = line.strip_prefix("Diamond ") {
                let (_, arms) = rest.split_once(" ? ").expect("a diamond's arms");
                let (then, otherwise) = arms.split_once(" : ").expect("a diamond's two arms");
                next = Some(edge(if *choices.next()? { then } else { otherwise }));
            } else if let Some((result, op)) = line.split_once(" = ") {
                let (result, ty) = result.split_once(": ").expect("a result and its type");
                let result = result.to_string();
                let held = if let Some(target) = op.strip_prefix("Take ") {
                    let place = place_of(&values, target);
                    let held = places.get(&place).cloned().unwrap_or(Held::Nothing);
                    if ty == "String" && !matches!(held, Held::Text(_)) {
                        emptied.push(place.clone());
                    }
                    if matches!(held, Held::Text(_)) {
                        places.insert(place, Held::Nothing);
                    }
                    held
                } else if let Some(target) = op.strip_prefix("Ref(mut) ") {
                    Held::Lends(target.to_string())
                } else {
                    Held::Nothing
                };
                values.insert(result, held);
            }
        }
        let (to, args) = next.expect("a block ends in a terminator");
        let passed: Vec<Held> = args.iter().map(|arg| values.remove(*arg).unwrap_or(Held::Nothing)).collect();
        for (param, held) in blocks[to].params.iter().zip(passed) {
            values.insert((*param).to_string(), held);
        }
        block = to;
    }
}

/// On every run of the printed body, down each choice of arms, each
/// content the inputs and the stores bring ends exactly once: a drop or the
/// return ends it, or storage outside the body or a context holds it; no
/// `String` is taken out of a place that holds none, no store replaces a
/// content no one ended, and no place the body owns holds one at the
/// return.
fn assert_each_content_ends_once(printed: &str, inputs: &[Input<'_>]) {
    let names: Vec<String> = inputs
        .iter()
        .filter_map(|input| match input {
            Input::Text(name) => Some((*name).to_string()),
            Input::Lends(name) => Some(format!("*{name}")),
            Input::Context(name) => Some(format!("@{name}")),
            Input::Word => None,
        })
        .collect();
    let mut runs = 0;
    for arms in 0..4usize {
        for pick in 0..(1usize << arms) {
            let choices: Vec<bool> = (0..arms).map(|bit| pick & (1 << bit) != 0).collect();
            let Some(Ran { ended, lost, emptied, places }) = run(printed, inputs, &choices) else {
                continue;
            };
            runs += 1;
            assert!(emptied.is_empty(), "down {choices:?}, a String is taken out of {emptied:?}, which held none:\n{printed}");
            assert!(lost.is_empty(), "down {choices:?}, a store replaced {lost:?}, which nothing ended:\n{printed}");
            let held: Vec<String> = places
                .iter()
                .filter_map(|(place, held)| match held {
                    Held::Text(content) => {
                        assert!(
                            place.starts_with('*') || place.starts_with('@'),
                            "down {choices:?}, {place} still holds {content} at the return:\n{printed}"
                        );
                        Some(content.clone())
                    }
                    Held::Lends(_) | Held::Nothing => None,
                })
                .collect();
            for name in &names {
                let ends = ended.iter().chain(&held).filter(|content| *content == name).count();
                assert_eq!(ends, 1, "down {choices:?}, {name} ends {ends} times; ended {ended:?}, held {held:?}:\n{printed}");
            }
        }
    }
    assert!(runs > 0, "no run of the body returns:\n{printed}");
}

// -- Storage outside the body ----------------------------------------------------

#[test]
fn a_string_stored_through_a_parameter_drops_what_it_replaces_at_the_store() {
    let mut case = Case::new();
    let text = string(&mut case);
    let to_text = exclusive(&mut case, text);
    let main = case.script(&["app"], "main", "*$r = $s;");
    case.entry(main, &[("r", to_text), ("s", text)], None);
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [true], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Lends("r"), Input::Text("s")]);
}

#[test]
fn a_copy_stored_through_a_parameter_drops_nothing() {
    let mut case = Case::new();
    let int = case.ty(TyKind::Int(IntTy::I64));
    let to_int = exclusive(&mut case, int);
    let main = case.script(&["app"], "main", "*$r = $n;");
    case.entry(main, &[("r", to_int), ("n", int)], None);
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [false], "{printed}");
    assert!(!printed.contains(" = Take *"), "{printed}");
}

#[test]
fn a_store_through_a_parameter_after_a_move_through_it_is_a_refill() {
    let mut case = Case::new();
    let text = string(&mut case);
    let to_text = exclusive(&mut case, text);
    let main = case.script(&["app"], "main", "let x = *$r; *$r = $s; x");
    case.entry(main, &[("r", to_text), ("s", text)], Some(text));
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [false], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Lends("r"), Input::Text("s")]);
}

#[test]
fn a_second_store_through_a_parameter_drops_the_first() {
    let mut case = Case::new();
    let text = string(&mut case);
    let to_text = exclusive(&mut case, text);
    let main = case.script(&["app"], "main", "let x = *$r; *$r = $s; *$r = $t; x");
    case.entry(main, &[("r", to_text), ("s", text), ("t", text)], Some(text));
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [false, true], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Lends("r"), Input::Text("s"), Input::Text("t")]);
}

#[test]
fn a_store_through_a_join_of_two_parameters_drops_what_it_replaces_at_the_store() {
    let mut case = Case::new();
    let text = string(&mut case);
    let to_text = exclusive(&mut case, text);
    let condition = case.ty(TyKind::Bool);
    let main = case.script(&["app"], "main", "let r = if $c { $p } else { $q }; *r = $s;");
    case.entry(main, &[("c", condition), ("p", to_text), ("q", to_text), ("s", text)], None);
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [true], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Word, Input::Lends("p"), Input::Lends("q"), Input::Text("s")]);
}

#[test]
fn a_store_through_a_parameter_over_content_moved_on_one_path_only_drops_it_on_the_edge_where_it_is_held() {
    let mut case = Case::new();
    let text = string(&mut case);
    let to_text = exclusive(&mut case, text);
    let condition = case.ty(TyKind::Bool);
    let main = case.script(&["app"], "main", "let x = if $c { *$r } else { $t }; *$r = $s; x");
    case.entry(main, &[("c", condition), ("r", to_text), ("s", text), ("t", text)], Some(text));
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [false], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Word, Input::Lends("r"), Input::Text("t"), Input::Text("s")]);
}

#[test]
fn a_reference_made_after_the_meeting_reaches_the_place_the_edge_names() {
    let mut case = Case::new();
    let text = string(&mut case);
    let to_text = exclusive(&mut case, text);
    let condition = case.ty(TyKind::Bool);
    let main = case.script(&["app"], "main", "let x = if $c { *$r } else { $t }; let q = &mut *$r; *q = $s; x");
    case.entry(main, &[("c", condition), ("r", to_text), ("s", text), ("t", text)], Some(text));
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [false], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Word, Input::Lends("r"), Input::Text("t"), Input::Text("s")]);
}

#[test]
fn a_one_place_store_no_root_names_on_the_edge_is_refused() {
    let mut case = Case::new();
    let text = string(&mut case);
    let to_text = exclusive(&mut case, text);
    let condition = case.ty(TyKind::Bool);
    let source = "let x = if $c { *$r } else { $t }; let q = if $d { $r } else { $r }; *q = $s; x";
    let main = case.script(&["app"], "main", source);
    case.entry(main, &[("c", condition), ("d", condition), ("r", to_text), ("s", text), ("t", text)], Some(text));
    let moved = source.find("*$r").expect("the move");
    let stored = source.find("*q = $s").expect("the store");
    let refused = case.refused();
    let expected = format!("app::main: StoreOverContentMaybeMoved moved @{moved}..{} @{stored}..", moved + 3);
    assert!(refused.starts_with(&expected) && refused.lines().count() == 1, "{refused}");
}

// -- A context -------------------------------------------------------------------

#[test]
fn a_context_taken_on_one_path_only_is_dropped_by_its_name_on_the_edge_where_it_is_held() {
    let mut case = Case::new();
    let text = string(&mut case);
    let condition = case.ty(TyKind::Bool);
    case.context("n", text);
    let main = case.script(&["app"], "main", "let x = if $c { @n } else { $t }; @n = $s; x");
    case.entry(main, &[("c", condition), ("t", text), ("s", text)], Some(text));
    let printed = case.admitted();
    assert_each_content_ends_once(&printed, &[Input::Context("n"), Input::Word, Input::Text("t"), Input::Text("s")]);
}

// -- Storage the body owns -------------------------------------------------------

#[test]
fn a_store_through_a_reference_to_a_binding_ends_the_old_content_by_its_liveness() {
    let mut case = Case::new();
    let text = string(&mut case);
    let main = case.script(&["app"], "main", "let b = $t; let r = &mut b; *r = $u; b");
    case.entry(main, &[("t", text), ("u", text)], Some(text));
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [false], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Text("t"), Input::Text("u")]);
}

#[test]
fn a_content_stored_through_a_reference_to_a_binding_no_read_follows_ends_at_the_store() {
    let mut case = Case::new();
    let text = string(&mut case);
    let int = case.ty(TyKind::Int(IntTy::I64));
    let main = case.script(&["app"], "main", "let b = $t; let r = &mut b; *r = $u; 0");
    case.entry(main, &[("t", text), ("u", text)], Some(int));
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [false], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Text("t"), Input::Text("u")]);
}

#[test]
fn a_store_through_a_join_of_two_bindings_drops_what_it_replaces_at_the_store() {
    let mut case = Case::new();
    let text = string(&mut case);
    let int = case.ty(TyKind::Int(IntTy::I64));
    let condition = case.ty(TyKind::Bool);
    let main = case.script(&["app"], "main", "let a = $s; let b = $t; let r = if $c { &mut a } else { &mut b }; *r = $u; 0");
    case.entry(main, &[("c", condition), ("s", text), ("t", text), ("u", text)], Some(int));
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [true], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Text("s"), Input::Text("t"), Input::Word, Input::Text("u")]);
}

#[test]
fn a_store_through_a_join_of_two_bindings_one_read_after_drops_each_content_once() {
    let mut case = Case::new();
    let text = string(&mut case);
    let condition = case.ty(TyKind::Bool);
    let main = case.script(&["app"], "main", "let a = $s; let b = $t; let r = if $c { &mut a } else { &mut b }; *r = $u; a");
    case.entry(main, &[("c", condition), ("s", text), ("t", text), ("u", text)], Some(text));
    let printed = case.admitted();
    assert_eq!(dropped_before_each_store(&printed), [true], "{printed}");
    assert_each_content_ends_once(&printed, &[Input::Text("s"), Input::Text("t"), Input::Word, Input::Text("u")]);
}

#[test]
fn a_store_through_a_join_one_of_whose_places_was_moved_out_is_refused() {
    let mut case = Case::new();
    let text = string(&mut case);
    let condition = case.ty(TyKind::Bool);
    let source = "let a = $s; let b = $t; let r1 = &mut a; let x = *r1; let r = if $c { r1 } else { &mut b }; *r = $u; x";
    let main = case.script(&["app"], "main", source);
    case.entry(main, &[("c", condition), ("s", text), ("t", text), ("u", text)], Some(text));
    let moved = source.find("*r1").expect("the move");
    let stored = source.find("*r = $u").expect("the store");
    let refused = case.refused();
    let expected = format!("app::main: StoreOverContentMaybeMoved moved @{moved}..{} @{stored}..", moved + 3);
    assert!(refused.starts_with(&expected) && refused.lines().count() == 1, "{refused}");
}

// -- The check itself ------------------------------------------------------------

/// The body a double drop printed before a store through a reference to a
/// binding ended its old content by liveness: `t` dropped where its
/// liveness ended and taken out again at the store.
#[test]
#[should_panic(expected = "which held none")]
fn a_content_taken_out_twice_fails_the_check() {
    let printed = "module app::main\nbody0 entry=b0\nb0():\n  Assign slot0, p0\n  Assign slot1, p1\n  v0: String = Take slot0\n  Assign slot2, v0\n  v1: String = Take slot2\n  Drop v1\n  v2: &mut String = Ref(mut) slot2\n  Assign slot3, v2\n  Drop v2\n  v3: &mut String = Take slot3\n  v4: &mut String = Take slot3\n  Drop v4\n  v5: String = Take slot1\n  v6: String = Take *v3\n  Drop v6\n  Assign *v3, v5\n  Drop v3\n  v7: String = Take slot2\n  Return v7\nb1():\n  Diverge\n";
    assert_each_content_ends_once(printed, &[Input::Text("t"), Input::Text("u")]);
}

/// A store through a reference that replaces a content nothing dropped.
#[test]
#[should_panic(expected = "which nothing ended")]
fn a_content_replaced_undropped_fails_the_check() {
    let printed = "module app::main\nbody0 entry=b0\nb0():\n  Assign slot0, p0\n  v0: &mut String = Ref(mut) slot0\n  Assign *v0, p1\n  v1: String = Take slot0\n  Return v1\n";
    assert_each_content_ends_once(printed, &[Input::Text("t"), Input::Text("u")]);
}
