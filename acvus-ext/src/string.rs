//! String operations under Rust `str`'s names and contracts. All pure. A
//! string is a value, not a container of characters (RFC-0028): there is no
//! `get` returning a reference into it, because a `String` is UTF-8 and a
//! scalar value is not a storage inside it; a character is read out by value
//! with `char_at`, as the `char` it is (RFC-0058).
//!
//! Every function that reads takes a `&str`. A function whose result is a
//! run of its argument's own bytes returns `&str` and the caller holds the
//! argument's loan for as long as the result; a function that builds new
//! bytes returns `String` (RFC-0062 rules 2 and 3). Two units coexist
//! here and each function states its own: `len`, `find`, `rfind`,
//! `substring`, `is_char_boundary`, `char_indices` and `match_indices` are
//! in bytes, `char_at`, `chars` and the `pad_*` width in Unicode scalar
//! values.
//!
//! Where Rust panics on a byte offset that is not a character boundary,
//! this module refuses the run instead of returning an `Option`. The offset
//! comes from `find`, `rfind` or a regex match, all of which report
//! boundaries, so a non-boundary offset is a program that computed one, not
//! a value a caller can sensibly branch on.
//!
//! Ordering is `cmp`, `lt`, `le`, `gt` and `ge`, bytewise as Rust's
//! `Ord for str`. The `<` operator does not reach text: an operator instance
//! for a `String` waits for RFC-0067's `ord<T>`.

use acvus_extern::Ctx;
use std::cmp::Ordering;

use acvus_extern::{Erased, Registry, Runtime, TyArg, Var, extern_fn, extern_registry, kind};

use crate::iter::Items;

fn padding(fill: &str, count: usize) -> String {
    fill.chars().cycle().take(count).collect()
}

fn shortfall(s: &str, width: i64) -> usize {
    let Ok(width) = usize::try_from(width) else {
        return 0;
    };
    width.saturating_sub(s.chars().count())
}

/// A count the address space cannot hold is a refusal, not a saturation:
/// the result would not be the one the program asked for either way.
fn count_of(what: &str, n: u64) -> usize {
    let Ok(n) = usize::try_from(n) else {
        panic!("{what}: count {n} exceeds the address space")
    };
    n
}

/// The length in bytes.
#[extern_fn(effect = pure)]
fn len(s: &str) -> u64 {
    s.len() as u64
}

#[extern_fn(effect = pure)]
fn is_empty(s: &str) -> bool {
    s.is_empty()
}

/// The law's proof: both groupings of three are the bytes of `a`, `b` and
/// `c` in turn, and `""` adds no byte on either side.
#[extern_fn(effect = pure, law(associative, identity = ""))]
fn concat(a: &str, b: &str) -> String {
    let mut s = String::with_capacity(a.len() + b.len());
    s.push_str(a);
    s.push_str(b);
    s
}

/// A view of `s` with the outer whitespace cut away. The bytes are `s`'s
/// own, so the caller holds `s`'s loan for as long as the result.
#[extern_fn(effect = pure)]
fn trim(s: &str) -> &str {
    s.trim()
}

/// A view of `s` with the outer whitespace cut away. The bytes are `s`'s
/// own, so the caller holds `s`'s loan for as long as the result.
#[extern_fn(effect = pure)]
fn trim_start(s: &str) -> &str {
    s.trim_start()
}

/// A view of `s` with the outer whitespace cut away. The bytes are `s`'s
/// own, so the caller holds `s`'s loan for as long as the result.
#[extern_fn(effect = pure)]
fn trim_end(s: &str) -> &str {
    s.trim_end()
}

/// A view of `s` with every leading and trailing `pat` cut away. An empty
/// `pat` cuts nothing.
#[extern_fn(effect = pure)]
fn trim_matches<'a>(s: &'a str, pat: &str) -> &'a str {
    match pat.is_empty() {
        true => s,
        false => s.trim_start_matches(pat).trim_end_matches(pat),
    }
}

/// A view of `s` with every leading `pat` cut away. An empty `pat` cuts
/// nothing.
#[extern_fn(effect = pure)]
fn trim_start_matches<'a>(s: &'a str, pat: &str) -> &'a str {
    match pat.is_empty() {
        true => s,
        false => s.trim_start_matches(pat),
    }
}

/// A view of `s` with every trailing `pat` cut away. An empty `pat` cuts
/// nothing.
#[extern_fn(effect = pure)]
fn trim_end_matches<'a>(s: &'a str, pat: &str) -> &'a str {
    match pat.is_empty() {
        true => s,
        false => s.trim_end_matches(pat),
    }
}

#[extern_fn(effect = pure)]
fn upper(s: &str) -> String {
    s.to_uppercase()
}

#[extern_fn(effect = pure)]
fn lower(s: &str) -> String {
    s.to_lowercase()
}

#[extern_fn(effect = pure)]
fn to_ascii_uppercase(s: &str) -> String {
    s.to_ascii_uppercase()
}

#[extern_fn(effect = pure)]
fn to_ascii_lowercase(s: &str) -> String {
    s.to_ascii_lowercase()
}

#[extern_fn(effect = pure)]
fn is_ascii(s: &str) -> bool {
    s.is_ascii()
}

/// `total`: `str::contains` with a `&str` pattern runs the two-way
/// substring search, which ends after a number of steps linear in the two
/// lengths, and does not panic.
#[extern_fn(effect = pure, total)]
fn contains(s: &str, pat: &str) -> bool {
    s.contains(pat)
}

#[extern_fn(effect = pure)]
fn starts_with(s: &str, pat: &str) -> bool {
    s.starts_with(pat)
}

#[extern_fn(effect = pure)]
fn ends_with(s: &str, pat: &str) -> bool {
    s.ends_with(pat)
}

#[extern_fn(effect = pure)]
fn replace(s: &str, from: &str, to: &str) -> String {
    s.replace(from, to)
}

/// `s` with the first `n` occurrences of `from` replaced; an `n` of 0
/// replaces nothing.
#[extern_fn(effect = pure)]
fn replacen(s: &str, from: &str, to: &str, n: u64) -> String {
    s.replacen(from, to, count_of("replacen", n))
}

#[extern_fn(effect = pure)]
fn repeat(s: &str, n: u64) -> String {
    s.repeat(count_of("repeat", n))
}

/// Whether byte `i` is the first byte of a character, the end of `s`
/// included.
#[extern_fn(effect = pure)]
fn is_char_boundary(s: &str, i: u64) -> bool {
    usize::try_from(i).is_ok_and(|i| s.is_char_boundary(i))
}

/// A view of the bytes `[start, end)` of `s`, which are `s`'s own, so the
/// caller holds `s`'s loan for as long as the result. Both offsets are byte
/// offsets and both must be on a character boundary; `start` past `end`, an
/// offset past the length, or an offset inside a character is refused
/// (RFC-0062 rule 2).
#[extern_fn(effect = pure)]
fn substring(s: &str, start: u64, end: u64) -> &str {
    let (Ok(from), Ok(to)) = (usize::try_from(start), usize::try_from(end)) else {
        refuse_substring("an offset exceeds the address space", s, start, end)
    };
    if from > to {
        refuse_substring("the range is inverted", s, start, end)
    }
    if !s.is_char_boundary(from) {
        refuse_substring("start is not on a character boundary", s, start, end)
    }
    if !s.is_char_boundary(to) {
        refuse_substring("end is not on a character boundary", s, start, end)
    }
    &s[from..to]
}

/// Obligation across artifacts: `asm_probe` asserts the operation holding
/// `substring` tail-calls its successor, which a refusal capturing its
/// message's parts by reference would break.
#[cold]
#[inline(never)]
fn refuse_substring(what: &str, s: &str, start: u64, end: u64) -> ! {
    panic!(
        "substring: {what} for the range {start}..{end} over {} bytes of {s:?}",
        s.len()
    )
}

#[extern_fn(effect = pure)]
fn to_bytes(s: String) -> Vec<u8> {
    s.into_bytes()
}

#[extern_fn(effect = pure)]
fn to_utf8(bytes: Vec<u8>) -> Option<String> {
    String::from_utf8(bytes).ok()
}

#[extern_fn(effect = pure)]
fn to_utf8_lossy(bytes: Vec<u8>) -> String {
    String::from_utf8_lossy(&bytes).into_owned()
}

/// `i` counts Unicode scalar values.
#[extern_fn(effect = pure)]
fn char_at(s: &str, i: i64) -> char {
    let out_of_range = || -> ! {
        panic!(
            "char_at: index {i} is out of range for length {}",
            s.chars().count()
        )
    };
    let Ok(index) = usize::try_from(i) else {
        out_of_range()
    };
    let Some(c) = s.chars().nth(index) else {
        out_of_range()
    };
    c
}

// -- Ordering -----------------------------------------------------------

/// Rust's `Ord for str`: the bytes compared lexicographically, `-1`, `0` or
/// `1` as `a` is before, equal to, or after `b`.
#[extern_fn(effect = pure, law(total_order))]
fn cmp(a: &str, b: &str) -> i64 {
    match a.cmp(b) {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

#[extern_fn(effect = pure)]
fn lt(a: &str, b: &str) -> bool {
    a < b
}

#[extern_fn(effect = pure)]
fn le(a: &str, b: &str) -> bool {
    a <= b
}

#[extern_fn(effect = pure)]
fn gt(a: &str, b: &str) -> bool {
    a > b
}

#[extern_fn(effect = pure)]
fn ge(a: &str, b: &str) -> bool {
    a >= b
}

// -- Producers ----------------------------------------------------------

/// One Unicode scalar value per step.
#[extern_fn(effect = pure)]
fn chars<I, Rt>(s: &str) -> Items<char, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.chars().collect())
}

/// A character and the byte offset it begins at.
#[derive(TyArg)]
pub struct CharIndex {
    index: u64,
    ch: char,
}

/// One Unicode scalar value per step, each with its own byte offset.
#[extern_fn(effect = pure)]
fn char_indices<I, Rt>(s: &str) -> Items<CharIndex, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(
        s.char_indices()
            .map(|(index, ch)| CharIndex {
                index: index as u64,
                ch,
            })
            .collect(),
    )
}

#[extern_fn(effect = pure)]
fn lines<I, Rt>(s: &str) -> Items<String, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.lines().map(str::to_owned).collect())
}

/// One byte per step.
#[extern_fn(effect = pure)]
fn bytes<I, Rt>(s: &str) -> Items<i64, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.bytes().map(i64::from).collect())
}

#[extern_fn(effect = pure)]
fn split_whitespace<I, Rt>(s: &str) -> Items<String, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.split_whitespace().map(str::to_owned).collect())
}

// -- Splitting ----------------------------------------------------------

#[extern_fn(effect = pure)]
fn split<I, Rt>(s: &str, pat: &str) -> Items<String, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.split(pat).map(str::to_owned).collect())
}

#[extern_fn(effect = pure)]
fn rsplit<I, Rt>(s: &str, pat: &str) -> Items<String, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.rsplit(pat).map(str::to_owned).collect())
}

/// At most `n` pieces: the last one holds the rest of `s`, separators and
/// all. An `n` of 0 gives no piece.
#[extern_fn(effect = pure)]
fn splitn<I, Rt>(s: &str, n: u64, pat: &str) -> Items<String, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(
        s.splitn(count_of("splitn", n), pat)
            .map(str::to_owned)
            .collect(),
    )
}

/// As `splitn`, from the right: the last piece holds the start of `s`.
#[extern_fn(effect = pure)]
fn rsplitn<I, Rt>(s: &str, n: u64, pat: &str) -> Items<String, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(
        s.rsplitn(count_of("rsplitn", n), pat)
            .map(str::to_owned)
            .collect(),
    )
}

/// As `split`, without the empty piece a trailing `pat` would give.
#[extern_fn(effect = pure)]
fn split_terminator<I, Rt>(s: &str, pat: &str) -> Items<String, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.split_terminator(pat).map(str::to_owned).collect())
}

/// Every non-overlapping `pat` in `s`, left to right.
#[extern_fn(effect = pure)]
fn matches<I, Rt>(s: &str, pat: &str) -> Items<String, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(s.matches(pat).map(str::to_owned).collect())
}

/// A match and the byte offset it begins at.
#[derive(TyArg)]
pub struct MatchIndex {
    index: u64,
    text: String,
}

/// Every non-overlapping `pat` in `s` with its byte offset, left to right.
#[extern_fn(effect = pure)]
fn match_indices<I, Rt>(s: &str, pat: &str) -> Items<MatchIndex, I, Rt>
where
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    Items::of(
        s.match_indices(pat)
            .map(|(index, text)| MatchIndex {
                index: index as u64,
                text: text.to_owned(),
            })
            .collect(),
    )
}

// -- Searching and shaping ---------------------------------------------

/// The byte offset of the first `pat` in `s`.
#[extern_fn(effect = pure)]
fn find(s: &str, pat: &str) -> Option<i64> {
    s.find(pat).map(|byte| byte as i64)
}

/// The byte offset of the last `pat` in `s`.
#[extern_fn(effect = pure)]
fn rfind(s: &str, pat: &str) -> Option<i64> {
    s.rfind(pat).map(|byte| byte as i64)
}

/// JS `padStart`: `fill` repeated and cut to the shortfall on the left; an
/// empty `fill` or a `width` at or below the length leaves `s` as it is.
/// `width` counts Unicode scalar values.
#[extern_fn(effect = pure)]
fn pad_start(s: &str, width: i64, fill: &str) -> String {
    let mut padded = padding(fill, shortfall(s, width));
    padded.push_str(s);
    padded
}

/// JS `padEnd`: as `pad_start`, on the right.
#[extern_fn(effect = pure)]
fn pad_end(s: &str, width: i64, fill: &str) -> String {
    let mut padded = s.to_owned();
    padded.push_str(&padding(fill, shortfall(s, width)));
    padded
}

#[extern_fn(effect = pure)]
fn strip_prefix(s: &str, pat: &str) -> Option<String> {
    s.strip_prefix(pat).map(str::to_owned)
}

#[extern_fn(effect = pure)]
fn strip_suffix(s: &str, pat: &str) -> Option<String> {
    s.strip_suffix(pat).map(str::to_owned)
}

/// The text before and after the first `pat`, as a two-element Vec: an
/// extern function returns no tuple (`acvus-extern` has no `Cross` for
/// one) and no array of a constant length (`Nth<kind::Length, K>` is a length variable).
#[extern_fn(effect = pure)]
fn split_once<Rt>(ctx: &mut Ctx<'_, Rt>, s: &str, pat: &str) -> Option<Vec<Erased<Rt, String>>>
where
    Rt: Runtime,
{
    let rt = ctx.rt;
    s.split_once(pat).map(|(head, tail)| {
        vec![
            Erased::new(rt, head.to_owned()),
            Erased::new(rt, tail.to_owned()),
        ]
    })
}

/// Rust's `str::eq_ignore_ascii_case`: only the ASCII letters fold.
#[extern_fn(effect = pure)]
fn eq_ignore_ascii_case(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// The full Unicode fold, which Rust's `str` has no method for: `a` and `b`
/// lowercased and compared.
#[extern_fn(effect = pure)]
fn eq_ignore_case(a: &str, b: &str) -> bool {
    a.to_lowercase() == b.to_lowercase()
}

#[extern_fn(effect = pure)]
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut out: String = first.to_uppercase().collect();
    out.push_str(chars.as_str());
    out
}

crate::iter::next_items_of!(element: String, next: next_items_string);
crate::iter::next_items_of!(element: char, next: next_items_char);
crate::iter::next_items_of!(element: i64, next: next_items_byte);
crate::iter::next_items_of!(element: CharIndex, next: next_items_char_index);
crate::iter::next_items_of!(element: MatchIndex, next: next_items_match_index);

pub fn string_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "string",
        fns: [
            len, is_empty, concat, trim, trim_start, trim_end,
            trim_matches, trim_start_matches, trim_end_matches,
            upper, lower, to_ascii_uppercase, to_ascii_lowercase, is_ascii,
            contains, starts_with, ends_with, replace, replacen, repeat,
            is_char_boundary, substring, to_bytes, to_utf8, to_utf8_lossy,
            cmp, lt, le, gt, ge,
            char_at, chars, char_indices, lines, bytes, split_whitespace,
            split, rsplit, splitn, rsplitn, split_terminator,
            matches, match_indices,
            find, rfind, pad_start, pad_end, strip_prefix, strip_suffix, split_once,
            eq_ignore_ascii_case, eq_ignore_case, capitalize,
            next_items_string, next_items_char, next_items_byte, next_items_char_index,
            next_items_match_index,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_extern::{Externs, Interner, TypesOnly};

    /// RFC-0082 rule 10 sampled: the sign of `cmp` is antisymmetric,
    /// transitive and total, and `0` exactly where the bytes are the same.
    #[test]
    fn total_order_holds_over_cmp() {
        let words: Vec<String> = ["", "a", "ab", "b", "é", "a\0", "\u{10FFFF}"]
            .into_iter()
            .map(str::to_string)
            .chain((0u64..48).map(|at| format!("{:x}", at.wrapping_mul(0x9e37_79b9) % 4096)))
            .collect();
        for a in &words {
            for b in &words {
                let ab = cmp(a, b);
                assert!([-1, 0, 1].contains(&ab), "the sign at {a:?}, {b:?}");
                assert_eq!(ab, -cmp(b, a), "antisymmetric at {a:?}, {b:?}");
                assert_eq!(ab == 0, a == b, "equal values are one value at {a:?}, {b:?}");
                for c in words.iter().take(16) {
                    if ab <= 0 && cmp(b, c) <= 0 {
                        assert!(cmp(a, c) <= 0, "transitive at {a:?}, {b:?}, {c:?}");
                    }
                }
            }
        }
    }

    /// RFC-0082 rule 5 sampled: `concat` is associative with `""` as its
    /// identity, and it does not commute, which it does not declare.
    #[test]
    fn concat_is_associative_with_the_empty_identity_and_does_not_commute() {
        let words: Vec<String> = ["", "a", "ab", "é", "\u{10FFFF}", "a\0"]
            .into_iter()
            .map(str::to_string)
            .chain((0u64..12).map(|at| format!("{:x}", at.wrapping_mul(0x9e37_79b9) % 4096)))
            .collect();
        for a in &words {
            assert_eq!(concat("", a), *a, "left identity at {a:?}");
            assert_eq!(concat(a, ""), *a, "right identity at {a:?}");
            for b in &words {
                for c in &words {
                    assert_eq!(
                        concat(&concat(a, b), c),
                        concat(a, &concat(b, c)),
                        "associative at {a:?}, {b:?}, {c:?}"
                    );
                }
            }
        }
        assert_ne!(concat("a", "b"), concat("b", "a"));
    }

    /// The law is `concat`'s own over the `str` a `String` lends, and it
    /// states no commutation.
    #[test]
    fn concat_declares_an_associative_law_with_the_empty_identity() {
        let i = Interner::new();
        let reg = Externs::combine(
            vec![
                crate::iterator_registry::<TypesOnly>(),
                string_registry::<TypesOnly>(),
            ],
            &i,
        )
        .expect("registry combines");
        let qref = acvus_extern::QualifiedRef::qualified(i.intern("string"), i.intern("concat"));
        let function = reg
            .functions
            .iter()
            .find(|f| f.qref == qref)
            .expect("concat is declared");
        let acvus_extern::FnKind::Extern { instances, .. } = &function.kind else {
            panic!("concat is an extern")
        };
        let generic = instances.generic.as_ref().map(|generic| &generic.laws);
        let concrete = instances.concrete.iter().map(|instance| &instance.laws);
        let declared: Vec<&acvus_extern::Laws> = concrete.chain(generic).collect();
        assert_eq!(
            declared,
            vec![&acvus_extern::Laws::Binary(acvus_extern::BinaryLaws {
                associative: true,
                commutative: false,
                identity: Some(acvus_extern::Identity::Const(acvus_extern::Literal::String(
                    String::new()
                ))),
            })]
        );
    }

    #[test]
    fn registry_produces_functions() {
        let i = Interner::new();
        let reg = Externs::combine(
            vec![
                crate::iterator_registry::<TypesOnly>(),
                string_registry::<TypesOnly>(),
            ],
            &i,
        )
        .expect("registry combines");
        let core = Externs::combine(vec![crate::iterator_registry::<TypesOnly>()], &i)
            .expect("the baseline combines");
        assert_eq!(reg.functions.len() - core.functions.len(), 53);
        assert_eq!(reg.handlers.len() - core.handlers.len(), 53);
    }
}
