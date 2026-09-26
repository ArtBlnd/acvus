//! RFC-0080 rule 5: a cast is its crate's boundary module's. Outside
//! `acvus_extern::repr` and the interpreter's own `repr`, no source of the
//! workspace writes a `transmute`, a pointer cast, a slice or `Vec` rebuilt
//! from raw parts, a pointer made from an address, or a `union`.
//!
//! The scan reads each file's syntax (`syn`), so a comment is no match. A
//! macro's body — a `macro_rules!` arm, a `quote!` the proc macro emits — is
//! tokens and not syntax, so it is scanned token by token for the same forms.

use std::path::{Path, PathBuf};

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

/// The two boundary modules, relative to the workspace root.
const BOUNDARY: &[&str] = &["acvus-extern/src/repr.rs", "acvus-interpreter/src/repr.rs"];

/// A function whose call is a cast by itself: a reinterpretation of bytes,
/// a run rebuilt at a type and length its caller states, or a pointer made
/// from an address.
const CASTING_FUNCTIONS: &[&str] = &[
    "transmute",
    "transmute_copy",
    "from_raw_parts",
    "from_raw_parts_mut",
    "with_exposed_provenance",
    "with_exposed_provenance_mut",
];

/// The sites of one form a file keeps this round, counted: a site added to
/// the file changes the count and fails the scan as a new site anywhere
/// would.
struct Left {
    path: &'static str,
    form: Form,
    count: usize,
    reason: &'static str,
}

const STAND_IN_RUNTIME: &str = "a test's stand-in runtime names a reference's target as a \
    pointer to its own value type and back; the file is the parallel lane's this round, and \
    the same sites in acvus-extern/tests keep the type through `ptr::from_ref` and `cast_mut`";

const LEFT: &[Left] = &[
    Left {
        path: "acvus-utils/src/astr.rs",
        form: Form::AsPointer,
        count: 1,
        reason: "the interner extends a string's lifetime from its shard's read guard to the \
            interner; acvus-utils sits below the contract and holds no runtime's storage \
            (RFC-0080 rule 5)",
    },
    Left {
        path: "acvus-ext/tests/erased.rs",
        form: Form::AsPointer,
        count: 2,
        reason: STAND_IN_RUNTIME,
    },
    Left {
        path: "acvus-ext/tests/hash_instances.rs",
        form: Form::AsPointer,
        count: 2,
        reason: STAND_IN_RUNTIME,
    },
    Left {
        path: "acvus-ext/tests/owned_stage_holders.rs",
        form: Form::AsPointer,
        count: 2,
        reason: STAND_IN_RUNTIME,
    },
    Left {
        path: "acvus-ext/tests/regression_0041.rs",
        form: Form::AsPointer,
        count: 2,
        reason: STAND_IN_RUNTIME,
    },
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Form {
    /// `e as *const T`, `e as *mut T`.
    AsPointer,
    /// `p.cast()`, `p.cast::<T>()`, `NonNull::cast(p)`. A pointer's
    /// `cast_mut` and `cast_const` keep the pointee, and a witness's `cast`
    /// takes the value it casts, so neither is this form.
    Cast,
    /// A call, a path or an import of one of `CASTING_FUNCTIONS`.
    Function,
    /// A `union`, whose read at another field is a transmute.
    Union,
}

#[derive(PartialEq, Eq, Debug)]
struct Site {
    line: usize,
    form: Form,
}

#[derive(Default)]
struct Scan {
    sites: Vec<Site>,
}

impl Scan {
    fn found(&mut self, line: usize, form: Form) {
        self.sites.push(Site { line, form });
    }

    fn macro_tokens(&mut self, stream: TokenStream) {
        let trees: Vec<TokenTree> = stream.into_iter().collect();
        for (at, tree) in trees.iter().enumerate() {
            match tree {
                TokenTree::Group(group) => self.macro_tokens(group.stream()),
                TokenTree::Ident(ident) => {
                    let line = ident.span().start().line;
                    let after = &trees[at + 1..];
                    let before = at.checked_sub(1).map(|b| &trees[b]);
                    if CASTING_FUNCTIONS.iter().any(|f| ident == f) {
                        self.found(line, Form::Function);
                    } else if ident == "as" && is_pointer_type(after) {
                        self.found(line, Form::AsPointer);
                    } else if ident == "union" && is_union_body(after) {
                        self.found(line, Form::Union);
                    } else if ident == "cast"
                        && matches!(before, Some(TokenTree::Punct(p)) if p.as_char() == '.')
                        && is_call_of_no_argument(after)
                    {
                        self.found(line, Form::Cast);
                    }
                }
                TokenTree::Punct(_) | TokenTree::Literal(_) => {}
            }
        }
    }
}

fn is_punct(tree: Option<&TokenTree>, c: char) -> bool {
    matches!(tree, Some(TokenTree::Punct(p)) if p.as_char() == c)
}

/// `* const` or `* mut`.
fn is_pointer_type(after: &[TokenTree]) -> bool {
    is_punct(after.first(), '*')
        && matches!(after.get(1), Some(TokenTree::Ident(q)) if q == "const" || q == "mut")
}

/// A name, then a braced body.
fn is_union_body(after: &[TokenTree]) -> bool {
    matches!(after.first(), Some(TokenTree::Ident(_)))
        && matches!(after.get(1), Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace)
}

/// After a method's name: an optional `::<..>`, then an empty `()`.
fn is_call_of_no_argument(after: &[TokenTree]) -> bool {
    let mut rest = after.iter();
    let mut call = rest.next();
    if is_punct(call, ':') {
        if !is_punct(rest.next(), ':') || !is_punct(rest.next(), '<') {
            return false;
        }
        let mut open = 1usize;
        while open > 0 {
            match rest.next() {
                Some(TokenTree::Punct(p)) if p.as_char() == '<' => open += 1,
                Some(TokenTree::Punct(p)) if p.as_char() == '>' => open -= 1,
                Some(_) => {}
                None => return false,
            }
        }
        call = rest.next();
    }
    matches!(
        call,
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis && g.stream().is_empty()
    )
}

impl<'ast> Visit<'ast> for Scan {
    fn visit_expr_cast(&mut self, cast: &'ast syn::ExprCast) {
        if let syn::Type::Ptr(_) = &*cast.ty {
            self.found(cast.as_token.span.start().line, Form::AsPointer);
        }
        visit::visit_expr_cast(self, cast);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "cast" && call.args.is_empty() {
            self.found(call.method.span().start().line, Form::Cast);
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func
            && path
                .path
                .segments
                .last()
                .is_some_and(|last| last.ident == "cast")
            && call.args.len() == 1
        {
            self.found(path.span().start().line, Form::Cast);
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if let Some(last) = path.segments.last()
            && CASTING_FUNCTIONS.iter().any(|f| last.ident == f)
        {
            self.found(last.ident.span().start().line, Form::Function);
        }
        visit::visit_path(self, path);
    }

    fn visit_use_name(&mut self, name: &'ast syn::UseName) {
        if CASTING_FUNCTIONS.iter().any(|f| name.ident == f) {
            self.found(name.ident.span().start().line, Form::Function);
        }
    }

    fn visit_use_rename(&mut self, rename: &'ast syn::UseRename) {
        if CASTING_FUNCTIONS.iter().any(|f| rename.ident == f) {
            self.found(rename.ident.span().start().line, Form::Function);
        }
    }

    fn visit_item_union(&mut self, union: &'ast syn::ItemUnion) {
        self.found(union.union_token.span.start().line, Form::Union);
        visit::visit_item_union(self, union);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.macro_tokens(mac.tokens.clone());
        visit::visit_macro(self, mac);
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the test crate sits in the workspace")
        .to_path_buf()
}

fn name_of(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| panic!("{} has no UTF-8 name", path.display()))
}

fn rust_sources_outside_build_output(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
                .path()
        })
        .collect();
    entries.sort();
    for path in entries {
        let name = name_of(&path);
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rust_sources_outside_build_output(&path, out);
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

fn scan(source: &str, path: &str) -> Vec<Site> {
    let file = syn::parse_file(source).unwrap_or_else(|e| panic!("{path} does not parse: {e}"));
    let mut scan = Scan::default();
    scan.visit_file(&file);
    scan.sites
}

#[test]
fn no_cast_is_written_outside_the_boundary_modules() {
    let root = workspace_root();
    let mut files = Vec::new();
    rust_sources_outside_build_output(&root, &mut files);
    assert!(
        files
            .iter()
            .any(|f| f.ends_with("acvus-interpreter/src/regs.rs")),
        "the scan of {} reaches the interpreter's sources",
        root.display()
    );

    let mut refused = Vec::new();
    let mut left_found = vec![0usize; LEFT.len()];
    for file in &files {
        let rel = file
            .strip_prefix(&root)
            .expect("a source under the root")
            .to_str()
            .unwrap_or_else(|| panic!("{} is no UTF-8 path", file.display()));
        if BOUNDARY.contains(&rel) {
            continue;
        }
        let source = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("reading {rel}: {e}"));
        for site in scan(&source, rel) {
            match LEFT
                .iter()
                .position(|left| left.path == rel && left.form == site.form)
            {
                Some(entry) => left_found[entry] += 1,
                None => refused.push(format!("{rel}:{}: {:?}", site.line, site.form)),
            }
        }
    }
    for (left, found) in LEFT.iter().zip(left_found) {
        if found != left.count {
            refused.push(format!(
                "{}: {found} sites of {:?}, where {} are left because {}",
                left.path, left.form, left.count, left.reason
            ));
        }
    }
    assert!(
        refused.is_empty(),
        "a cast outside `acvus_extern::repr` and the interpreter's `repr` (RFC-0080 rule 5):\n{}",
        refused.join("\n")
    );
}

#[test]
fn the_scan_finds_each_form_and_not_a_cast_that_keeps_the_type() {
    let source = r#"
        use std::mem::transmute;
        fn f(p: *const u64, q: std::ptr::NonNull<[u8]>, r: &u64) {
            let _ = p as *const u8;
            let _ = r as *const u64;
            let _ = p as *mut u64;
            let _ = q.cast::<u8>();
            let _ = q.cast();
            let _ = std::ptr::NonNull::cast::<u8>(q);
            let _ = unsafe { std::mem::transmute_copy::<u64, f64>(&1) };
            let _ = unsafe { std::slice::from_raw_parts(p, 1) };
            let _ = std::ptr::with_exposed_provenance::<u8>(8);
            macro_rules! m { ($p:expr) => { $p.cast::<Vec<u8>>() as *const u8 }; }
            quote::quote! { core::mem::transmute(x) };
            union U { a: u64, b: f64 }
        }
        fn keeps(p: *const u64, w: SameLayout<A, B>, a: A) {
            let _ = p.cast_mut();
            let _ = p.cast_const();
            let _ = w.cast(a);
            let _ = std::ptr::from_ref(&1u64);
            let _ = std::ptr::NonNull::slice_from_raw_parts(std::ptr::NonNull::<u8>::dangling(), 0);
            let _ = 1u64 as usize;
            let _ = s.cast_unsigned();
            let union = 1;
            macro_rules! n { ($w:expr, $a:expr) => { $w.cast($a) }; }
            // p as *const u8, q.cast(), transmute
        }
    "#;
    let at = |line, form| Site { line, form };
    assert_eq!(
        scan(source, "inline"),
        [
            at(2, Form::Function),
            at(4, Form::AsPointer),
            at(5, Form::AsPointer),
            at(6, Form::AsPointer),
            at(7, Form::Cast),
            at(8, Form::Cast),
            at(9, Form::Cast),
            at(10, Form::Function),
            at(11, Form::Function),
            at(12, Form::Function),
            at(13, Form::Cast),
            at(13, Form::AsPointer),
            at(14, Form::Function),
            at(15, Form::Union),
        ]
    );
}
