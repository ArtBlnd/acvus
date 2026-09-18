/// A dot-product attention over the context `query`, `keys`, `values`,
/// leaving the attended row in `out`; a consumer appends the trailing
/// expression that selects from `out`.
///
/// Each inner loop binds its row before the loop rather than writing
/// `@keys[t][i]` inline. That is a decision, not a habit: the inline form
/// takes a slice of the row on every iteration, and the bound form takes
/// one per row. `acvus-mir-test/tests/index.rs` holds both shapes and the
/// rule that separates them.
pub const ATTENTION: &str = "
let d = @query.len();
let n = @keys.len();
let scale = 1.0 / (d as f64).sqrt();

let scores = deque();
let t = 0;
while t < n {
    let key = &@keys[t];
    let s = 0.0;
    let i = 0;
    while i < d {
        s = s + @query[i] * key[i];
        i = i + 1;
    }
    scores.push_back(s * scale);
    t = t + 1;
}

let m = if let Some(m) = scores.as_iter().map(|s| -> *s).max() { m } else { 0.0 };
let weights = scores.as_iter().map(|s| -> (*s - m).exp()).collect();
let z = weights.as_iter().map(|w| -> *w).sum();

let out = deque();
let j = 0;
while j < d {
    let acc = 0.0;
    let t = 0;
    while t < n {
        let value = &@values[t];
        acc = acc + weights[t] / z * value[j];
        t = t + 1;
    }
    out.push_back(acc);
    j = j + 1;
}
";

/// `ATTENTION` with `Vec` in place of `Deque`, and with nothing else
/// changed, so that the attention bench's container axis measures the
/// container rather than the script.
///
/// The empty `Vec` is spelled `vec([])` because `std::vec` is the demotion
/// signature `fn vec<C, T>(items: C) -> Vec<T>` and takes one argument;
/// there is no nullary `vec()` the way there is a nullary `deque()`. An
/// `Arr<T, 0>` demotes to a `Vec<T>` of capacity zero, which is the same
/// empty start `deque()` gives.
pub const ATTENTION_VEC: &str = "
let d = @query.len();
let n = @keys.len();
let scale = 1.0 / (d as f64).sqrt();

let scores = vec([]);
let t = 0;
while t < n {
    let key = &@keys[t];
    let s = 0.0;
    let i = 0;
    while i < d {
        s = s + @query[i] * key[i];
        i = i + 1;
    }
    scores.push(s * scale);
    t = t + 1;
}

let m = if let Some(m) = scores.as_iter().map(|s| -> *s).max() { m } else { 0.0 };
let weights = scores.as_iter().map(|s| -> (*s - m).exp()).collect();
let z = weights.as_iter().map(|w| -> *w).sum();

let out = vec([]);
let j = 0;
while j < d {
    let acc = 0.0;
    let t = 0;
    while t < n {
        let value = &@values[t];
        acc = acc + weights[t] / z * value[j];
        t = t + 1;
    }
    out.push(acc);
    j = j + 1;
}
";
