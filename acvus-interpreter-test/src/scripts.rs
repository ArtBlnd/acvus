/// A dot-product attention over the context `query`, `keys`, `values`,
/// leaving the attended row in `out`; a consumer appends the trailing
/// expression that selects from `out`.
pub const ATTENTION: &str = "
let d = @query.len();
let n = @keys.len();
let scale = 1.0 / d.to_float().sqrt();

let scores = deque();
let t = 0;
while t < n {
    let s = 0.0;
    let i = 0;
    while i < d {
        s = s + *@query.get(i) * *@keys.get(t).get(i);
        i = i + 1;
    }
    scores.push_back(s * scale);
    t = t + 1;
}

let m = if let Some(m) = scores.as_iter().map(|s| -> *s).max() { m } else { 0.0 };
let weights = scores.as_iter().map(|s| -> (*s - *m).exp()).collect();
let z = weights.as_iter().map(|w| -> *w).sum();

let out = deque();
let j = 0;
while j < d {
    let acc = 0.0;
    let t = 0;
    while t < n {
        acc = acc + *weights.get(t) / z * *@values.get(t).get(j);
        t = t + 1;
    }
    out.push_back(acc);
    j = j + 1;
}
";
