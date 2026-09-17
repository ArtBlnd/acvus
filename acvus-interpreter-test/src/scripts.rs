/// A dot-product attention over the context `query`, `keys`, `values`,
/// leaving the attended row in `out`; a consumer appends the trailing
/// expression that selects from `out`.
pub const ATTENTION: &str = "
let d = len(&@query);
let n = len(&@keys);
let scale = 1.0 / sqrt(to_float(d));

let scores = deque();
let t = 0;
while t < n {
    let s = 0.0;
    let i = 0;
    while i < d {
        s = s + *get(&@query, i) * *get(get(&@keys, t), i);
        i = i + 1;
    }
    push_back(&mut scores, s * scale);
    t = t + 1;
}

let m = if let Some(m) = as_iter(&scores) | map(|s| -> *s) | max { m } else { 0.0 };
let weights = as_iter(&scores) | map(|s| -> exp(*s - *m)) | collect;
let z = as_iter(&weights) | map(|w| -> *w) | sum;

let out = deque();
let j = 0;
while j < d {
    let acc = 0.0;
    let t = 0;
    while t < n {
        acc = acc + *get(&weights, t) / z * *get(get(&@values, t), j);
        t = t + 1;
    }
    push_back(&mut out, acc);
    j = j + 1;
}
";
