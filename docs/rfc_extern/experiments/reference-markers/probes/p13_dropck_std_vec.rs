// expect: pass
// std's Vec drops with `may_dangle`, so a reference may dangle when it drops.
pub fn f() { let mut v: Vec<&i64> = Vec::new(); let x = 5i64; v.push(&x); }
