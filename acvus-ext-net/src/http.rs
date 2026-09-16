//! `http::fetch_get(url)`: one GET, the body as a String.

use acvus_extern::{Registry, Runtime, Trap, extern_fn, extern_registry};

#[extern_fn]
async fn fetch_get(#[state] client: &reqwest::Client, url: String) -> Result<String, Trap> {
    let failed = |e: reqwest::Error| Trap::call("fetch_get", e.to_string());
    client
        .get(&url)
        .send()
        .await
        .map_err(failed)?
        .error_for_status()
        .map_err(failed)?
        .text()
        .await
        .map_err(failed)
}

pub fn http_registry<R>() -> Registry<R>
where
    R: Runtime,
{
    extern_registry! {
        ns: "http",
        fns: [fetch_get(reqwest::Client::new())],
    }
}
