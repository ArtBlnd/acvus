//! `http::fetch_get(url)`: one GET, the body as a String.

use acvus_extern::{Registry, Runtime, extern_fn, extern_registry};

#[extern_fn]
async fn fetch_get(#[state] client: &reqwest::Client, url: String) -> String {
    fn failed<T>(e: reqwest::Error) -> T {
        panic!("fetch_get: {e}")
    }
    client
        .get(&url)
        .send()
        .await
        .unwrap_or_else(failed)
        .error_for_status()
        .unwrap_or_else(failed)
        .text()
        .await
        .unwrap_or_else(failed)
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
