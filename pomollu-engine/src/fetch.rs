use Future;

use acvus_orchestration::HttpRequest;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

// ---------------------------------------------------------------------------
// UnsafeSend -- WASM is single-threaded, safe to mark as Send
// ---------------------------------------------------------------------------

pub(crate) struct UnsafeSend<T>(pub T);
unsafe impl<T> Send for UnsafeSend<T> {}

impl<T> Future for UnsafeSend<T>
where
    T: Future,
{
    type Output = T::Output;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: WASM is single-threaded, no concurrent access.
        unsafe { self.map_unchecked_mut(|s| &mut s.0).poll(cx) }
    }
}

// ---------------------------------------------------------------------------
// WebFetch -- browser fetch API
// ---------------------------------------------------------------------------

pub(crate) struct WebFetch;

impl acvus_orchestration::Fetch for WebFetch {
    fn fetch(
        &self,
        request: &HttpRequest,
    ) -> impl Future<Output = Result<serde_json::Value, String>> + Send {
        let url = request.url.clone();
        let headers = request.headers.clone();
        let body = request.body.clone();

        UnsafeSend(async move {
            use web_sys::{Headers as WHeaders, Request, RequestInit, Response};

            let opts = RequestInit::new();
            opts.set_method("POST");

            let h = WHeaders::new().map_err(|e| format!("{e:?}"))?;
            h.set("Content-Type", "application/json")
                .map_err(|e| format!("{e:?}"))?;
            for (k, v) in &headers {
                h.set(k, v).map_err(|e| format!("{e:?}"))?;
            }
            opts.set_headers(&h);

            let body_str = serde_json::to_string(&body).map_err(|e| e.to_string())?;
            opts.set_body(&JsValue::from_str(&body_str));

            let req = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;

            let window = web_sys::window().ok_or("no window")?;
            let resp_val = JsFuture::from(window.fetch_with_request(&req))
                .await
                .map_err(|e| format!("{e:?}"))?;
            let resp: Response = resp_val.dyn_into().map_err(|e| format!("{e:?}"))?;

            let json_promise = resp.json().map_err(|e| format!("{e:?}"))?;
            let json_val = JsFuture::from(json_promise)
                .await
                .map_err(|e| format!("{e:?}"))?;

            let json_str = js_sys::JSON::stringify(&json_val)
                .map_err(|e| format!("{e:?}"))?
                .as_string()
                .ok_or_else(|| "JSON.stringify returned non-string".to_string())?;
            serde_json::from_str(&json_str).map_err(|e| e.to_string())
        })
    }
}
