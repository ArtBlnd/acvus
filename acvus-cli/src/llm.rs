//! The LLM registries over a real HTTP client.

use std::sync::Arc;

use acvus_ext_llm::{Fetch, HttpRequest};
use acvus_extern::Registry;
use acvus_interpreter::AcvusRuntime;

struct Client(reqwest::Client);

impl Fetch for Client {
    async fn fetch(&self, request: &HttpRequest) -> Result<serde_json::Value, String> {
        let mut req = self.0.post(&request.url).json(&request.body);
        for (k, v) in &request.headers {
            req = req.header(k, v);
        }
        req.send()
            .await
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())
    }
}

pub fn registries() -> Vec<Registry<AcvusRuntime>> {
    let client = Arc::new(Client(reqwest::Client::new()));
    vec![
        acvus_ext_llm::openai_registry(Arc::clone(&client)),
        acvus_ext_llm::anthropic_registry(Arc::clone(&client)),
        acvus_ext_llm::google_registry(client),
    ]
}
