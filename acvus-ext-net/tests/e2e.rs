//! `http::fetch_get` at its contract: a script's call reaches a socket, and
//! what the server wrote comes back as the call's String.

use acvus_ext_net::http_registry;
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

struct LocalServer {
    port: u16,
    request_line: tokio::task::JoinHandle<String>,
}

/// Serve one request on a local port with `status` and `body`.
async fn serve_once(status: &'static str, body: &'static str) -> LocalServer {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a local port");
    let port = listener.local_addr().expect("bound address").port();
    let served = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("one connection");
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = socket.read(&mut buf).await.expect("read the request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
        }
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write the response");
        socket.shutdown().await.expect("close the socket");
        String::from_utf8_lossy(&request)
            .lines()
            .next()
            .expect("a request line")
            .to_string()
    });
    LocalServer {
        port,
        request_line: served,
    }
}

#[tokio::test]
async fn fetch_get_returns_the_body_the_server_wrote() {
    let server = serve_once("200 OK", "hello from a socket").await;
    let i = Interner::new();
    let src = format!("fetch_get(\"http://127.0.0.1:{}/greeting\")", server.port);
    let ran = run_script_with_externs(
        &i,
        &src,
        FxHashMap::default(),
        vec![http_registry()],
        Ty::String,
    )
    .await;
    assert_eq!(unsafe { ran.value.as_str() }, "hello from a socket");
    assert_eq!(
        server.request_line.await.expect("the server task"),
        "GET /greeting HTTP/1.1"
    );
}
