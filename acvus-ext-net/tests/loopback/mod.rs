//! A server on loopback for the `http` tests to call, speaking HTTP/1.1 on
//! a socket rather than through a mock: what is under test is the path from
//! a script's call to a socket and back.

use std::time::Duration;

use acvus_ext_net::Header;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

/// How long `/slow` takes, which a test's `timeout_ms` must sit under.
pub const SLOW: Duration = Duration::from_millis(400);

/// What the echo route writes where the request carried no such header.
const ABSENT: &str = "-";

/// RFC 9112 §6: a request with no `Content-Length` and no transfer coding
/// has no body.
const NO_BODY: usize = 0;

pub struct Server {
    pub port: u16,
    task: JoinHandle<()>,
}

/// The task is aborted rather than joined: it is an accept loop with no end
/// of its own, and it is this process's, started by this test.
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub struct Incoming {
    pub method: String,
    pub target: String,
    pub headers: Vec<Header>,
    pub body: String,
}

impl Incoming {
    fn header(&self, name: &str) -> &str {
        self.headers
            .iter()
            .find(|h| h.name.eq_ignore_ascii_case(name))
            .map_or(ABSENT, |h| h.value.as_str())
    }
}

pub struct Reply {
    status: &'static str,
    location: Option<&'static str>,
    body: Vec<u8>,
}

impl Reply {
    fn text(status: &'static str, body: &str) -> Reply {
        Reply {
            status,
            location: None,
            body: body.as_bytes().to_vec(),
        }
    }
}

pub async fn start() -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("bound address").port();
    let task = tokio::spawn(async move {
        loop {
            let Ok((socket, _)) = listener.accept().await else {
                return;
            };
            drop(tokio::spawn(serve(socket)));
        }
    });
    Server { port, task }
}

/// A port nothing listens on: bound to learn a free one, then released.
pub async fn closed_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("bound address").port();
    drop(listener);
    port
}

async fn serve(mut socket: TcpStream) {
    let Some(incoming) = read_request(&mut socket).await else {
        return;
    };
    let reply = route(&incoming).await;
    let head_only = incoming.method == "HEAD";
    let mut out = format!(
        "HTTP/1.1 {}\r\nContent-Length: {}\r\nContent-Type: text/plain\r\nConnection: close\r\n",
        reply.status,
        reply.body.len()
    );
    if let Some(location) = reply.location {
        out.push_str(&format!("Location: {location}\r\n"));
    }
    out.push_str("\r\n");
    let mut bytes = out.into_bytes();
    if !head_only {
        bytes.extend_from_slice(&reply.body);
    }
    socket.write_all(&bytes).await.expect("write the response");
    socket.shutdown().await.expect("close the socket");
}

async fn route(incoming: &Incoming) -> Reply {
    let path = incoming.target.split('?').next().expect("a path");
    match path {
        "/text" => Reply::text("200 OK", "hello from a socket"),
        "/missing" => Reply::text("404 Not Found", "missing"),
        "/boom" => Reply::text("500 Internal Server Error", "boom"),
        "/moved" => Reply {
            status: "302 Found",
            location: Some("/text"),
            body: Vec::new(),
        },
        "/binary" => Reply {
            status: "200 OK",
            location: None,
            body: vec![0xFF, 0xFE],
        },
        "/slow" => {
            tokio::time::sleep(SLOW).await;
            Reply::text("200 OK", "late")
        }
        _ => Reply::text(
            "200 OK",
            &format!(
                "{} {}|{}|{}|{}|{}",
                incoming.method,
                incoming.target,
                incoming.header("x-probe"),
                incoming.header("content-type"),
                incoming.header("authorization"),
                incoming.body,
            ),
        ),
    }
}

async fn read_request(socket: &mut TcpStream) -> Option<Incoming> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    let head_end = loop {
        if let Some(at) = window(&raw, b"\r\n\r\n") {
            break at + 4;
        }
        let read = socket.read(&mut buf).await.expect("read the request");
        if read == 0 {
            return None;
        }
        raw.extend_from_slice(&buf[..read]);
    };
    let head = std::str::from_utf8(&raw[..head_end]).expect("a request head this test wrote");
    let mut lines = head.lines();
    let mut request_line = lines.next().expect("a request line").split(' ');
    let method = request_line.next().expect("a method").to_owned();
    let target = request_line.next().expect("a target").to_owned();
    let headers: Vec<Header> = lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.split_once(": ")
                .expect("a header line this test wrote")
        })
        .map(|(name, value)| Header {
            name: name.to_owned(),
            value: value.to_owned(),
        })
        .collect();
    let length: usize = headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case("content-length"))
        .map_or(Ok(NO_BODY), |h| h.value.parse())
        .expect("a Content-Length this test wrote");
    let mut body = raw.split_off(head_end);
    while body.len() < length {
        let read = socket.read(&mut buf).await.expect("read the body");
        if read == 0 {
            break;
        }
        body.extend_from_slice(&buf[..read]);
    }
    Some(Incoming {
        method,
        target,
        headers,
        body: String::from_utf8(body).expect("a body this test wrote"),
    })
}

fn window(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|slice| slice == needle)
}
