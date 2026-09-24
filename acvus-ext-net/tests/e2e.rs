//! `http` at its contract: what a script writes reaches a socket, and what
//! the server wrote comes back as the call's value.

mod loopback;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use acvus_ext_net::http_registry;
use acvus_extern::Externs;
use acvus_interpreter::{AcvusRuntime, Executor, SequentialExecutor};
use acvus_interpreter_test::{CompileResult, Refusal, check_source, execute_compiled};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    CompilationGraph, FnKind, Function, ParsedAst, QualifiedRef, extract, infer,
};
use acvus_mir::ty::{PolyBuilder, Reissue, Ty, TyTerm, lift_declaration};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

use loopback::Server;

/// Under `loopback::SLOW`, so `/slow` is still sleeping when the timeout
/// the script set expires.
const TIMEOUT_MS: u64 = 50;

fn registries() -> Vec<acvus_extern::Registry<AcvusRuntime>> {
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(http_registry());
    registries
}

fn parsed(i: &Interner, source: &str) -> ParsedAst {
    ParsedAst::Script(acvus_ast::parse_script(i, source).expect("parse"))
}

fn compiled(i: &Interner, source: &str, opt: Opt) -> Result<CompileResult, Refusal> {
    check_source(
        i,
        parsed(i, source),
        &FxHashMap::default(),
        registries(),
        Ty::String,
        opt,
        |_| {},
    )
}

async fn ran(source: &str, opt: Opt) -> String {
    let i = Interner::new();
    let cr = compiled(&i, source, opt)
        .unwrap_or_else(|r| panic!("refused:\n  {}", r.messages.join("\n  ")));
    let executor: Arc<dyn Executor> = Arc::new(SequentialExecutor);
    let (_, mut interp) = execute_compiled(&i, cr, HashMap::new(), executor);
    let value = interp.execute().await.expect("the seeds hold every context the run fetches");
    // SAFETY: the script's declared return type is `String`.
    unsafe { value.as_str() }.to_owned()
}

/// The script at both levels, which must agree: `Opt::Full` is what the
/// default harness runs, and a call an optimizer moved or dropped is the
/// failure the effect declarations exist to prevent.
async fn text(source: &str) -> String {
    let none = ran(source, Opt::None).await;
    let full = ran(source, Opt::Full).await;
    assert_eq!(none, full, "Opt::None and Opt::Full disagree");
    full
}

/// The reissue level `source` was inferred at.
///
/// The graph is built here rather than through `check_source` because that
/// harness declares its entry `Effect::OPAQUE`, which is the answer this
/// assertion asks for. Here the entry's effect is a fresh variable, so what
/// comes back is what the body's calls joined to.
fn reissue_of(source: &str) -> Reissue {
    let i = Interner::new();
    let Externs {
        functions: extern_fns,
        types,
        ..
    } = Externs::combine(registries(), &i).expect("registries combine");
    let entry = QualifiedRef::root(i.intern("main"));
    let mut pb = PolyBuilder::new();
    let mut functions = vec![Function {
        qref: entry,
        kind: FnKind::Local(parsed(&i, source), acvus_mir::graph::Inputs::FromReads),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(lift_declaration(&Ty::String, &mut pb)),
            captures: vec![],
            effect: pb.fresh_effect_var(),
            flows: acvus_mir::ty::Flows::Every.into(),
        },
    }];
    functions.extend(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
        types: Freeze::new(types),
        bindings: acvus_mir::graph::Bindings::default(),
        access: acvus_mir::graph::Access::Sync,
        entries: vec![entry],
    };
    let ext = extract::extract(&i, &graph);
    let inf = infer::infer(&i, &graph, &ext);
    assert!(!inf.has_errors(), "infer refused: {:?}", inf.errors());
    inf.outcomes
        .get(&entry)
        .expect("the entry was inferred")
        .meta()
        .ty
        .effect()
        .expect("a function type")
        .reissue
}

/// Every `HttpError` variant as a word the assertion can name, so one shape
/// of script covers every failure route.
fn classify(call: &str) -> String {
    format!(
        r#"match {call} {{
             Ok(v) => "ok".to_string(),
             Err(e) => match e {{
               HttpError::Timeout => "timeout".to_string(),
               HttpError::Connect(m) => "connect".to_string(),
               HttpError::Status(s) => "status ".to_string() + s.to_string(),
               HttpError::Body(m) => "body".to_string(),
               HttpError::InvalidUrl(m) => "invalid_url".to_string(),
               HttpError::Method(m) => "method".to_string(),
               HttpError::Header(m) => "header".to_string(),
             }},
           }}"#
    )
}

/// `match <call> { Ok(r) => text(r), Err => "err" }`, which most routes are
/// read through.
fn body_of(call: &str) -> String {
    format!(
        r#"match {call} {{
             Ok(r) => match text(r) {{ Ok(s) => s, Err(e) => "err".to_string() }},
             Err(e) => "err".to_string(),
           }}"#
    )
}

fn url(server: &Server, path: &str) -> String {
    format!("http://127.0.0.1:{}{path}", server.port)
}

// -- The plain functions ------------------------------------------------

#[tokio::test]
async fn get_text_returns_the_body_the_server_wrote() {
    let server = loopback::start().await;
    let source = format!(
        r#"match get_text("{}".to_string()) {{ Ok(s) => s, Err(e) => "err".to_string() }}"#,
        url(&server, "/text")
    );
    assert_eq!(text(&source).await, "hello from a socket");
}

#[tokio::test]
async fn a_response_reads_its_status_url_and_headers() {
    let server = loopback::start().await;
    let source = format!(
        r#"match get("{}".to_string()) {{
             Ok(r) => {{
               let code = status(&r);
               let good = ok(&r);
               let came_from = url(&r);
               let named = headers(&r);
               let count = len(&named);
               let kind = match header(&r, "content-type") {{
                 Some(v) => v,
                 None => "none".to_string(),
               }};
               code.to_string() + " " + good.to_string() + " " + came_from
                 + " " + count.to_string() + " " + kind
             }},
             Err(e) => "err".to_string(),
           }}"#,
        url(&server, "/text")
    );
    assert_eq!(
        text(&source).await,
        format!("200 true {} 3 text/plain", url(&server, "/text"))
    );
}

#[tokio::test]
async fn text_consumes_the_response_and_bytes_reads_the_same_body() {
    let server = loopback::start().await;
    let source = format!(
        r#"({}) + " " +
           match get("{}".to_string()) {{
             Ok(r) => {{ let raw = bytes(r); let count = len(&raw); count.to_string() }},
             Err(e) => "err".to_string(),
           }}"#,
        body_of(&format!(r#"get("{}".to_string())"#, url(&server, "/text"))),
        url(&server, "/text")
    );
    assert_eq!(text(&source).await, "hello from a socket 19");
}

#[tokio::test]
async fn head_answers_with_a_status_and_no_body() {
    let server = loopback::start().await;
    let source = format!(
        r#"match head("{}".to_string()) {{
             Ok(r) => {{
               let code = status(&r);
               let raw = bytes(r);
               let count = len(&raw);
               code.to_string() + " " + count.to_string()
             }},
             Err(e) => "err".to_string(),
           }}"#,
        url(&server, "/text")
    );
    assert_eq!(text(&source).await, "200 0");
}

#[tokio::test]
async fn post_sends_its_body_and_the_echo_route_returns_it() {
    let server = loopback::start().await;
    let source = body_of(&format!(
        r#"post("{}".to_string(), "payload".to_string())"#,
        url(&server, "/echo")
    ));
    assert_eq!(text(&source).await, "POST /echo|-|-|-|payload");
}

#[tokio::test]
async fn put_patch_and_delete_reach_the_method_the_server_saw() {
    let server = loopback::start().await;
    for (call, expected) in [
        (
            format!(
                r#"put("{}".to_string(), "p".to_string())"#,
                url(&server, "/echo")
            ),
            "PUT /echo|-|-|-|p",
        ),
        (
            format!(
                r#"patch("{}".to_string(), "q".to_string())"#,
                url(&server, "/echo")
            ),
            "PATCH /echo|-|-|-|q",
        ),
        (
            format!(r#"delete("{}".to_string())"#, url(&server, "/echo")),
            "DELETE /echo|-|-|-|",
        ),
    ] {
        assert_eq!(text(&body_of(&call)).await, expected);
    }
}

#[tokio::test]
async fn request_sends_the_method_and_headers_it_was_given() {
    let server = loopback::start().await;
    let source = body_of(&format!(
        r#"request("PUT".to_string(), "{}".to_string(), "b".to_string(),
                  vec([{{ name: "x-probe".to_string(), value: "seen".to_string(), }},]))"#,
        url(&server, "/echo")
    ));
    assert_eq!(text(&source).await, "PUT /echo|seen|-|-|b");
}

// -- The request value --------------------------------------------------

#[tokio::test]
async fn a_built_request_carries_every_piece_the_script_set() {
    let server = loopback::start().await;
    let source = format!(
        r#"let r = request_for("POST", "{}")
             .header("x-probe", "built")
             .query("a", "one two")
             .bearer("tok")
             .json("{{}}");
           {}"#,
        url(&server, "/echo"),
        body_of("send(r)")
    );
    assert_eq!(
        text(&source).await,
        "POST /echo?a=one+two|built|application/json|Bearer tok|{}"
    );
}

/// The same eight setters at the other request type: one signature, two
/// instances, and a chain that starts idempotent stays idempotent.
#[tokio::test]
async fn an_idempotent_request_carries_the_same_pieces() {
    let server = loopback::start().await;
    let source = format!(
        r#"let r = get_request("{}")
             .header("x-probe", "safe")
             .query("a", "one two")
             .bearer("tok");
           {}"#,
        url(&server, "/echo"),
        body_of("send(r)")
    );
    assert_eq!(
        text(&source).await,
        "GET /echo?a=one+two|safe|-|Bearer tok|"
    );
}

#[tokio::test]
async fn the_other_idempotent_constructors_reach_their_methods() {
    let server = loopback::start().await;
    for (call, expected) in [
        (
            format!(r#"put_request("{}").body("p")"#, url(&server, "/echo")),
            "PUT /echo|-|-|-|p",
        ),
        (
            format!(r#"delete_request("{}")"#, url(&server, "/echo")),
            "DELETE /echo|-|-|-|",
        ),
    ] {
        let source = format!("let r = {call};\n{}", body_of("send(r)"));
        assert_eq!(text(&source).await, expected);
    }

    let head = format!(
        r#"let r = head_request("{}");
           match send(r) {{
             Ok(resp) => {{ let code = status(&resp); code.to_string() }},
             Err(e) => "err".to_string(),
           }}"#,
        url(&server, "/echo")
    );
    assert_eq!(text(&head).await, "200");
}

#[tokio::test]
async fn a_form_body_is_url_encoded_and_a_basic_auth_is_base64() {
    let server = loopback::start().await;
    let source = format!(
        r#"let r = request_for("POST", "{}")
             .basic("user", "pass")
             .form(vec([{{ name: "a".to_string(), value: "one two".to_string(), }},
                        {{ name: "b".to_string(), value: "2".to_string(), }},]));
           {}"#,
        url(&server, "/echo"),
        body_of("send(r)")
    );
    assert_eq!(
        text(&source).await,
        "POST /echo|-|application/x-www-form-urlencoded|Basic dXNlcjpwYXNz|a=one+two&b=2"
    );
}

// -- The client value ---------------------------------------------------

#[tokio::test]
async fn a_client_joins_a_relative_url_onto_its_base() {
    let server = loopback::start().await;
    let source = format!(
        r#"let settings = client_settings();
           settings.base_url = Some("{}".to_string());
           match client_with(settings) {{
             Ok(c) => match c.get_text("echo".to_string()) {{
                        Ok(s) => s,
                        Err(e) => "err".to_string(),
                      }},
             Err(e) => "no client".to_string(),
           }}"#,
        url(&server, "/")
    );
    assert_eq!(text(&source).await, "GET /echo|-|-|-|");
}

#[tokio::test]
async fn a_clients_default_headers_and_user_agent_reach_the_server() {
    let server = loopback::start().await;
    let source = format!(
        r#"let settings = client_settings();
           settings.headers = vec([{{ name: "x-probe".to_string(),
                                     value: "from the client".to_string(), }},]);
           settings.user_agent = Some("acvus".to_string());
           match client_with(settings) {{
             Ok(c) => {},
             Err(e) => "no client".to_string(),
           }}"#,
        body_of(&format!(
            r#"c.post("{}".to_string(), "b".to_string())"#,
            url(&server, "/echo")
        ))
    );
    assert_eq!(text(&source).await, "POST /echo|from the client|-|-|b");
}

#[tokio::test]
async fn a_redirect_is_followed_by_default_and_not_when_the_client_says_so() {
    let server = loopback::start().await;
    let followed = format!(
        r#"match get_text("{}".to_string()) {{ Ok(s) => s, Err(e) => "err".to_string() }}"#,
        url(&server, "/moved")
    );
    assert_eq!(text(&followed).await, "hello from a socket");

    let kept = format!(
        r#"let settings = client_settings();
           settings.follow_redirects = false;
           match client_with(settings) {{
             Ok(c) => match c.get("{}".to_string()) {{
                        Ok(r) => {{ let code = status(&r); code.to_string() }},
                        Err(e) => "err".to_string(),
                      }},
             Err(e) => "no client".to_string(),
           }}"#,
        url(&server, "/moved")
    );
    assert_eq!(text(&kept).await, "302");
}

#[tokio::test]
async fn client_settings_is_what_client_builds_with() {
    let server = loopback::start().await;
    let source = format!(
        r#"let built = match client_with(client_settings()) {{
             Ok(c) => match c.get_text("{}".to_string()) {{
                        Ok(s) => s,
                        Err(e) => "err".to_string(),
                      }},
             Err(e) => "no client".to_string(),
           }};
           let host = client();
           let plain = match host.get_text("{}".to_string()) {{
             Ok(s) => s,
             Err(e) => "err".to_string(),
           }};
           built + " " + plain"#,
        url(&server, "/text"),
        url(&server, "/text")
    );
    assert_eq!(
        text(&source).await,
        "hello from a socket hello from a socket"
    );
}

// -- Every error variant from a route -----------------------------------

#[tokio::test]
async fn a_status_outside_2xx_is_a_status_error_only_where_the_script_asks() {
    let server = loopback::start().await;
    let kept = format!(
        r#"match get("{}".to_string()) {{
             Ok(r) => {{
               let code = status(&r);
               let good = ok(&r);
               code.to_string() + " " + good.to_string()
             }},
             Err(e) => "err".to_string(),
           }}"#,
        url(&server, "/missing")
    );
    assert_eq!(text(&kept).await, "404 false");

    for (path, expected) in [("/missing", "status 404"), ("/boom", "status 500")] {
        let raised = classify(&format!(
            r#"match get("{}".to_string()) {{
                 Ok(r) => error_for_status(r),
                 Err(e) => Err(e),
               }}"#,
            url(&server, path)
        ));
        assert_eq!(text(&raised).await, expected);
    }
}

#[tokio::test]
async fn get_text_raises_the_status_the_server_answered_with() {
    let server = loopback::start().await;
    let source = classify(&format!(
        r#"get_text("{}".to_string())"#,
        url(&server, "/missing")
    ));
    assert_eq!(text(&source).await, "status 404");
}

#[tokio::test]
async fn a_timeout_the_script_set_is_a_timeout_error() {
    assert!(Duration::from_millis(TIMEOUT_MS) < loopback::SLOW);
    let server = loopback::start().await;
    let source = format!(
        r#"let r = get_request("{}").timeout_ms({TIMEOUT_MS}u64);
           {}"#,
        url(&server, "/slow"),
        classify("send(r)")
    );
    assert_eq!(text(&source).await, "timeout");
}

#[tokio::test]
async fn a_body_that_is_not_utf8_is_a_body_error() {
    let server = loopback::start().await;
    let source = classify(&format!(
        r#"match get("{}".to_string()) {{
             Ok(r) => text(r),
             Err(e) => Err(e),
           }}"#,
        url(&server, "/binary")
    ));
    assert_eq!(text(&source).await, "body");
}

#[tokio::test]
async fn a_port_nothing_listens_on_is_a_connect_error() {
    let port = loopback::closed_port().await;
    let source = classify(&format!(
        r#"get("http://127.0.0.1:{port}/text".to_string())"#
    ));
    assert_eq!(text(&source).await, "connect");
}

#[tokio::test]
async fn text_that_is_not_a_url_is_an_invalid_url_error() {
    let source = classify(r#"get("not a url".to_string())"#);
    assert_eq!(text(&source).await, "invalid_url");
}

#[tokio::test]
async fn text_that_is_not_a_method_is_a_method_error() {
    let server = loopback::start().await;
    let source = classify(&format!(
        r#"request("BAD METHOD".to_string(), "{}".to_string(), "".to_string(), vec([]))"#,
        url(&server, "/echo")
    ));
    assert_eq!(text(&source).await, "method");
}

#[tokio::test]
async fn text_that_is_not_a_header_name_is_a_header_error() {
    let server = loopback::start().await;
    let source = format!(
        r#"let r = request_for("GET", "{}").header("bad name", "v");
           {}"#,
        url(&server, "/echo"),
        classify("send(r)")
    );
    assert_eq!(text(&source).await, "header");
}

// -- What the method means to the order chain ---------------------------

#[tokio::test]
async fn two_gets_type_as_idempotent_and_their_effectful_twins_as_opaque() {
    let server = loopback::start().await;
    let safe = format!(
        r#"let a = get_text("{}".to_string());
           let b = get_text("{}".to_string());
           "done".to_string()"#,
        url(&server, "/text"),
        url(&server, "/text")
    );
    assert_eq!(reissue_of(&safe), Reissue::Idempotent);

    let overruled = format!(
        r#"let a = effectful_get_text("{}".to_string());
           let b = effectful_get_text("{}".to_string());
           "done".to_string()"#,
        url(&server, "/text"),
        url(&server, "/text")
    );
    assert_eq!(reissue_of(&overruled), Reissue::Opaque);
}

/// The request's own type is what `send` types as, so a script cannot write
/// a POST that stands outside the order chain and cannot lose an ordering it
/// asked for.
#[tokio::test]
async fn send_types_as_the_request_it_was_given() {
    let server = loopback::start().await;
    let safe = format!(
        r#"let a = send(get_request("{}"));
           let b = send(get_request("{}"));
           "done".to_string()"#,
        url(&server, "/text"),
        url(&server, "/text")
    );
    assert_eq!(reissue_of(&safe), Reissue::Idempotent);

    let by_text = format!(
        r#"let a = send(request_for("POST", "{}"));
           "done".to_string()"#,
        url(&server, "/echo")
    );
    assert_eq!(reissue_of(&by_text), Reissue::Opaque);

    let overruled = format!(
        r#"let a = send(effectful(get_request("{}")));
           "done".to_string()"#,
        url(&server, "/text")
    );
    assert_eq!(reissue_of(&overruled), Reissue::Opaque);
}

#[tokio::test]
async fn a_post_types_as_opaque() {
    let server = loopback::start().await;
    let source = format!(
        r#"let a = post("{}".to_string(), "b".to_string());
           "done".to_string()"#,
        url(&server, "/echo")
    );
    assert_eq!(reissue_of(&source), Reissue::Opaque);
}

#[tokio::test]
async fn a_response_text_read_twice_is_refused() {
    let server = loopback::start().await;
    let source = format!(
        r#"match get("{}".to_string()) {{
             Ok(r) => match text(r) {{
                        Ok(s) => {{ let code = status(&r); s + code.to_string() }},
                        Err(e) => "err".to_string(),
                      }},
             Err(e) => "err".to_string(),
           }}"#,
        url(&server, "/text")
    );
    let i = Interner::new();
    let refusal = compiled(&i, &source, Opt::Full)
        .err()
        .expect("a response `text` consumed is not readable again");
    assert!(
        refusal
            .messages
            .iter()
            .any(|m| m.contains("moved") || m.contains("move")),
        "the refusal names the move: {:?}",
        refusal.messages
    );
}
