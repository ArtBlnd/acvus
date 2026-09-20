# HTTP

`acvus-ext-net`'s `http_registry` puts `reqwest` in front of a script: plain
functions that need no client, a `Client` the script builds with its own
settings, a request it fills a piece at a time, and the `Response` all three
answer with.

```
match get_text("https://example.test/hello") { Ok(s) => s, Err(e) => "" }
```

A bare name is picked by its receiver, as the regex module's `find` is:
`get(url)` is the plain function and `c.get(url)` is the client's,
`header(&response, "etag")` and `request.header("accept", "text/plain")`
are two different functions of one name, and `send` is the idempotent one
or the opaque one by the request it is handed.

## Errors are values

No function here panics on a network or an HTTP condition. Every one that
can fail returns `Result<_, HttpError>`, and `HttpError` is a structural
enum a script matches on:

| variant | raised when |
| --- | --- |
| `Timeout` | the request outlived the timeout set on it |
| `Connect(String)` | the request reached no server: name resolution, connect, TLS, or a redirect chain that did not end |
| `Status(u16)` | `error_for_status` or `get_text` met a 4xx or 5xx |
| `Body(String)` | the body did not read as what was asked for — `text` on bytes that are not UTF-8 |
| `InvalidUrl(String)` | the text given as the URL, joined onto any base URL, is not one |
| `Method(String)` | the text given as the method is not an HTTP token |
| `Header(String)` | a header name or value the script wrote is not one |

A status outside 2xx is **not** an error by itself: `get` answers `Ok` with
the response, and `status`, `ok` and `error_for_status` are how a script
decides. `get_text` is the exception and raises `Status`, because it has no
response to hand back.

## The plain functions

Each sends on the host's own client, with no base URL and no default
headers. `get`, `head`, `put` and `delete` are the methods RFC 9110 calls
safe or idempotent, so they are declared `idempotent`: two such calls
commute and stand outside RFC-0007's order chain. Servers break that spec,
so each has a twin with the same body declared `opaque`, which the order
chain holds in place — a script that knows its server writes the twin.

| name | signature | `reqwest` twin | effect | difference |
| --- | --- | --- | --- | --- |
| `get` | `(url: String) -> Result<Response, HttpError>` | `Client::get(..).send()` | idempotent | — |
| `effectful_get` | same | same | opaque | same body, kept in the order chain |
| `head` | `(url: String) -> Result<Response, HttpError>` | `Client::head` | idempotent | — |
| `effectful_head` | same | same | opaque | — |
| `put` | `(url: String, body: String) -> Result<Response, HttpError>` | `Client::put(..).body(..)` | idempotent | — |
| `effectful_put` | same | same | opaque | — |
| `delete` | `(url: String) -> Result<Response, HttpError>` | `Client::delete` | idempotent | — |
| `effectful_delete` | same | same | opaque | — |
| `post` | `(url: String, body: String) -> Result<Response, HttpError>` | `Client::post(..).body(..)` | opaque | — |
| `patch` | `(url: String, body: String) -> Result<Response, HttpError>` | `Client::patch` | opaque | — |
| `request` | `(method: String, url: String, body: String, headers: Vec<Header>) -> Result<Response, HttpError>` | `Client::request` | opaque | the method is text, so the declaration cannot know it is safe |
| `get_text` | `(url: String) -> Result<String, HttpError>` | `get(..).error_for_status()?.text()` | idempotent | one call: send, raise a 4xx or 5xx, read the body |
| `effectful_get_text` | same | same | opaque | — |
| `request_for` | `(method: &str, url: &str) -> Request` | `Client::request` | pure | builds; sends nothing |
| `get_request` | `(url: &str) -> IdempotentRequest` | `Client::get` | pure | builds; `send` on it is idempotent |
| `head_request` | `(url: &str) -> IdempotentRequest` | `Client::head` | pure | — |
| `put_request` | `(url: &str) -> IdempotentRequest` | `Client::put` | pure | the body is set with `body` or `form` |
| `delete_request` | `(url: &str) -> IdempotentRequest` | `Client::delete` | pure | — |
| `client` | `() -> Client` | `Client::clone` | pure | the host's client, so no second connection pool |
| `client_with` | `(settings: ClientSettings) -> Result<Client, HttpError>` | `ClientBuilder::build` | pure | — |
| `client_settings` | `() -> ClientSettings` | `ClientBuilder::new` | pure | the settings `client()` already has |

A sending function takes its text as `String` and a pure one as `&str`.
That is not a style choice: an awaited declaration's parameters are each one
value, and a string slice is a register pair (RFC-0062), so `&str` is
refused on an `async` declaration.

## The client

`Client` is an extension type: a `reqwest::Client` with the settings it was
built from, plus the base URL those settings named. Every plain function
above except `client`, `client_with` and `client_settings` is also a method
on it, at the same signature with `&Client` in front and the same effect:

```
let c = client_settings();
c.base_url = Some("https://example.test/v1/");
match client_with(c) {
  Ok(api) => match api.get_text("users") { Ok(s) => s, Err(e) => "" },
  Err(e) => "",
}
```

`base_url` joins by RFC 3986 §5: an absolute URL replaces the base, a rooted
one keeps its origin, and a relative one is taken from the base's directory
— so a base that names a directory ends in `/`.

### The settings

Every field is written, as `RegexFlags` is: an object of four fields is not
a `ClientSettings` (RFC-0042). A script starts from `client_settings()` and
assigns the fields it wants.

| field | type | default | `reqwest` twin |
| --- | --- | --- | --- |
| `timeout_ms` | `Option<u64>` | `None` | `ClientBuilder::timeout` |
| `base_url` | `Option<String>` | `None` | — (`reqwest` has none) |
| `headers` | `Vec<Header>` | `vec([])` | `ClientBuilder::default_headers` |
| `follow_redirects` | `Bool` | `true` | `ClientBuilder::redirect`, limited to 10 |
| `user_agent` | `Option<String>` | `None` | `ClientBuilder::user_agent` |

Writing the settings as one object literal does not work today. An object
literal with a field whose value is `None` is accepted by the checker and
trips the interpreter at run time — `an operation takes a register its frame
does not own: a double take`, `acvus-interpreter/src/regs.rs:631` — with or
without an intervening `let`. `client_settings()` and field assignment is
the way through, and the defect is in the interpreter, not here.

## The request, and which `send` it gets

A request that has sent nothing is one of two types, and which one a
constructor answers with is what the method means to the order chain:

| constructor | type | `send` on it |
| --- | --- | --- |
| `get_request`, `head_request`, `put_request`, `delete_request` | `IdempotentRequest` | idempotent |
| `request_for(method, url)` | `Request` | opaque |

`request_for` takes its method as text the script chose at run time, so no
declaration can read whether RFC 9110 calls it safe; `Request` is the type
that says so. The four named constructors do know, and answer with the type
whose `send` stands outside the order chain.

`send` is one bare name resolved by its receiver (RFC-0043), so the effect a
call types is the request's own, with nothing for a script to remember:

| name | signature | `reqwest` twin | effect | difference |
| --- | --- | --- | --- | --- |
| `send` | `(IdempotentRequest) -> Result<Response, HttpError>` | `RequestBuilder::send` | idempotent | two such calls commute |
| `send` | `(Request) -> Result<Response, HttpError>` | same | opaque | held in place by the order chain |
| `effectful` | `(IdempotentRequest) -> Request` | — | pure | the downgrade for a server that mutates on a safe method |

Servers break the spec — a GET that mutates is common — and `effectful` is
how a script that knows its server says so: the request keeps everything set
on it and its `send` becomes the opaque one.

Each builder method takes the request and gives back its own type, so a
chain cannot lose the method's meaning halfway, and `send` consumes it.
`c.request_for(..)` and `c.get_request(..)` start from the client's settings
and base URL. Each setter is one shared signature (RFC-0019) with an
instance at each request type, as `num::abs` has one at each width; `R` below
is whichever type the chain started at.

| name | signature | `reqwest` twin | effect | difference |
| --- | --- | --- | --- | --- |
| `header` | `(R, name: &str, value: &str) -> R` | `RequestBuilder::header` | pure | a bad name or value is `Header`, raised at `send` |
| `query` | `(R, name: &str, value: &str) -> R` | `RequestBuilder::query` | pure | one pair, URL-encoded |
| `bearer` | `(R, token: &str) -> R` | `bearer_auth` | pure | — |
| `basic` | `(R, user: &str, password: &str) -> R` | `basic_auth` | pure | the password is always sent |
| `body` | `(R, text: &str) -> R` | `RequestBuilder::body` | pure | sets no content type |
| `form` | `(R, fields: Vec<Field>) -> R` | `RequestBuilder::form` | pure | `application/x-www-form-urlencoded` |
| `json` | `(R, text: &str) -> R` | `RequestBuilder::body` + a header | pure | the text is the body; `application/json` is set |
| `timeout_ms` | `(R, ms: u64) -> R` | `RequestBuilder::timeout` | pure | — |

`json` takes text, not a value. The language has no JSON value type for a
script to build or read, so there is nothing else for it to take.

## The response

`Response` is an extension type holding a status, a URL, the headers and the
body, all read before the value crosses. Streaming is not offered, so a
`Response` owns no connection and reading it cannot fail on the network.

| name | signature | `reqwest` twin | effect | difference |
| --- | --- | --- | --- | --- |
| `status` | `(&Response) -> u16` | `Response::status` | pure | one width with `HttpError::Status` |
| `ok` | `(&Response) -> Bool` | `StatusCode::is_success` | pure | 2xx |
| `url` | `(&Response) -> String` | `Response::url` | pure | after redirects |
| `header` | `(&Response, name: &str) -> Option<String>` | `Response::headers().get` | pure | matched case-insensitively |
| `headers` | `(&Response) -> Vec<Header>` | `Response::headers` | pure | in the order the server sent them |
| `text` | `(Response) -> Result<String, HttpError>` | `Response::text` | pure | **consumes**; bytes that are not UTF-8 are `Body` |
| `bytes` | `(Response) -> Vec<u8>` | `Response::bytes` | pure | **consumes**; no `Result`, the body is already read |
| `error_for_status` | `(Response) -> Result<Response, HttpError>` | `Response::error_for_status` | pure | 4xx and 5xx become `Status` |

`text` and `bytes` take the response by value, so a script that reads the
status must read it first:

```
match get(url) {
  Ok(r) => { let code = status(&r); let s = text(r); ... },
  Err(e) => ...,
}
```

Reading `status(&r)` after `text(r)` is refused at compile time.

A response whose header value is not UTF-8 is `Err(HttpError::Header(..))`:
nothing here silently rewrites a value the server sent.

## Header and Field

`Header` and `Field` are structural objects the script writes:
`{ name: "accept".to_string(), value: "text/plain".to_string(), }`. They are
two types rather than one `(String, String)` because no tuple crosses the
extern boundary today — `acvus-extern` implements `TyArg` for tuples and
`OneValue` for none, so a `Vec<(K, V)>` is not a type a declaration can
take. `Header` carries a header, `Field` a form field or a query parameter.

A list literal is an `Array<T, N>`, so a `Vec` parameter takes `vec([..])`.

## Not offered

Streaming bodies, multipart, proxies, cookie jars, HTTP/2 settings, TLS
options, and a connection pool as a value. Each is a decision, not an
omission: the first five need a value that stays open across calls or a
type the language has no reader for, and the last is what `client()` and
`client_with` hand out instead.

There is no `examples/` entry for this module. Every example is a script
run against a golden file, and an HTTP example needs a server the example
cannot start; the tests start one on `127.0.0.1:0` instead.
