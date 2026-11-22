//! A minimal HTTP/1.1 server on `std::net`: request line, headers,
//! `Content-Length` bodies, fixed or chunked responses, one connection per
//! thread, `Connection: close` after every response.
//!
//! Owning the socket is the point: a client that goes away is noticed the
//! moment a write fails or, from the moment its request is queued, when its
//! connection reads a reset or an EOF that a probe confirms ([`Watched`]),
//! which is what lets the engine stop a prefill or a decode for a departed
//! client. General-purpose crates hide that behind buffering and background
//! writers.

use std::io::{BufRead as _, BufReader, ErrorKind, Read as _, Write as _};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail, ensure};

/// The request line and headers together; a line is read through a
/// [`Read::take`] of what is left of it, so one without a newline cannot
/// grow past it.
const MAX_HEADER_BYTES: usize = 64 << 10;
/// The request line and headers must arrive within this, however the bytes
/// trickle in.
const HEADER_TIMEOUT: Duration = Duration::from_secs(30);
/// The longest wait for the next bytes of a body.
const BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
/// The whole request (head and body) must arrive within this.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// A response written in one piece ([`respond_raw`], [`respond_unread`])
/// must be taken by the client within this: the largest are the
/// dashboard's script (about 130 KB) and its history, to a loopback peer.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
/// The longest a write of a relayed response ([`Watched`]) waits for the
/// client to take more. Such a response has no total limit, a generation
/// runs as long as it runs, but a client that has taken nothing for this
/// long while bytes wait for it is treated as gone, like one whose body
/// stopped arriving ([`BODY_IDLE_TIMEOUT`]). The engine's pauses do not
/// count: the timeout runs only while a write is blocked.
const WRITE_STALL_TIMEOUT: Duration = Duration::from_secs(120);

pub struct Request {
    pub method: String,
    /// Whether the request line said HTTP/1.1 (rather than 1.0).
    pub http11: bool,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// How long [`read_head_within`] and [`Body::read_with`] may take.
#[derive(Debug, Clone, Copy)]
pub struct Deadlines {
    /// For the request line and the headers.
    pub head: Duration,
    /// For the head and the body together.
    pub total: Duration,
    /// For the next bytes of the body.
    pub body_idle: Duration,
}

impl Default for Deadlines {
    fn default() -> Self {
        Self {
            head: HEADER_TIMEOUT,
            total: REQUEST_TIMEOUT,
            body_idle: BODY_IDLE_TIMEOUT,
        }
    }
}

/// The socket read with a deadline: each read waits at most until `until`
/// (and at most `idle`), and one past it fails with `TimedOut`, so a client
/// that sends a byte now and then cannot hold the connection open.
struct Deadlined {
    stream: TcpStream,
    until: Instant,
    idle: Duration,
}

impl std::io::Read for Deadlined {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(ErrorKind::TimedOut, "request timed out"));
        }
        self.stream.set_read_timeout(Some(left.min(self.idle)))?;
        self.stream.read(buf)
    }
}

/// A request's body, not read yet: [`read_head_within`] returns it so the
/// caller can refuse the request before it reads (or buffers) a byte of it.
pub struct Body {
    reader: BufReader<Deadlined>,
    reply_continue: bool,
    length: usize,
    total_deadline: Instant,
    body_idle: Duration,
}

/// The body's bytes read off the socket at a time ([`Body::read_with`]).
const BODY_CHUNK: usize = 64 << 10;

/// [`Body::read_with`] stopped because its `admit` refused the next bytes.
#[derive(Debug)]
pub struct OverBudget;

impl std::fmt::Display for OverBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no room for the request body")
    }
}

impl std::error::Error for OverBudget {}

/// [`read_head_within`] refused a `Content-Length` past its `max_body`.
#[derive(Debug)]
pub struct BodyTooLarge {
    pub limit: usize,
}

impl std::fmt::Display for BodyTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "request body exceeds {} bytes", self.limit)
    }
}

impl std::error::Error for BodyTooLarge {}

/// Whether `error` is a read that reached its deadline ([`Deadlines`]): a
/// socket read that timed out reports `WouldBlock` or `TimedOut`.
pub fn timed_out(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause.downcast_ref::<std::io::Error>().is_some_and(|io| {
            matches!(io.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock)
        })
    })
}

impl Body {
    /// The body's length (`Content-Length`, at most the `max_body` given).
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Reads the body into `request`, within the request's deadline.
    pub fn read(self, request: &mut Request) -> Result<()> {
        self.read_with(request, &mut |_| true)
    }

    /// [`Self::read`], asking `admit` for every run of bytes as it arrives
    /// (at most [`BODY_CHUNK`] at a time) before keeping it: a refusal ends
    /// the read with [`OverBudget`]. The buffer grows with what has
    /// arrived, never past the declared length, so a length that is
    /// declared and not sent takes neither memory nor budget.
    pub fn read_with(
        mut self,
        request: &mut Request,
        admit: &mut dyn FnMut(usize) -> bool,
    ) -> Result<()> {
        if self.reply_continue && self.length > 0 {
            let stream = self.reader.get_mut();
            stream
                .stream
                .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                .context("writing 100 Continue")?;
        }
        let reader = self.reader.get_mut();
        reader.until = self.total_deadline;
        reader.idle = self.body_idle;
        let mut body = Vec::new();
        let mut chunk = vec![0u8; BODY_CHUNK.min(self.length)];
        while body.len() < self.length {
            let want = chunk.len().min(self.length - body.len());
            let got = match self.reader.read(&mut chunk[..want]) {
                Ok(0) => bail!("reading request body: connection closed"),
                Ok(got) => got,
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                Err(error) => return Err(error).context("reading request body"),
            };
            if !admit(got) {
                return Err(OverBudget.into());
            }
            if body.capacity() - body.len() < got {
                // Doubling, but never past what the request declared.
                let target =
                    (body.capacity() * 2).max(body.len() + got).min(self.length);
                body.reserve_exact(target - body.len());
            }
            body.extend_from_slice(&chunk[..got]);
        }
        request.body = body;
        Ok(())
    }
}

/// Reads one request from `stream`. `max_body` bounds `Content-Length`.
pub fn read_request(stream: &mut TcpStream, max_body: usize) -> Result<Request> {
    let (mut request, body) = read_head_within(stream, max_body, Deadlines::default())?;
    body.read(&mut request)?;
    Ok(request)
}

/// Reads the request line and headers from `stream` (the request's body
/// left empty) within `deadlines.head`, at most [`MAX_HEADER_BYTES`] of
/// them; the [`Body`] reads the rest. `max_body` bounds `Content-Length`.
pub fn read_head_within(
    stream: &TcpStream,
    max_body: usize,
    deadlines: Deadlines,
) -> Result<(Request, Body)> {
    let started = Instant::now();
    let total_deadline = started + deadlines.total;
    let mut reader = BufReader::new(Deadlined {
        stream: stream.try_clone().context("cloning socket")?,
        until: (started + deadlines.head).min(total_deadline),
        idle: deadlines.head,
    });
    let mut total = 0usize;
    // One line through a `take` of what the head has left, plus one byte
    // so a line that reaches the limit is told from one that ends there.
    let mut read_line = |line: &mut String, what: &'static str| -> Result<()> {
        line.clear();
        let left = (MAX_HEADER_BYTES - total.min(MAX_HEADER_BYTES)) as u64 + 1;
        (&mut reader).take(left).read_line(line).context(what)?;
        total += line.len();
        ensure!(total <= MAX_HEADER_BYTES, "request headers too large");
        Ok(())
    };
    let mut line = String::new();
    read_line(&mut line, "reading request line")?;
    ensure!(!line.is_empty(), "connection closed before a request");
    let mut parts = line.split_whitespace();
    let method = parts.next().context("missing method")?.to_string();
    let target = parts.next().context("missing request target")?.to_string();
    let version = parts.next().context("missing HTTP version")?;
    ensure!(version.starts_with("HTTP/1."), "unsupported protocol {version}");
    let http11 = version != "HTTP/1.0";
    let mut headers = Vec::new();
    loop {
        read_line(&mut line, "reading headers")?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        let (name, value) = trimmed.split_once(':').context("malformed header line")?;
        headers.push((name.trim().to_string(), value.trim().to_string()));
    }
    let request = Request { method, http11, path: target, headers, body: Vec::new() };
    if request
        .header("Transfer-Encoding")
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
    {
        bail!("chunked request bodies are not supported; send Content-Length");
    }
    let length: usize = match request.header("Content-Length") {
        Some(v) => v.trim().parse().context("invalid Content-Length")?,
        None => 0,
    };
    if length > max_body {
        return Err(BodyTooLarge { limit: max_body }.into());
    }
    let reply_continue = request
        .header("Expect")
        .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"));
    let body = Body {
        reader,
        reply_continue,
        length,
        total_deadline,
        body_idle: deadlines.body_idle,
    };
    Ok((request, body))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "",
    }
}

fn head(status: u16, content_type: &str, extra: &[(&str, &str)]) -> String {
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nConnection: close\r\n",
        reason(status)
    );
    for (k, v) in extra {
        head.push_str(k);
        head.push_str(": ");
        head.push_str(v);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    head
}

/// Writes a complete response and closes the connection.
pub fn respond(
    stream: TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let length = body.len().to_string();
    respond_raw(stream, status, content_type, &[("Content-Length", &length)], body)
}

/// Writes `bytes` to `stream`, all of them by `deadline`: a client that
/// does not read, or reads a little now and then, fails the write with
/// `TimedOut` instead of holding the connection's thread. A write timeout
/// already set on the socket still bounds each single write.
fn write_all_by(
    mut stream: &TcpStream,
    mut bytes: &[u8],
    deadline: Instant,
) -> std::io::Result<()> {
    // A socket whose options cannot be read or set any more (macOS refuses
    // them once the connection is gone) fails the write itself.
    let own = stream.write_timeout().ok().flatten();
    while !bytes.is_empty() {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(ErrorKind::TimedOut, "response timed out"));
        }
        stream.set_write_timeout(Some(own.map_or(left, |own| own.min(left)))).ok();
        match stream.write(bytes) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(wrote) => bytes = &bytes[wrote..],
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Writes `headers` as given (`Content-Length` among them: a HEAD answer
/// states its GET's length over an empty `body`) and closes the connection.
/// The client has [`RESPONSE_TIMEOUT`] to take the response.
pub fn respond_raw(
    stream: TcpStream,
    status: u16,
    content_type: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> Result<()> {
    respond_raw_within(stream, status, content_type, headers, body, RESPONSE_TIMEOUT)
}

/// [`respond_raw`] with `limit` as its [`RESPONSE_TIMEOUT`]. The socket is
/// closed whether or not the response was written whole.
fn respond_raw_within(
    stream: TcpStream,
    status: u16,
    content_type: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    limit: Duration,
) -> Result<()> {
    let written = write_response(&stream, status, content_type, headers, body, limit);
    stream.shutdown(Shutdown::Both).ok();
    written
}

/// The head and `body` of a response, written within `limit`.
fn write_response(
    stream: &TcpStream,
    status: u16,
    content_type: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    limit: Duration,
) -> Result<()> {
    let deadline = Instant::now() + limit;
    write_all_by(stream, head(status, content_type, headers).as_bytes(), deadline)
        .context("writing response head")?;
    write_all_by(stream, body, deadline).context("writing response body")
}

/// The most a refusal reads of a request it did not read whole
/// ([`respond_unread`]), and the longest it waits for it.
const UNREAD_DRAIN_BYTES: usize = 1 << 20;
const UNREAD_DRAIN_TIME: Duration = Duration::from_secs(1);

/// [`respond_raw`] for a request whose body was not read (or not read
/// whole): closing a socket that holds unread bytes resets the connection,
/// and the reset can discard the response before the client has read it.
/// So the write side is closed first and what the client still sends is
/// read and dropped, up to [`UNREAD_DRAIN_BYTES`] or [`UNREAD_DRAIN_TIME`],
/// before the socket closes.
pub fn respond_unread(
    mut stream: TcpStream,
    status: u16,
    content_type: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> Result<()> {
    if let Err(error) =
        write_response(&stream, status, content_type, headers, body, RESPONSE_TIMEOUT)
    {
        stream.shutdown(Shutdown::Both).ok();
        return Err(error);
    }
    stream.shutdown(Shutdown::Write).ok();
    let deadline = Instant::now() + UNREAD_DRAIN_TIME;
    let mut left = UNREAD_DRAIN_BYTES;
    let mut sink = [0u8; 16 << 10];
    while left > 0 {
        let wait = deadline.saturating_duration_since(Instant::now());
        if wait.is_zero() || stream.set_read_timeout(Some(wait)).is_err() {
            break;
        }
        match stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(got) => left = left.saturating_sub(got),
        }
    }
    stream.shutdown(Shutdown::Both).ok();
    Ok(())
}

/// How long a probe waits for the peer's reset before it takes the
/// connection for a half-closed client that still reads (loopback answers
/// within microseconds, a LAN within milliseconds).
const PROBE_WINDOW: Duration = Duration::from_millis(250);
const PROBE_STEP: Duration = Duration::from_millis(5);

/// What has been written to a watched connection so far, which decides how
/// the watcher may probe it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Nothing yet: the request is queued or the engine is working on it.
    Waiting,
    /// The head of an event stream: further writes are chunks of events.
    Events,
    /// The head of any other chunked body, whose bytes follow at once.
    Body,
    /// The response is complete or abandoned; the server closed the socket.
    Done,
}

struct Conn {
    stream: TcpStream,
    /// Held around every write, so a probe never lands inside a response
    /// line or a chunk.
    phase: Mutex<Phase>,
    cancelled: Arc<AtomicBool>,
    /// Whether an interim `1xx` response may be sent (HTTP/1.1 only).
    interim: bool,
}

impl Conn {
    fn phase(&self) -> MutexGuard<'_, Phase> {
        self.phase.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The peer's read side reached EOF: either it closed the connection, or
    /// it only shut down its sending side and still waits for the response
    /// (an HTTP/1.1 client may half-close after its request). Only a write
    /// tells the two apart: a closed peer answers it with a reset. The probe
    /// is something the client must accept anyway: an interim `102
    /// Processing` before the head (RFC 9110 §15.2: a client parses 1xx
    /// responses it did not expect), an SSE comment chunk in an event
    /// stream. Inside any other body nothing can be written, and the
    /// response's own writes decide.
    fn departed(&self) -> bool {
        {
            let phase = self.phase();
            let probe: &[u8] = match *phase {
                Phase::Waiting if self.interim => b"HTTP/1.1 102 Processing\r\n\r\n",
                Phase::Events => b"3\r\n:\n\n\r\n",
                Phase::Waiting | Phase::Body | Phase::Done => return false,
            };
            if (&self.stream).write_all(probe).is_err() {
                return true;
            }
        }
        let deadline = Instant::now() + PROBE_WINDOW;
        loop {
            match self.stream.take_error() {
                Ok(None) => {}
                Ok(Some(_)) | Err(_) => return true,
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(PROBE_STEP);
        }
    }

    /// Writes `bytes` under the lock unless the response is done; a failure
    /// flags the client as gone.
    fn write(&self, phase: &mut Phase, bytes: &[u8]) -> std::io::Result<()> {
        if *phase == Phase::Done {
            return Err(std::io::ErrorKind::NotConnected.into());
        }
        let result = (&self.stream).write_all(bytes);
        if result.is_err() {
            self.cancelled.store(true, Ordering::Relaxed);
        }
        result
    }

    fn close(&self, phase: &mut Phase) {
        *phase = Phase::Done;
        (&self.stream).flush().ok();
        self.stream.shutdown(Shutdown::Both).ok();
    }
}

/// A connection whose request the engine is working on, watched for the
/// client going away from the moment the request is queued, before any
/// byte of the response exists: a helper thread blocks in `read` on it, and
/// with `Connection: close` the client sends nothing more, so a read that
/// completes means it left. A reset sets `cancelled` at once; an EOF only
/// once a probe confirmed the peer is gone ([`Conn::departed`]), so a
/// client that half-closed its side after the request is still served (and
/// no longer watched: it can only be noticed by a failing write).
///
/// Dropping it, or the response made from it, closes the connection.
pub struct Watched {
    conn: Arc<Conn>,
}

impl Watched {
    /// Starts watching `stream`. `http11` says whether the request was
    /// HTTP/1.1, which allows the interim probe. A write of the response
    /// that the client does not take within [`WRITE_STALL_TIMEOUT`] fails,
    /// which flags the client as gone like any failed write.
    pub fn new(stream: TcpStream, http11: bool, cancelled: Arc<AtomicBool>) -> Self {
        Self::with_stall(stream, http11, cancelled, WRITE_STALL_TIMEOUT)
    }

    /// [`Self::new`] with `stall` as its [`WRITE_STALL_TIMEOUT`].
    fn with_stall(
        stream: TcpStream,
        http11: bool,
        cancelled: Arc<AtomicBool>,
        stall: Duration,
    ) -> Self {
        stream.set_write_timeout(Some(stall)).ok();
        let reader = stream.try_clone();
        let conn = Arc::new(Conn {
            stream,
            phase: Mutex::new(Phase::Waiting),
            cancelled,
            interim: http11,
        });
        if let Ok(mut reader) = reader {
            let conn = conn.clone();
            std::thread::Builder::new()
                .name("lily-watch".into())
                .spawn(move || {
                    reader.set_read_timeout(None).ok();
                    let mut scratch = [0u8; 256];
                    let reset = loop {
                        match reader.read(&mut scratch) {
                            Ok(0) => break false,
                            Ok(_) => continue,
                            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                            Err(_) => break true,
                        }
                    };
                    let gone = if reset {
                        *conn.phase() != Phase::Done
                    } else {
                        conn.departed()
                    };
                    if gone {
                        conn.cancelled.store(true, Ordering::Relaxed);
                    }
                })
                .ok();
        }
        Self { conn }
    }

    /// Writes a complete response and closes the connection.
    pub fn respond(self, status: u16, content_type: &str, body: &[u8]) -> Result<()> {
        let mut phase = self.conn.phase();
        let length = body.len().to_string();
        let head = head(status, content_type, &[("Content-Length", &length)]);
        let result = self
            .conn
            .write(&mut phase, head.as_bytes())
            .and_then(|_| self.conn.write(&mut phase, body))
            .context("writing response");
        self.conn.close(&mut phase);
        result
    }
}

impl Drop for Watched {
    fn drop(&mut self) {
        let mut phase = self.conn.phase();
        if *phase != Phase::Done {
            self.conn.close(&mut phase);
        }
    }
}

/// A chunked response in progress on a [`Watched`] connection. Dropping it
/// without [`Self::finish`] closes the connection mid-body, which clients
/// report as an error.
pub struct ChunkedResponse {
    watched: Watched,
}

impl ChunkedResponse {
    /// Writes the head. The connection's `cancelled` flag is set as soon as
    /// a write fails or the watcher finds the client gone.
    pub fn start(watched: Watched, status: u16, content_type: &str) -> Result<Self> {
        let conn = &watched.conn;
        conn.stream.set_nodelay(true).ok();
        let events = content_type.starts_with("text/event-stream");
        let extra: &[(&str, &str)] = if events {
            &[
                ("Transfer-Encoding", "chunked"),
                ("Cache-Control", "no-cache"),
                ("X-Accel-Buffering", "no"),
            ]
        } else {
            &[("Transfer-Encoding", "chunked")]
        };
        {
            let mut phase = conn.phase();
            conn.write(&mut phase, head(status, content_type, extra).as_bytes())
                .context("writing response head")?;
            *phase = if events { Phase::Events } else { Phase::Body };
        }
        Ok(Self { watched })
    }

    pub fn write_chunk(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let conn = &self.watched.conn;
        let mut phase = conn.phase();
        let mut chunk = format!("{:x}\r\n", bytes.len()).into_bytes();
        chunk.extend_from_slice(bytes);
        chunk.extend_from_slice(b"\r\n");
        conn.write(&mut phase, &chunk).context("writing chunk")
    }

    pub fn finish(self) -> Result<()> {
        let conn = &self.watched.conn;
        let mut phase = conn.phase();
        let result =
            conn.write(&mut phase, b"0\r\n\r\n").context("writing final chunk");
        conn.close(&mut phase);
        result
    }
}

#[cfg(test)]
#[path = "../../tests/unit/serve/http.rs"]
mod tests;
