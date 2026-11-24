//! The connection watcher over real loopback sockets: a client that closes or
//! resets is flagged, one that only half-closed its side is still served.

use std::io::{Read as _, Write as _};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::os::fd::AsRawFd as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::{
    BODY_CHUNK, BodyTooLarge, ChunkedResponse, Deadlines, MAX_HEADER_BYTES, OverBudget,
    PROBE_WINDOW, Request, Watched, head, read_head_within, read_request, reason,
    respond_raw_within, respond_unread, timed_out,
};

/// The server's and the client's end of one loopback connection, the
/// request already read off the server's end.
fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let client =
        TcpStream::connect(listener.local_addr().expect("addr")).expect("connect");
    let (server, _) = listener.accept().expect("accept");
    (server, client)
}

fn watch(server: TcpStream) -> (Watched, Arc<AtomicBool>) {
    let cancelled = Arc::new(AtomicBool::new(false));
    (Watched::new(server, true, cancelled.clone()), cancelled)
}

/// Whether `flag` is raised within `within`.
fn raised(flag: &AtomicBool, within: Duration) -> bool {
    let deadline = Instant::now() + within;
    while Instant::now() < deadline {
        if flag.load(Ordering::Relaxed) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    flag.load(Ordering::Relaxed)
}

/// Closes `stream` with a reset instead of a FIN (`SO_LINGER` of zero).
fn reset(stream: TcpStream) {
    let linger = libc::linger { l_onoff: 1, l_linger: 0 };
    // SAFETY: a valid socket and a correctly sized option value.
    let rc = unsafe {
        libc::setsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_LINGER,
            (&raw const linger).cast(),
            size_of::<libc::linger>() as libc::socklen_t,
        )
    };
    assert_eq!(rc, 0, "SO_LINGER");
    drop(stream);
}

fn read_all(mut client: TcpStream) -> String {
    // macOS refuses socket options once the connection is closed (EINVAL).
    client.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut out = Vec::new();
    if let Err(e) = client.read_to_end(&mut out) {
        panic!("read: {e} after {:?}", String::from_utf8_lossy(&out));
    }
    String::from_utf8(out).expect("utf-8")
}

#[test]
fn a_client_that_closes_while_its_request_waits_is_flagged() {
    let (server, client) = pair();
    let (_watched, cancelled) = watch(server);
    assert!(!raised(&cancelled, Duration::from_millis(50)), "nothing happened yet");
    drop(client);
    assert!(raised(&cancelled, PROBE_WINDOW * 4), "a closed client must be flagged");
}

#[test]
fn a_client_that_resets_is_flagged() {
    let (server, client) = pair();
    let (_watched, cancelled) = watch(server);
    reset(client);
    assert!(raised(&cancelled, PROBE_WINDOW * 4), "a reset must be flagged");
}

#[test]
fn a_half_closed_client_is_not_flagged_and_gets_its_response() {
    let (server, client) = pair();
    let (watched, cancelled) = watch(server);
    // The client is done sending but still reads.
    client.shutdown(Shutdown::Write).expect("half-close");
    assert!(
        !raised(&cancelled, PROBE_WINDOW * 3),
        "a half-closed client that still reads is not gone"
    );
    watched.respond(200, "application/json", b"{\"ok\":true}").expect("respond");
    let text = read_all(client);
    // The probe was an interim response, which an HTTP/1.1 client skips.
    assert!(
        text.starts_with("HTTP/1.1 102 Processing\r\n\r\nHTTP/1.1 200 OK\r\n"),
        "{text}"
    );
    assert!(text.ends_with("{\"ok\":true}"), "{text}");
    assert!(!cancelled.load(Ordering::Relaxed));
}

#[test]
fn an_http10_client_is_never_probed_before_the_head() {
    let (server, client) = pair();
    let cancelled = Arc::new(AtomicBool::new(false));
    let watched = Watched::new(server, false, cancelled.clone());
    client.shutdown(Shutdown::Write).expect("half-close");
    assert!(!raised(&cancelled, PROBE_WINDOW * 2));
    watched.respond(200, "text/plain", b"hi").expect("respond");
    let text = read_all(client);
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
}

#[test]
fn an_event_stream_is_probed_with_a_comment_and_a_closed_client_is_flagged() {
    // Half-closed during the stream: a comment chunk, then the stream goes on.
    let (server, client) = pair();
    let (watched, cancelled) = watch(server);
    let mut response =
        ChunkedResponse::start(watched, 200, "text/event-stream").expect("head");
    response.write_chunk(b"data: 1\n\n").expect("chunk");
    client.shutdown(Shutdown::Write).expect("half-close");
    assert!(!raised(&cancelled, PROBE_WINDOW * 3));
    response.write_chunk(b"data: 2\n\n").expect("chunk");
    response.finish().expect("finish");
    let text = read_all(client);
    let body = &text[text.find("\r\n\r\n").expect("head end") + 4..];
    assert_eq!(body, "9\r\ndata: 1\n\n\r\n3\r\n:\n\n\r\n9\r\ndata: 2\n\n\r\n0\r\n\r\n");
    assert!(!cancelled.load(Ordering::Relaxed));

    // Closed during the stream: flagged without another write of ours.
    let (server, client) = pair();
    let (watched, cancelled) = watch(server);
    let mut response =
        ChunkedResponse::start(watched, 200, "text/event-stream").expect("head");
    response.write_chunk(b"data: 1\n\n").expect("chunk");
    drop(client);
    assert!(raised(&cancelled, PROBE_WINDOW * 4));
    // Its next writes fail rather than block or succeed.
    std::thread::sleep(Duration::from_millis(20));
    assert!(response.write_chunk(b"data: 2\n\n").is_err());
}

#[test]
fn the_server_closing_the_connection_is_not_a_departure() {
    let (server, client) = pair();
    let (watched, cancelled) = watch(server);
    let response =
        ChunkedResponse::start(watched, 200, "application/json").expect("head");
    response.finish().expect("finish");
    let text = read_all(client);
    assert!(text.ends_with("0\r\n\r\n"), "{text}");
    assert!(!raised(&cancelled, Duration::from_millis(100)));
}

#[test]
fn dropping_a_watched_connection_closes_it() {
    let (server, mut client) = pair();
    let (watched, _cancelled) = watch(server);
    drop(watched);
    client.set_read_timeout(Some(Duration::from_secs(2))).expect("timeout");
    let mut buf = [0u8; 8];
    assert_eq!(client.read(&mut buf).expect("eof"), 0);
    let _ = client.write_all(b"x");
}

#[test]
fn every_status_lily_answers_has_its_reason_phrase() {
    // The OpenAI routes' (the API key's 401, a deadline's 408 and a wrong
    // content type's 415 among them) and the dashboard's (the settings
    // writes' 409, 413 and 504 among them); the interim probe writes its
    // own head.
    let statuses =
        [200, 400, 401, 403, 404, 405, 408, 409, 411, 413, 415, 500, 503, 504];
    for status in statuses {
        assert!(!reason(status).is_empty(), "{status} has no reason phrase");
    }
    assert_eq!(reason(504), "Gateway Timeout");
    assert_eq!(reason(413), "Payload Too Large");
    assert!(
        head(504, "application/json", &[])
            .starts_with("HTTP/1.1 504 Gateway Timeout\r\n")
    );
}

/// Runs `read` on the server's end in a thread; its result, or `None` when
/// it did not return within `within`.
fn read_within<T: Send + 'static>(
    server: TcpStream,
    within: Duration,
    read: impl FnOnce(TcpStream) -> T + Send + 'static,
) -> Option<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(read(server));
    });
    rx.recv_timeout(within).ok()
}

/// Writes `bytes` from a thread (the server may stop reading part way).
fn send(client: &TcpStream, bytes: Vec<u8>) {
    let mut client = client.try_clone().expect("clone");
    std::thread::spawn(move || {
        let _ = client.write_all(&bytes);
    });
}

#[test]
fn a_request_with_a_body_reads_whole() {
    let (server, mut client) = pair();
    client
        .write_all(
            b"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n\
              Content-Length: 11\r\n\r\nhello world",
        )
        .expect("write");
    let request = read_within(server, Duration::from_secs(5), |mut server| {
        read_request(&mut server, 1 << 20)
    })
    .expect("returns")
    .expect("a valid request");
    assert_eq!(
        (request.method.as_str(), request.path.as_str()),
        ("POST", "/v1/completions")
    );
    assert_eq!(request.header("host"), Some("127.0.0.1:8000"));
    assert_eq!(request.body, b"hello world");
}

#[test]
fn a_head_line_without_a_newline_is_refused_at_the_limit_without_waiting() {
    for prefix in [&b""[..], b"GET / HTTP/1.1\r\nX-Long: "] {
        let (server, client) = pair();
        // Twice the limit and no newline; the connection stays open.
        let mut bytes = prefix.to_vec();
        bytes.resize(prefix.len() + 2 * MAX_HEADER_BYTES, b'a');
        send(&client, bytes);
        let result = read_within(server, Duration::from_secs(5), |mut server| {
            read_request(&mut server, 1 << 20).map(drop).map_err(|e| format!("{e:#}"))
        });
        let error = result
            .expect("the head must be refused once it passes the limit, not read on")
            .expect_err("an endless head line is refused");
        assert!(error.contains("too large"), "{error}");
        drop(client);
    }
}

#[test]
fn a_trickled_head_or_body_is_cut_off_at_its_deadline() {
    let deadlines = Deadlines {
        head: Duration::from_millis(300),
        total: Duration::from_millis(600),
        body_idle: Duration::from_secs(30),
    };
    let trickle = |client: &TcpStream, first: &'static [u8]| {
        let mut client = client.try_clone().expect("clone");
        std::thread::spawn(move || {
            let _ = client.write_all(first);
            // A byte every 20 ms: no single read ever waits long.
            while client.write_all(b"a").is_ok() {
                std::thread::sleep(Duration::from_millis(20));
            }
        });
    };
    // The head: header bytes that never end.
    let (server, client) = pair();
    trickle(&client, b"GET /health HTTP/1.1\r\nX-Slow: ");
    let started = Instant::now();
    let result = read_within(server, Duration::from_secs(5), move |server| {
        read_head_within(&server, 1 << 20, deadlines).map(drop)
    });
    let error = result.expect("returns at the head deadline").expect_err("cut off");
    assert!(timed_out(&error), "{error:#}");
    assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
    drop(client);
    // The body: a complete head, then a body that never completes.
    let (server, client) = pair();
    trickle(
        &client,
        b"POST /v1/completions HTTP/1.1\r\nContent-Length: 100000\r\n\r\n",
    );
    let started = Instant::now();
    let result = read_within(server, Duration::from_secs(5), move |server| {
        let (mut request, body) = read_head_within(&server, 1 << 20, deadlines)?;
        body.read(&mut request)
    });
    let error = result.expect("returns at the total deadline").expect_err("cut off");
    assert!(timed_out(&error), "{error:#}");
    assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
    drop(client);
}

#[test]
fn a_body_past_the_limit_is_refused_before_it_is_read() {
    let (server, mut client) = pair();
    client
        .write_all(b"POST /v1/completions HTTP/1.1\r\nContent-Length: 2000000\r\n\r\n")
        .expect("write");
    let error = read_within(server, Duration::from_secs(5), |server| {
        read_head_within(&server, 1 << 20, Deadlines::default()).map(drop)
    })
    .expect("returns")
    .expect_err("refused");
    assert!(format!("{error:#}").contains("exceeds"), "{error:#}");
    // Told apart from a malformed request: the answer is 413, not 400.
    let too_large = error.downcast_ref::<BodyTooLarge>().expect("typed");
    assert_eq!(too_large.limit, 1 << 20);
}

#[test]
fn a_body_is_admitted_as_it_arrives_and_a_refusal_ends_the_read() {
    const LENGTH: usize = 5 * BODY_CHUNK + 123;
    let post = |sent: usize| {
        let (server, mut client) = pair();
        let head = format!(
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {LENGTH}\r\n\r\n"
        );
        client.write_all(head.as_bytes()).expect("head");
        send(&client, (0..sent).map(|i| i as u8).collect());
        (server, client)
    };
    let deadlines = Deadlines {
        head: Duration::from_secs(5),
        total: Duration::from_secs(5),
        body_idle: Duration::from_secs(5),
    };
    // The whole body: admitted in runs of at most a chunk, all of it.
    let (server, _client) = post(LENGTH);
    let (mut request, body) =
        read_head_within(&server, 1 << 20, deadlines).expect("head");
    assert_eq!(body.len(), LENGTH);
    let mut admitted = Vec::new();
    body.read_with(&mut request, &mut |bytes| {
        admitted.push(bytes);
        true
    })
    .expect("read");
    assert!(
        admitted.iter().all(|&bytes| bytes > 0 && bytes <= BODY_CHUNK),
        "{admitted:?}"
    );
    assert_eq!(admitted.iter().sum::<usize>(), LENGTH);
    assert!(request.body.iter().enumerate().all(|(i, &b)| b == i as u8));
    assert_eq!(request.body.len(), LENGTH);
    assert!(request.body.capacity() <= LENGTH, "never past the declared length");
    // A refusal part way: the read ends there, with what says so.
    let (server, _client) = post(LENGTH);
    let (mut request, body) =
        read_head_within(&server, 1 << 20, deadlines).expect("head");
    let mut total = 0;
    let error = body
        .read_with(&mut request, &mut |bytes| {
            total += bytes;
            total <= 2 * BODY_CHUNK
        })
        .expect_err("refused");
    assert!(error.is::<OverBudget>(), "{error:#}");
    assert!(total <= 3 * BODY_CHUNK, "{total}: nothing is read past the refusal");
    assert!(request.body.is_empty());
    // A client that goes before its body is whole: an error, not a short body.
    let (server, client) = post(0);
    let (mut request, body) =
        read_head_within(&server, 1 << 20, deadlines).expect("head");
    drop(client);
    let error = body.read(&mut request).expect_err("closed");
    assert!(!timed_out(&error) && !error.is::<OverBudget>(), "{error:#}");
}

/// A refusal sent while the client is still writing its body reaches the
/// client whole: the server reads on for a moment rather than close a
/// socket that holds unread bytes.
#[test]
fn a_refusal_of_an_unread_body_reaches_a_client_that_is_still_sending() {
    const SENT: usize = 512 << 10;
    let (server, mut client) = pair();
    let head =
        format!("POST /v1/completions HTTP/1.1\r\nContent-Length: {SENT}\r\n\r\n");
    client.write_all(head.as_bytes()).expect("head");
    let (request, _body): (Request, _) =
        read_head_within(&server, 1 << 20, Deadlines::default()).expect("head");
    assert_eq!(request.method, "POST");
    let writer = {
        let mut client = client.try_clone().expect("clone");
        std::thread::spawn(move || client.write_all(&vec![b'x'; SENT]).is_ok())
    };
    let started = Instant::now();
    respond_unread(server, 503, "application/json", &[("Content-Length", "2")], b"{}")
        .expect("respond");
    assert!(started.elapsed() < Duration::from_secs(3), "{:?}", started.elapsed());
    assert!(writer.join().expect("writer"), "the client's body was taken, not reset");
    let mut response = String::new();
    client.read_to_string(&mut response).expect("the response, then the close");
    assert!(response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"), "{response}");
    assert!(response.ends_with("\r\n\r\n{}"), "{response}");
}

#[test]
fn a_refusal_does_not_wait_long_for_a_client_that_sends_nothing_more() {
    let (server, mut client) = pair();
    client
        .write_all(b"POST /v1/completions HTTP/1.1\r\nContent-Length: 1000\r\n\r\n")
        .expect("head");
    let (_request, _body) =
        read_head_within(&server, 1 << 20, Deadlines::default()).expect("head");
    let started = Instant::now();
    respond_unread(server, 401, "application/json", &[("Content-Length", "2")], b"{}")
        .expect("respond");
    let waited = started.elapsed();
    assert!(waited < Duration::from_secs(3), "{waited:?}");
    let mut response = String::new();
    client.read_to_string(&mut response).expect("the response");
    assert!(response.starts_with("HTTP/1.1 401 Unauthorized\r\n"), "{response}");
}

/// More than a loopback connection buffers on both ends together.
const UNTAKEN: usize = 64 << 20;

/// A response is written within its limit or not at all: a client that
/// never reads ends the write with a timeout instead of holding the
/// connection's thread, and so does one that reads a little now and then.
#[test]
fn a_response_the_client_does_not_take_is_given_up_at_its_deadline() {
    let limit = Duration::from_millis(300);
    let respond = move |server: TcpStream| {
        let length = UNTAKEN.to_string();
        let headers = [("Content-Length", length.as_str())];
        let started = Instant::now();
        let written = respond_raw_within(
            server,
            200,
            "application/octet-stream",
            &headers,
            &vec![0u8; UNTAKEN],
            limit,
        );
        (written, started.elapsed())
    };
    // A client that reads nothing.
    let (server, client) = pair();
    let (written, took) =
        read_within(server, Duration::from_secs(10), respond).expect("returns");
    let error = written.expect_err("not written whole");
    assert!(timed_out(&error), "{error:#}");
    assert!(took >= limit && took < Duration::from_secs(5), "{took:?}");
    drop(client);
    // A client that takes a few bytes at a time: each write makes progress,
    // the response as a whole does not make its deadline.
    let (server, mut client) = pair();
    let stop = Arc::new(AtomicBool::new(false));
    let reader = {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut few = [0u8; 4096];
            while !stop.load(Ordering::Relaxed) {
                if !matches!(client.read(&mut few), Ok(got) if got > 0) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        })
    };
    let (written, took) =
        read_within(server, Duration::from_secs(10), respond).expect("returns");
    stop.store(true, Ordering::Relaxed);
    reader.join().expect("reader");
    let error = written.expect_err("not written whole");
    assert!(timed_out(&error), "{error:#}");
    assert!(took < Duration::from_secs(5), "{took:?}");
}

/// A write timeout the caller set on the socket (the accept loop's short
/// one for a refusal) still bounds each write of a response.
#[test]
fn a_shorter_write_timeout_of_the_sockets_own_is_kept() {
    let (server, _client) = pair();
    server.set_write_timeout(Some(Duration::from_millis(100))).expect("timeout");
    let (written, took) = read_within(server, Duration::from_secs(10), |server| {
        let started = Instant::now();
        let written = respond_raw_within(
            server,
            200,
            "application/octet-stream",
            &[],
            &vec![0u8; UNTAKEN],
            Duration::from_secs(30),
        );
        (written, started.elapsed())
    })
    .expect("returns");
    assert!(timed_out(&written.expect_err("not written whole")));
    assert!(took < Duration::from_secs(5), "{took:?}");
}

/// A response that reaches a reading client is not cut short by its limit.
#[test]
fn a_response_that_is_taken_arrives_whole() {
    const SIZE: usize = 4 << 20;
    let (server, client) = pair();
    let reader = std::thread::spawn(move || read_all_bytes(client));
    let length = SIZE.to_string();
    respond_raw_within(
        server,
        200,
        "application/octet-stream",
        &[("Content-Length", length.as_str())],
        &vec![b'x'; SIZE],
        Duration::from_secs(30),
    )
    .expect("written");
    let got = reader.join().expect("reader");
    assert!(got.starts_with(b"HTTP/1.1 200 OK\r\n"));
    assert_eq!(got.iter().filter(|&&b| b == b'x').count(), SIZE);
}

fn read_all_bytes(mut client: TcpStream) -> Vec<u8> {
    client.set_read_timeout(Some(Duration::from_secs(10))).ok();
    let mut out = Vec::new();
    client.read_to_end(&mut out).expect("read");
    out
}

/// A relayed response waits for a slow client, but not for one that takes
/// nothing at all while bytes wait for it: the write fails after the stall
/// timeout and the request is flagged as abandoned, so the engine stops.
#[test]
fn a_relayed_response_gives_up_on_a_client_that_takes_nothing() {
    let (server, client) = pair();
    let cancelled = Arc::new(AtomicBool::new(false));
    let watched = Watched::with_stall(
        server,
        true,
        cancelled.clone(),
        Duration::from_millis(200),
    );
    let (written, took) =
        read_within(client.try_clone().expect("clone"), Duration::from_secs(10), {
            move |_keep_open| {
                let started = Instant::now();
                let mut response =
                    ChunkedResponse::start(watched, 200, "text/event-stream")
                        .expect("head");
                let written = response.write_chunk(&vec![b'x'; UNTAKEN]);
                (written, started.elapsed())
            }
        })
        .expect("returns");
    assert!(written.is_err(), "the chunk was not taken");
    assert!(took < Duration::from_secs(5), "{took:?}");
    assert!(cancelled.load(Ordering::Relaxed), "the client is flagged as gone");
    drop(client);
}

/// The stall timeout is on the socket from the moment it is watched, and
/// leaves a client that reads its stream alone.
#[test]
fn a_watched_connection_has_the_stall_timeout_and_serves_a_reading_client() {
    let (server, client) = pair();
    let socket = server.try_clone().expect("clone");
    let (watched, cancelled) = watch(server);
    assert_eq!(
        socket.write_timeout().expect("timeout"),
        Some(super::WRITE_STALL_TIMEOUT)
    );
    let reader = std::thread::spawn(move || read_all_bytes(client));
    let mut response =
        ChunkedResponse::start(watched, 200, "text/event-stream").expect("head");
    for _ in 0..64 {
        response.write_chunk(&vec![b'x'; 64 << 10]).expect("chunk");
    }
    response.finish().expect("finish");
    let got = reader.join().expect("reader");
    let body = got
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|at| &got[at + 4..])
        .expect("a response head");
    assert_eq!(body.iter().filter(|&&b| b == b'x').count(), 64 * (64 << 10));
    assert!(!cancelled.load(Ordering::Relaxed));
}
