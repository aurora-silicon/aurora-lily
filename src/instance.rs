//! One lily per machine. Every process that loads the model's weights (the
//! server, `lily-bench`, `lily-probe`, `lily-vision-probe`, `lily-experts`
//! and the ignored tests that load a real checkpoint) takes an exclusive
//! advisory lock first, and refuses to run while another process holds it:
//! two full models on a 128 GB machine panicked it on 2026-09-12 (102.8 GB
//! and 68.6 GB resident, 62 MB free).
//!
//! On macOS the lock is machine-wide, not per account: the memory a second
//! model would take is the machine's, whoever starts it, so a lily of
//! another user (root under `sudo` included) is refused the same way. On
//! Linux it is per account ([`default_path`]), in the account's own home
//! (or, for an account without one, in a directory of its own under
//! `/dev/shm`): a multi-user Linux host is not one owner's machine, and its
//! accounts do not share a lock.
//!
//! The lock is `flock(LOCK_EX | LOCK_NB)` on [`default_path`], taken once per
//! process ([`acquire`], from the load path, so a reload after an idle unload
//! finds it held) and never released: the kernel drops it when the process
//! exits, however it exits, so no stale lock can block the next start. The
//! holder writes its pid and binary into the file for the refusal message;
//! the contents are informational only, the lock is what counts.
//!
//! A refused process exits with [`EXIT_ALREADY_RUNNING`] (75, `EX_TEMPFAIL`
//! in `sysexits.h`: try again later). Under the launchd agent that is a
//! failed exit, so launchd starts it again after its 30 s throttle and the
//! service comes back by itself once the other instance has exited.

use std::ffi::{CStr, OsStr, OsString};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Seek as _, Write as _};
use std::os::fd::AsRawFd as _;
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::{
    DirBuilderExt as _, MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _,
};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;

use anyhow::{Context as _, Result, bail, ensure};

/// The exit status of a process refused because another instance runs.
pub const EXIT_ALREADY_RUNNING: u8 = 75;

/// macOS: the lock every weight-loading process of any user takes,
/// `/Users/Shared/lily-instance.lock`. `/Users/Shared` is macOS's directory
/// for data shared between accounts, world-writable with the sticky bit
/// (`drwxrwxrwt`), so every account can create the file and nobody but its
/// owner (or root) can remove or replace it; the file sits directly in it
/// because a subdirectory would carry its creator's mode. Until 2026-10-22
/// the lock was `~/Library/Caches/lily/instance.lock`, per account, so a
/// lily of another user did not see it. Deliberately not configurable, so no
/// two processes can disagree about it; tests lock their own file through
/// [`InstanceLock::acquire_at`].
#[cfg(target_os = "macos")]
pub fn default_path() -> Result<PathBuf> {
    Ok(PathBuf::from(DEFAULT_PATH))
}

/// Linux: the account's own lock, `<home>/.cache/lily/instance.<host>.lock`,
/// the home being the user database's ([`account_home`]), never `HOME`: one
/// path per account and machine whatever the environment or the session, so
/// a lily started from cron, from a login session or from a terminal
/// multiplexer that outlived its session all take the same lock. Until
/// 2026-11 it was `/tmp/lily-instance.lock`, in a directory shared by all
/// accounts. Lily processes of one account exclude each other; another
/// account's lily is not seen.
///
/// `<host>` ([`host_tag`]) names the machine, because a home may be mounted
/// on several machines at once and a lock on a network file system is held
/// on its server: without it the lily of one machine would refuse the same
/// account's lily on another, which shares none of its memory.
///
/// The home and `.cache` must be directories this account owns; their
/// modes are the account's business (a home its group may write is common
/// where every user has a group of their own). `.cache/lily` is lily's:
/// created `0700`, it must be the account's, and one its group or others
/// may write (as an earlier lily left it under a umask of `002`) is made
/// `0700`.
///
/// An account without a home of its own (no entry in the user database, or
/// a home that does not exist or belongs to another account, as `/` does
/// for a system account) locks [`HOMELESS_LOCK_ROOT`]`/lily-<uid>/instance.lock`
/// instead (a place that is the machine's own, so without `<host>`), under
/// the same rules for lily's directory. That too is one path per
/// account: which of the two applies follows from the user database and
/// the home's owner, not from the environment. Anything else that is wrong
/// (a `.cache` of another account, a directory that cannot be created) is
/// an error that says what, never a third place.
#[cfg(target_os = "linux")]
pub fn default_path() -> Result<PathBuf> {
    // SAFETY: getuid cannot fail.
    let uid = unsafe { libc::getuid() };
    let machine_id = std::fs::read_to_string(MACHINE_ID).ok();
    let host = host_tag(machine_id.as_deref(), &host_name());
    account_lock_path(account_home(), Path::new(HOMELESS_LOCK_ROOT), uid, &host)
}

/// Linux: the machine's identity as systemd and D-Bus keep it, 32
/// hexadecimal digits set once when the system is installed.
#[cfg(target_os = "linux")]
const MACHINE_ID: &str = "/etc/machine-id";

/// The `<host>` of the lock's file name under a home ([`default_path`]):
/// the machine id (the contents of [`MACHINE_ID`]) when it is one, since it
/// stays the same while the machine runs and across restarts; otherwise
/// the host name `name` (a system without a machine id, as a minimal
/// container is), with every byte that is not an ASCII letter, a digit, `-`
/// or `_` replaced by `_`; `unnamed` when that is empty too. Which one
/// applies follows from the machine, not from the environment.
#[cfg(target_os = "linux")]
fn host_tag(machine_id: Option<&str>, name: &[u8]) -> String {
    if let Some(id) = machine_id.map(str::trim)
        && id.len() == 32
        && id.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return id.to_ascii_lowercase();
    }
    if name.is_empty() {
        return "unnamed".to_owned();
    }
    name.iter()
        .map(|&b| {
            if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' {
                char::from(b)
            } else {
                '_'
            }
        })
        .collect()
}

/// The kernel's host name (`gethostname`), empty when it cannot be read.
#[cfg(target_os = "linux")]
fn host_name() -> Vec<u8> {
    let mut buf = [0u8; 256];
    // SAFETY: `buf` outlives the call and `buf.len()` is its size.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    if rc != 0 {
        return Vec::new();
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    buf[..len].to_vec()
}

/// Linux: where an account without a home of its own keeps its lock
/// directory ([`default_path`]). A fixed place every account can create a
/// directory in, as `/tmp` is, but one that a service's private `/tmp`
/// (systemd's `PrivateTmp=`, which a dynamic service account implies) does
/// not replace: two services of one account, or a service and a shell,
/// must see the same lock. The directory must be the account's own and is
/// `0700`.
#[cfg(target_os = "linux")]
const HOMELESS_LOCK_ROOT: &str = "/dev/shm";

/// [`default_path`] for the account `uid` whose user database entry names
/// `home` (`None`: no entry), with `homeless_root` as
/// [`HOMELESS_LOCK_ROOT`] and `host` as [`host_tag`].
#[cfg(target_os = "linux")]
fn account_lock_path(
    home: Option<PathBuf>,
    homeless_root: &Path,
    uid: u32,
    host: &str,
) -> Result<PathBuf> {
    let file = match own_home(home, uid).map_err(lock_place_error)? {
        Ok(home) => {
            let cache = home.join(".cache");
            create_private(&cache).map_err(lock_place_error)?;
            // Followed if it is a link: the account may keep its cache on
            // another volume.
            let meta = std::fs::metadata(&cache).map_err(|e| {
                lock_place_error(format!("{} cannot be read: {e}", cache.display()))
            })?;
            if !meta.is_dir() || meta.uid() != uid {
                return Err(lock_place_error(format!(
                    "{} is not a directory of this account (uid {uid})",
                    cache.display()
                )));
            }
            let dir = cache.join("lily");
            own_lock_dir(&dir, uid, true).map_err(lock_place_error)?;
            dir.join(format!("instance.{host}.lock"))
        }
        Err(homeless) => {
            let dir = homeless_root.join(format!("lily-{uid}"));
            // Not followed: the parent is shared, and a link there is
            // another account's.
            own_lock_dir(&dir, uid, false).map_err(|why| {
                lock_place_error(format!(
                    "{homeless}, and the place for an account without a home is not \
                     usable: {why}. Remove it, or give the account a home directory \
                     of its own"
                ))
            })?;
            dir.join("instance.lock")
        }
    };
    Ok(file)
}

#[cfg(target_os = "linux")]
fn lock_place_error(why: String) -> anyhow::Error {
    anyhow::anyhow!(
        "no place for the instance lock: {why}. The lock is kept in <home>/.cache/lily, \
         the home being the account's in the user database, or in {HOMELESS_LOCK_ROOT}/lily-<uid> \
         for an account without a home of its own"
    )
}

/// The home of the account `uid` when it has one of its own: `Ok(Err(why))`
/// when it has none (no entry, a relative path, nothing there, not a
/// directory, another account's), `Err` when the home cannot be judged.
#[cfg(target_os = "linux")]
fn own_home(
    home: Option<PathBuf>,
    uid: u32,
) -> Result<Result<PathBuf, String>, String> {
    let Some(home) = home else {
        return Ok(Err(format!(
            "the user database has no home directory for uid {uid}"
        )));
    };
    if !home.is_absolute() {
        return Ok(Err(format!(
            "the home directory {} of uid {uid} is not an absolute path",
            home.display()
        )));
    }
    let meta = match std::fs::metadata(&home) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Err(format!(
                "the home directory {} of uid {uid} does not exist",
                home.display()
            )));
        }
        Err(e) => return Err(format!("{} cannot be read: {e}", home.display())),
    };
    if !meta.is_dir() {
        return Ok(Err(format!(
            "the home directory {} of uid {uid} is not a directory",
            home.display()
        )));
    }
    if meta.uid() != uid {
        return Ok(Err(format!(
            "the home directory {} of uid {uid} is owned by uid {}",
            home.display(),
            meta.uid()
        )));
    }
    Ok(Ok(home))
}

/// Creates the directory `dir` `0700` unless something is there already.
#[cfg(target_os = "linux")]
fn create_private(dir: &Path) -> Result<(), String> {
    match std::fs::DirBuilder::new().mode(0o700).create(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(format!("{} cannot be created: {e}", dir.display())),
    }
}

/// Makes `dir` lily's own lock directory: created `0700` when missing, it
/// must be a directory owned by `uid`, and is made `0700` when its group or
/// others may write it. The directory is judged and changed through one
/// descriptor, a link in its place followed only with `follow`. Says why
/// not otherwise.
#[cfg(target_os = "linux")]
fn own_lock_dir(dir: &Path, uid: u32, follow: bool) -> Result<(), String> {
    create_private(dir)?;
    let nofollow = if follow { 0 } else { libc::O_NOFOLLOW };
    let handle = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | nofollow)
        .open(dir)
        .map_err(|e| {
            format!("{} cannot be opened as a directory: {e}", dir.display())
        })?;
    let meta = handle
        .metadata()
        .map_err(|e| format!("{} cannot be read: {e}", dir.display()))?;
    if meta.uid() != uid {
        return Err(format!(
            "{} is owned by uid {}, not by this account (uid {uid})",
            dir.display(),
            meta.uid()
        ));
    }
    if meta.mode() & 0o022 != 0 {
        handle.set_permissions(std::fs::Permissions::from_mode(0o700)).map_err(|e| {
            format!(
                "{} is writable by its group or by others (mode {:o}) and could not be \
                 made private: {e}",
                dir.display(),
                meta.mode() & 0o7777
            )
        })?;
    }
    Ok(())
}

/// [`default_path`] as a string, for messages.
#[cfg(target_os = "macos")]
pub const DEFAULT_PATH: &str = "/Users/Shared/lily-instance.lock";

/// The mode the holder gives a lock file it owns. macOS: `0666`, so the
/// next lily of another account can open the machine-wide file for writing
/// too. Linux: `0600`, the account's own file.
#[cfg(target_os = "macos")]
const LOCK_MODE: u32 = 0o666;
#[cfg(target_os = "linux")]
const LOCK_MODE: u32 = 0o600;

/// The account's home directory from the user database, not `HOME`: a
/// process started without `HOME` used to lock `./Library/...` relative to
/// its working directory, never saw the service's lock and could load a
/// second model (2026-10-19, benchmarks started with a cleared
/// environment). An absolute `HOME` stands in when the user database
/// lookup fails (no entry, or a sandbox or directory service refusing it);
/// with neither, this fails rather than guess. The instance lock used it
/// until it became machine-wide (2026-10-22; the Linux lock takes the user
/// database's home alone, [`default_path`]); the disk tier's default
/// directory, which stays per account, uses it.
pub fn home_dir() -> Result<PathBuf> {
    resolve_home(account_home(), std::env::var_os("HOME"))
}

/// [`home_dir`]'s choice: `account` (the user database's entry) when
/// absolute, else `env` (`HOME`) when absolute, else an error.
fn resolve_home(account: Option<PathBuf>, env: Option<OsString>) -> Result<PathBuf> {
    if let Some(home) = account.filter(|p| p.is_absolute()) {
        return Ok(home);
    }
    if let Some(home) = env.map(PathBuf::from).filter(|p| p.is_absolute()) {
        return Ok(home);
    }
    bail!(
        "no home directory: the user database has no entry for this user and HOME is unset \
         or relative"
    )
}

/// The current user's home directory from the user database
/// (`getpwuid_r`), whatever the environment says; `None` without an entry.
fn account_home() -> Option<PathBuf> {
    let mut buf: Vec<libc::c_char> = vec![0; 4096];
    loop {
        // SAFETY: an all-zero `passwd` is a valid out-parameter (null
        // pointers, zero ids) that `getpwuid_r` fills.
        let mut entry: libc::passwd = unsafe { std::mem::zeroed() };
        let mut found: *mut libc::passwd = std::ptr::null_mut();
        // SAFETY: `entry` and `buf` outlive the call, `buf.len()` is its
        // size, and `found` receives either null or `&mut entry`.
        let rc = unsafe {
            libc::getpwuid_r(
                libc::getuid(),
                &mut entry,
                buf.as_mut_ptr(),
                buf.len(),
                &mut found,
            )
        };
        if rc == libc::ERANGE && buf.len() < 1 << 20 {
            buf.resize(buf.len() * 2, 0);
            continue;
        }
        if rc != 0 || found.is_null() || entry.pw_dir.is_null() {
            return None;
        }
        // SAFETY: `pw_dir` points into `buf`, NUL-terminated, and `buf` is
        // still alive; the bytes are copied out before it goes.
        let dir = unsafe { CStr::from_ptr(entry.pw_dir) };
        return Some(PathBuf::from(OsStr::from_bytes(dir.to_bytes())));
    }
}

/// Another process holds the lock.
#[derive(Debug)]
pub struct AlreadyRunning {
    pub path: PathBuf,
    /// What the holder wrote into the file (`pid <n>` and its binary), when
    /// it had written anything yet.
    pub holder: Option<Holder>,
}

/// The holder's own record in the lock file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    pub pid: u32,
    pub binary: String,
}

impl Holder {
    /// The record as the holder wrote it into the lock file; `None` before
    /// it has written one.
    pub fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        let pid = lines.next()?.strip_prefix("pid ")?.trim().parse().ok()?;
        let binary = lines.next().unwrap_or("").trim().to_owned();
        Some(Self { pid, binary })
    }
}

impl fmt::Display for AlreadyRunning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let who = match &self.holder {
            Some(h) if !h.binary.is_empty() => format!("pid {} ({})", h.pid, h.binary),
            Some(h) => format!("pid {}", h.pid),
            None => "a process that has not written its pid yet".to_owned(),
        };
        #[cfg(target_os = "macos")]
        let stop = "If it is the background service, stop it with \
                    `tools/service/lily-service.sh stop` (it stays down until `start`)";
        // `lily-service.sh` manages the launchd agent, which Linux does not have.
        #[cfg(not(target_os = "macos"))]
        let stop = "Stop it first";
        write!(
            f,
            "another lily instance is running: {who} holds {}. Only one process may load the model \
             at a time (two full models do not fit this machine). {stop}",
            self.path.display()
        )
    }
}

impl std::error::Error for AlreadyRunning {}

/// A held lock; dropping it (or exiting) releases it.
#[derive(Debug)]
pub struct InstanceLock {
    _file: File,
    path: PathBuf,
}

impl InstanceLock {
    /// Takes the lock at `path` without waiting, creating the file and its
    /// directory as needed, and records this process in it. Fails with
    /// [`AlreadyRunning`] (downcastable from the error) when another open
    /// file description holds it, which includes another handle in this
    /// same process.
    ///
    /// On macOS the default path is in a directory every account can write
    /// (on Linux it is in one of the account's own), so the file is opened
    /// as one that another user may have planted: a symbolic
    /// link is not followed (`O_NOFOLLOW`), and anything but a regular file
    /// with a single link is refused, since the holder truncates the file
    /// to write its record and would otherwise truncate whatever a link
    /// points at. A file this process owns is made [`LOCK_MODE`] whatever
    /// the umask (`0666` on macOS, so the next lily of another user can open
    /// it for writing too; `0600` on Linux).
    /// A file another account owns without granting write access (its
    /// creator was killed between the open and the `fchmod`, or it was
    /// planted), or one on a file system mounted read-only, is opened
    /// read-only instead: `flock` needs no write access, so the exclusion
    /// holds, and only the record is left unwritten.
    pub fn acquire_at(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(dir)
                .with_context(|| format!("creating {}", dir.display()))?;
        }
        // Not truncated on open: the holder's record must survive a
        // refused open so the refusal can name it.
        let open = |write: bool| {
            OpenOptions::new()
                .read(true)
                .write(write)
                .create(write)
                .truncate(false)
                .mode(LOCK_MODE)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path)
        };
        let (opened, writable) = match open(true) {
            Err(e)
                if e.kind() == std::io::ErrorKind::PermissionDenied
                    || e.raw_os_error() == Some(libc::EROFS) =>
            {
                (open(false), false)
            }
            opened => (opened, true),
        };
        let mut file = opened
            .with_context(|| format!("opening the instance lock {}", path.display()))?;
        let meta = file
            .metadata()
            .with_context(|| format!("reading the instance lock {}", path.display()))?;
        ensure!(
            meta.file_type().is_file() && meta.nlink() == 1,
            "the instance lock {} is not a regular file with a single link (it may have been \
             planted by another account); remove it with `sudo rm {}`",
            path.display(),
            path.display()
        );
        // SAFETY: getuid cannot fail.
        if writable
            && meta.uid() == unsafe { libc::getuid() }
            && meta.mode() & 0o777 != LOCK_MODE
        {
            file.set_permissions(std::fs::Permissions::from_mode(LOCK_MODE))
                .with_context(|| {
                    format!("setting the mode of the instance lock {}", path.display())
                })?;
        }
        // SAFETY: flock on a descriptor we own.
        let rc =
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
                let mut text = String::new();
                let holder = file
                    .read_to_string(&mut text)
                    .ok()
                    .and_then(|_| Holder::parse(&text));
                return Err(AlreadyRunning { path: path.to_owned(), holder }.into());
            }
            return Err(error).with_context(|| {
                format!("locking the instance lock {}", path.display())
            });
        }
        if !writable {
            // The previous holder's record stays: a refusal may name a pid
            // that is gone, which `lily-service.sh` and `lily-ssd-probe`
            // already treat as stale.
            return Ok(Self { _file: file, path: path.to_owned() });
        }
        let binary = std::env::current_exe()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "unknown binary".to_owned());
        file.set_len(0)?;
        file.rewind()?;
        write!(file, "pid {}\n{binary}\n", std::process::id())
            .and_then(|()| file.flush())
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(Self { _file: file, path: path.to_owned() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The process's lock once taken; never dropped, so it lives until exit.
static HELD: Mutex<Option<InstanceLock>> = Mutex::new(None);

/// Takes the process-wide lock at [`default_path`] unless this process
/// already holds it: the first model load (or the server's start) takes it,
/// every later load (a reload after an idle unload or a GPU fault) finds it
/// held. Errors with [`AlreadyRunning`] when another process holds it.
pub fn acquire() -> Result<()> {
    let mut held = HELD.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if held.is_none() {
        *held = Some(InstanceLock::acquire_at(&default_path()?)?);
    }
    Ok(())
}

/// The exit status for an error that ends a process: 75 when it is (or was
/// caused by) [`AlreadyRunning`], 1 otherwise.
pub fn exit_status(error: &anyhow::Error) -> u8 {
    if error.chain().any(|e| e.is::<AlreadyRunning>()) {
        EXIT_ALREADY_RUNNING
    } else {
        1
    }
}

/// What a binary's `main` returns for `result`: success, or the error
/// printed the way a `main` returning `Result` prints it and the exit
/// status [`exit_status`] picks, except that a refusal prints only its
/// message.
pub fn exit_code(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            match error.chain().find_map(|e| e.downcast_ref::<AlreadyRunning>()) {
                Some(refused) => {
                    eprintln!(
                        "error: {refused}; exiting with status {EXIT_ALREADY_RUNNING}"
                    )
                }
                None => eprintln!("Error: {error:?}"),
            }
            ExitCode::from(exit_status(&error))
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/instance.rs"]
mod tests;
