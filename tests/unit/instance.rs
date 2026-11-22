use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Child, Command, Stdio};

use super::*;

/// Set in the environment of the child process the lock tests spawn: the
/// lock file it takes.
const CHILD_ENV: &str = "LILY_TEST_INSTANCE_LOCK_CHILD";

/// Held by every test that takes a lock. A child spawned by one test can
/// inherit another test's lock descriptor between its fork and its exec
/// (close-on-exec only closes it at the exec), which holds that lock for a
/// moment and failed "dropping frees it" now and then under a parallel run.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn temp_lock(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("lily-instance-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir.join("nested").join("instance.lock")
}

/// Not a test of its own: the body of the child process. It takes the lock
/// named by [`CHILD_ENV`], says so on stdout and holds it until its stdin
/// says `exit` (or it is killed).
#[test]
#[ignore = "run as the child process of the lock tests"]
fn lock_holder_child() {
    let Some(path) = std::env::var_os(CHILD_ENV) else { return };
    let lock =
        InstanceLock::acquire_at(Path::new(&path)).expect("child takes the lock");
    println!("child holds the lock");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    drop(lock);
}

/// Spawns this test binary as [`lock_holder_child`] on `path` and waits
/// until it holds the lock.
fn spawn_holder(path: &Path) -> Child {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "instance::tests::lock_holder_child",
            "--ignored",
            "--nocapture",
        ])
        .env(CHILD_ENV, path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn the lock holder");
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    loop {
        match lines.next() {
            Some(Ok(line)) if line.contains("child holds the lock") => break,
            Some(Ok(_)) => {}
            other => panic!("the lock holder ended before taking the lock: {other:?}"),
        }
    }
    // The harness in the child reports its result on stdout when it ends; a
    // closed pipe would fail that report.
    std::thread::spawn(move || lines.for_each(drop));
    child
}

fn refusal(path: &Path) -> AlreadyRunning {
    let error = InstanceLock::acquire_at(path).expect_err("the lock is held");
    assert_eq!(exit_status(&error), EXIT_ALREADY_RUNNING);
    error.downcast::<AlreadyRunning>().expect("refused as AlreadyRunning")
}

#[test]
fn a_second_process_is_refused_with_the_holders_pid_until_the_holder_is_killed() {
    let _serial = serial();
    let path = temp_lock("killed");
    let mut child = spawn_holder(&path);
    let refused = refusal(&path);
    let holder = refused.holder.clone().expect("the holder recorded itself");
    assert_eq!(holder.pid, child.id());
    assert!(!holder.binary.is_empty());
    let message = refused.to_string();
    assert!(message.contains(&format!("pid {}", child.id())), "{message}");
    // The launchd service and its script are macOS's.
    #[cfg(target_os = "macos")]
    assert!(message.contains("tools/service/lily-service.sh stop"), "{message}");
    assert!(message.contains(&path.display().to_string()), "{message}");

    // SIGKILL: no destructor runs, the kernel still drops the lock.
    child.kill().unwrap();
    child.wait().unwrap();
    let lock =
        InstanceLock::acquire_at(&path).expect("free after the holder was killed");
    assert_eq!(lock.path(), path);
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(Holder::parse(&text).unwrap().pid, std::process::id());
}

#[test]
fn the_lock_is_free_again_after_the_holder_exits() {
    let _serial = serial();
    let path = temp_lock("exited");
    let mut child = spawn_holder(&path);
    assert_eq!(refusal(&path).holder.map(|h| h.pid), Some(child.id()));
    child.stdin.take().unwrap().write_all(b"exit\n").unwrap();
    assert!(child.wait().unwrap().success());
    InstanceLock::acquire_at(&path).expect("free after the holder exited");
}

#[test]
fn a_second_handle_in_the_same_process_is_refused_and_dropping_frees_it() {
    let _serial = serial();
    let path = temp_lock("handles");
    let first = InstanceLock::acquire_at(&path).expect("first");
    let refused = refusal(&path);
    assert_eq!(refused.holder.map(|h| h.pid), Some(std::process::id()));
    drop(first);
    InstanceLock::acquire_at(&path).expect("free after the first handle dropped");
}

/// Set in the environment of [`lock_path_child`]: print the default lock
/// path and exit.
const PATH_CHILD_ENV: &str = "LILY_TEST_INSTANCE_PATH_CHILD";

/// Not a test of its own: the body of a child process that prints
/// [`default_path`] as it resolves in the child's environment.
#[test]
#[ignore = "run as the child process of the lock path test"]
fn lock_path_child() {
    if std::env::var_os(PATH_CHILD_ENV).is_none() {
        return;
    }
    let path = default_path().expect("the default lock path");
    println!("lock path: {}", path.display());
}

/// The lock path a child process of this test binary resolves with `home`
/// as its `HOME` (`None`: unset), in another working directory.
fn child_lock_path(home: Option<&str>) -> String {
    child_lock_path_with(&[("HOME", home)])
}

/// [`child_lock_path`] with each of `envs` set (or, with `None`, unset) in
/// the child's environment.
fn child_lock_path_with(envs: &[(&str, Option<&str>)]) -> String {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "instance::tests::lock_path_child",
            "--ignored",
            "--nocapture",
        ])
        .env(PATH_CHILD_ENV, "1")
        .current_dir(std::env::temp_dir())
        .stderr(Stdio::inherit());
    for (name, value) in envs {
        match value {
            Some(value) => command.env(name, value),
            None => command.env_remove(name),
        };
    }
    let output = command.output().expect("run the path child");
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("lock path: ").map(str::to_owned))
        .expect("the child printed its lock path")
}

/// Every process locks the same file whatever its `HOME`: unset, relative,
/// or pointing elsewhere, as a benchmark started with a cleared environment
/// once did. On macOS it is the machine-wide one.
#[test]
fn the_lock_path_does_not_depend_on_home() {
    // Children forked now could inherit another test's lock descriptor.
    let _serial = serial();
    let here = default_path().expect("the default lock path");
    #[cfg(target_os = "macos")]
    assert_eq!(here, Path::new("/Users/Shared/lily-instance.lock"));
    let here = here.display().to_string();
    assert_eq!(child_lock_path(None), here, "HOME unset");
    assert_eq!(child_lock_path(Some(".")), here, "HOME relative");
    assert_eq!(child_lock_path(Some("/nonexistent/elsewhere")), here, "HOME elsewhere");
}

/// A lock file this process owns gets [`LOCK_MODE`] whatever the umask
/// left it with (here `0644`, as a umask of `022` creates it): `0666` on
/// macOS, so the next lily of another account can open it too, `0600` on
/// Linux, where the lock is the account's own.
#[test]
fn an_owned_lock_file_gets_the_lock_mode() {
    use std::os::unix::fs::PermissionsExt as _;
    let _serial = serial();
    let path = temp_lock("mode");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    let lock = InstanceLock::acquire_at(&path).expect("lock");
    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
    #[cfg(target_os = "macos")]
    assert_eq!(mode & 0o777, 0o666, "{mode:o}");
    #[cfg(target_os = "linux")]
    assert_eq!(mode & 0o777, 0o600, "{mode:o}");
    drop(lock);
}

/// Linux: the lock sits where no other account can create, hold or
/// replace it, never in a directory every account can write such as
/// `/tmp`: its directory is this account's and closed to everyone else's
/// writes.
#[cfg(target_os = "linux")]
#[test]
fn the_linux_lock_is_in_a_directory_only_this_account_can_write() {
    use std::os::unix::fs::MetadataExt as _;
    let path = default_path().expect("the default lock path");
    assert!(!path.starts_with("/tmp"), "{}", path.display());
    // SAFETY: getuid cannot fail.
    let uid = unsafe { libc::getuid() };
    let dir = path.parent().expect("the lock's directory");
    let meta = std::fs::metadata(dir).expect("metadata");
    assert!(meta.is_dir(), "{}", dir.display());
    assert_eq!(meta.uid(), uid, "{} belongs to this account", dir.display());
    assert_eq!(meta.mode() & 0o022, 0, "{} is writable by others", dir.display());
}

/// Linux: one path per account, whatever the environment says about the
/// session: a runtime directory that is set, unset or elsewhere changes
/// nothing (a lily started from cron and one started from a login must
/// take the same lock).
#[cfg(target_os = "linux")]
#[test]
fn the_linux_lock_path_does_not_depend_on_the_session() {
    let _serial = serial();
    let here = default_path().expect("the default lock path");
    // SAFETY: getuid cannot fail.
    let uid = unsafe { libc::getuid() };
    let homeless = Path::new(HOMELESS_LOCK_ROOT).join(format!("lily-{uid}"));
    let at_home = account_home().map(|home| home.join(".cache/lily"));
    let dir = here.parent().expect("the lock's directory");
    assert!(
        Some(dir) == at_home.as_deref() || dir == homeless,
        "{} is neither under the user database's home nor {}",
        here.display(),
        homeless.display()
    );
    let here = here.display().to_string();
    let runtime =
        std::env::temp_dir().join(format!("lily-runtime-{}", std::process::id()));
    std::fs::create_dir_all(&runtime).unwrap();
    let runtime_dir = runtime.to_str().expect("utf-8");
    for (what, envs) in [
        ("a session's runtime directory", vec![("XDG_RUNTIME_DIR", Some(runtime_dir))]),
        ("no runtime directory", vec![("XDG_RUNTIME_DIR", None)]),
        (
            "another cache home",
            vec![("XDG_CACHE_HOME", Some(runtime_dir)), ("XDG_RUNTIME_DIR", None)],
        ),
        ("another HOME", vec![("HOME", Some(runtime_dir)), ("XDG_RUNTIME_DIR", None)]),
        ("another TMPDIR", vec![("TMPDIR", Some(runtime_dir))]),
    ] {
        assert_eq!(child_lock_path_with(&envs), here, "{what}");
    }
    let _ = std::fs::remove_dir_all(&runtime);
}

/// A scratch directory for the lock path tests, its permission helpers and
/// this account's uid.
#[cfg(target_os = "linux")]
fn lock_place(name: &str) -> (PathBuf, u32) {
    let root =
        std::env::temp_dir().join(format!("lily-lock-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    // SAFETY: getuid cannot fail.
    (root, unsafe { libc::getuid() })
}

#[cfg(target_os = "linux")]
fn chmod(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

#[cfg(target_os = "linux")]
fn mode_of(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::symlink_metadata(path).unwrap().permissions().mode() & 0o7777
}

/// Linux: under a home that is the account's own the lock's directories
/// are created `0700`. The home's and `.cache`'s modes are the account's
/// business (a home its group may write is refused nowhere); lily's own
/// directory is closed to the writes of others when it was left open.
#[cfg(target_os = "linux")]
#[test]
fn the_linux_lock_is_kept_under_a_home_of_the_accounts_own_whatever_its_mode() {
    let (root, uid) = lock_place("home");
    let shm = root.join("shm");
    std::fs::create_dir(&shm).unwrap();
    for open in [0o755, 0o775, 0o770, 0o777, 0o700] {
        let home = root.join(format!("home-{open:o}"));
        std::fs::create_dir(&home).unwrap();
        chmod(&home, open);
        let path = account_lock_path(Some(home.clone()), &shm, uid, "box")
            .unwrap_or_else(|e| panic!("a home with mode {open:o}: {e:#}"));
        assert_eq!(path, home.join(".cache/lily/instance.box.lock"));
        assert_eq!(mode_of(&home.join(".cache")), 0o700);
        assert_eq!(mode_of(&home.join(".cache/lily")), 0o700);
        assert_eq!(
            account_lock_path(Some(home.clone()), &shm, uid, "box").expect("again"),
            path
        );
        assert_eq!(mode_of(&home), open, "the home itself is not changed");
    }
    let home = root.join("home-755");
    // A `.cache` its group may write (a umask of 002) is the account's
    // business too and stays as it is.
    chmod(&home.join(".cache"), 0o775);
    assert!(account_lock_path(Some(home.clone()), &shm, uid, "box").is_ok());
    assert_eq!(mode_of(&home.join(".cache")), 0o775);
    // Lily's own directory, left writable for others, is closed to them.
    for open in [0o775, 0o777, 0o757] {
        chmod(&home.join(".cache/lily"), open);
        account_lock_path(Some(home.clone()), &shm, uid, "box")
            .unwrap_or_else(|e| panic!("lily's directory with mode {open:o}: {e:#}"));
        assert_eq!(mode_of(&home.join(".cache/lily")), 0o700, "from {open:o}");
    }
    // One that only lets others look is left alone.
    chmod(&home.join(".cache/lily"), 0o755);
    assert!(account_lock_path(Some(home.clone()), &shm, uid, "box").is_ok());
    assert_eq!(mode_of(&home.join(".cache/lily")), 0o755);
    // A `.cache` that is a file is an error, not another place.
    let broken = root.join("broken");
    std::fs::create_dir(&broken).unwrap();
    std::fs::write(broken.join(".cache"), "").unwrap();
    let error = account_lock_path(Some(broken), &shm, uid, "box").expect_err("a file");
    assert!(
        format!("{error:#}").contains("is not a directory of this account"),
        "{error:#}"
    );
    assert_eq!(std::fs::read_dir(&shm).unwrap().count(), 0, "nothing fell back");
    let _ = std::fs::remove_dir_all(&root);
}

/// Linux: under one home the lock is per machine, so a home mounted on two
/// machines gives each its own file and neither refuses the other; the
/// machine is named by its machine id, or by its host name where it has
/// none, made safe for a file name.
#[cfg(target_os = "linux")]
#[test]
fn the_linux_lock_under_a_home_is_named_after_the_machine() {
    let _serial = serial();
    let (root, uid) = lock_place("host");
    let shm = root.join("shm");
    let home = root.join("home");
    std::fs::create_dir(&shm).unwrap();
    std::fs::create_dir(&home).unwrap();
    let one = account_lock_path(Some(home.clone()), &shm, uid, "one").expect("one");
    let two = account_lock_path(Some(home.clone()), &shm, uid, "two").expect("two");
    assert_eq!(one, home.join(".cache/lily/instance.one.lock"));
    assert_eq!(two, home.join(".cache/lily/instance.two.lock"));
    let held = InstanceLock::acquire_at(&one).expect("the first machine's lock");
    let other = InstanceLock::acquire_at(&two).expect("the second machine's lock");
    // The same machine is still refused.
    let again = account_lock_path(Some(home.clone()), &shm, uid, "one").expect("one");
    assert_eq!(refusal(&again).path, one);
    drop((held, other));

    let id = "0123456789abcdef0123456789abcdef";
    assert_eq!(host_tag(Some(&format!("{id}\n")), b"box"), id);
    assert_eq!(host_tag(Some(&id.to_uppercase()), b"box"), id);
    // Not a machine id (missing, empty, as before its first boot, or not
    // 32 hexadecimal digits): the host name.
    for not_an_id in
        [None, Some(""), Some("uninitialized\n"), Some(&id[1..]), Some("../x")]
    {
        assert_eq!(host_tag(not_an_id, b"box-1"), "box-1", "{not_an_id:?}");
    }
    assert_eq!(host_tag(None, b"box.example.org"), "box_example_org");
    assert_eq!(host_tag(None, b"../a b\xff"), "___a_b_");
    assert_eq!(host_tag(None, b""), "unnamed");
    // What this machine resolves to is a single path component.
    let here = default_path().expect("the default lock path");
    let name = here.file_name().and_then(|n| n.to_str()).expect("a file name");
    assert!(name.starts_with("instance.") && name.ends_with(".lock"), "{name}");
    let _ = std::fs::remove_dir_all(&root);
}

/// Linux: an account without a home of its own (no entry in the user
/// database, a home that is relative, missing, a file or another
/// account's) keeps its lock in one fixed directory of its own, `0700`,
/// the same on every start.
#[cfg(target_os = "linux")]
#[test]
fn an_account_without_a_home_locks_one_fixed_directory_of_its_own() {
    let _serial = serial();
    let (root, uid) = lock_place("homeless");
    let shm = root.join("shm");
    std::fs::create_dir(&shm).unwrap();
    chmod(&shm, 0o1777);
    let expected = shm.join(format!("lily-{uid}")).join("instance.lock");
    std::fs::write(root.join("file"), "").unwrap();
    let mut homes = vec![
        None,
        Some(PathBuf::from("relative")),
        Some(root.join("missing")),
        Some(root.join("file")),
    ];
    if uid != 0 {
        // Root's `/`, as a system account has.
        homes.push(Some(PathBuf::from("/")));
    }
    for home in homes {
        let path = account_lock_path(home.clone(), &shm, uid, "box")
            .unwrap_or_else(|e| panic!("{home:?}: {e:#}"));
        assert_eq!(path, expected, "{home:?}");
    }
    assert!(!root.join("missing").exists(), "a missing home is not created");
    let dir = expected.parent().unwrap();
    assert_eq!(mode_of(dir), 0o700);
    // The lock is taken there like anywhere else.
    let lock = InstanceLock::acquire_at(&expected).expect("lock");
    assert_eq!(mode_of(&expected), 0o600);
    drop(lock);
    // Left writable for others, the directory is closed to them again.
    chmod(dir, 0o777);
    assert_eq!(account_lock_path(None, &shm, uid, "box").expect("again"), expected);
    assert_eq!(mode_of(dir), 0o700);
    let _ = std::fs::remove_dir_all(&root);
}

/// Linux: the directory of an account without a home is used only when it
/// is a directory of the account's own. A link or a file in its place, or
/// a directory of another account, is an error that names it, and nothing
/// is created or changed through it.
#[cfg(target_os = "linux")]
#[test]
fn a_homeless_lock_directory_that_is_not_the_accounts_own_is_refused() {
    let (root, uid) = lock_place("planted");
    let refusal = |shm: &Path, uid: u32| {
        let error = account_lock_path(None, shm, uid, "box").expect_err("refused");
        format!("{error:#}")
    };
    // A link to a directory of this account.
    let shm = root.join("linked");
    let target = root.join("target");
    std::fs::create_dir_all(&shm).unwrap();
    std::fs::create_dir_all(&target).unwrap();
    chmod(&target, 0o777);
    std::os::unix::fs::symlink(&target, shm.join(format!("lily-{uid}"))).unwrap();
    let linked = refusal(&shm, uid);
    assert!(linked.contains("cannot be opened as a directory"), "{linked}");
    assert!(linked.contains("no home directory"), "{linked}");
    assert_eq!(mode_of(&target), 0o777, "the link's target is not changed");
    assert_eq!(std::fs::read_dir(&target).unwrap().count(), 0);
    // A file.
    let shm = root.join("file");
    std::fs::create_dir_all(&shm).unwrap();
    std::fs::write(shm.join(format!("lily-{uid}")), "").unwrap();
    let file = refusal(&shm, uid);
    assert!(file.contains("cannot be opened as a directory"), "{file}");
    // A directory another account owns: this account's, judged as another
    // uid's.
    let shm = root.join("other");
    let other = uid + 1;
    std::fs::create_dir_all(shm.join(format!("lily-{other}"))).unwrap();
    chmod(&shm.join(format!("lily-{other}")), 0o777);
    let owned = refusal(&shm, other);
    assert!(owned.contains(&format!("is owned by uid {uid}")), "{owned}");
    assert_eq!(mode_of(&shm.join(format!("lily-{other}"))), 0o777, "left as it is");
    let _ = std::fs::remove_dir_all(&root);
}

/// A lock file this process may not write (as one of another account
/// without write access for others) is still locked, through a read-only
/// descriptor, and excludes a second handle; its record stays as it was.
#[test]
fn a_lock_file_without_write_access_is_locked_read_only() {
    use std::os::unix::fs::PermissionsExt as _;
    let _serial = serial();
    let path = temp_lock("readonly");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "pid 1\n/old/lily\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
    let lock = InstanceLock::acquire_at(&path).expect("locked read-only");
    let refused = refusal(&path);
    assert_eq!(refused.holder.map(|h| h.pid), Some(1));
    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o444, "{mode:o}");
    drop(lock);
    InstanceLock::acquire_at(&path).expect("free after the handle dropped");
}

/// A link planted at the lock's path is refused rather than followed, and
/// the file it points at is left as it was: the holder truncates the lock
/// file to write its record.
#[test]
fn a_planted_symlink_or_hard_link_is_refused_and_its_target_untouched() {
    let _serial = serial();
    let path = temp_lock("links");
    let dir = path.parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    let victim = dir.join("victim");
    std::fs::write(&victim, "precious").unwrap();

    std::os::unix::fs::symlink(&victim, &path).unwrap();
    let error = InstanceLock::acquire_at(&path).expect_err("symlink refused");
    assert_eq!(exit_status(&error), 1, "{error:?}");
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious");

    std::fs::remove_file(&path).unwrap();
    std::fs::hard_link(&victim, &path).unwrap();
    let error = InstanceLock::acquire_at(&path).expect_err("hard link refused");
    assert!(format!("{error}").contains("single link"), "{error}");
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious");
}

/// A FIFO planted at the lock's path neither blocks the open (read and
/// write: no wait for the other end) nor passes as the lock file.
#[test]
fn a_planted_fifo_is_refused_without_blocking() {
    use std::os::unix::ffi::OsStrExt as _;
    let _serial = serial();
    let path = temp_lock("fifo");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: a NUL-terminated path that outlives the call.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o666) }, 0);
    let error = InstanceLock::acquire_at(&path).expect_err("FIFO refused");
    assert!(format!("{error}").contains("not a regular file"), "{error}");
}

#[test]
fn the_home_falls_back_to_an_absolute_home_and_never_to_a_relative_one() {
    let account = Some(PathBuf::from("/Users/account"));
    assert_eq!(
        resolve_home(account.clone(), Some("/Users/env".into())).unwrap(),
        PathBuf::from("/Users/account")
    );
    assert_eq!(
        resolve_home(None, Some("/Users/env".into())).unwrap(),
        PathBuf::from("/Users/env")
    );
    assert_eq!(
        resolve_home(Some("relative".into()), Some("/Users/env".into())).unwrap(),
        PathBuf::from("/Users/env")
    );
    for env in [None, Some(OsString::from(".")), Some(OsString::from(""))] {
        let error = resolve_home(None, env.clone()).expect_err("no absolute home");
        assert!(format!("{error}").contains("no home directory"), "{env:?}: {error}");
    }
}

#[test]
fn a_refusal_behind_context_still_exits_75_and_other_errors_exit_1() {
    let refused =
        anyhow::Error::from(AlreadyRunning { path: "/x".into(), holder: None })
            .context("loading the model");
    assert_eq!(exit_status(&refused), 75);
    assert_eq!(exit_status(&anyhow::anyhow!("no checkpoint")), 1);
    // An empty or foreign file names no holder rather than failing.
    assert_eq!(Holder::parse(""), None);
    assert_eq!(Holder::parse("garbage"), None);
    assert_eq!(
        Holder::parse("pid 42\n/usr/local/bin/lily\n"),
        Some(Holder { pid: 42, binary: "/usr/local/bin/lily".into() })
    );
}
