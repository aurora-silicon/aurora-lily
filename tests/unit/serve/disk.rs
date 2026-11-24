use std::io::Read;

use super::*;

fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir()
        .join(format!("lily-disk-test-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    root
}

fn store_fake(
    store: &mut DiskStore,
    tokens: &[u32],
    prefix_bytes: usize,
    key: Option<&str>,
) -> Option<String> {
    let n = tokens.len();
    store
        .store(
            tokens,
            &[],
            key,
            None,
            &[n],
            &mut |w| Ok(w.write_all(&vec![7u8; prefix_bytes])?),
            &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
        )
        .expect("store")
}

fn store_fake_durable(
    store: &mut DiskStore,
    tokens: &[u32],
    prefix_bytes: usize,
) -> String {
    let n = tokens.len();
    store
        .store_durable(
            tokens,
            &[],
            None,
            None,
            &[n],
            &mut |w| Ok(w.write_all(&vec![9u8; prefix_bytes])?),
            &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
        )
        .expect("store durable")
        .expect("kept")
}

fn ids(store: &DiskStore) -> Vec<String> {
    let mut ids: Vec<String> = store.entries().iter().map(|e| e.id.clone()).collect();
    ids.sort();
    ids
}

#[test]
fn durable_flag_round_trips_and_old_meta_files_load_as_not_durable() {
    let root = temp_root("durable-meta");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let plain = store_fake(&mut store, &[1, 2, 3], 10, Some("k")).expect("plain");
    let durable = store_fake_durable(&mut store, &[1, 2, 3, 4], 10);
    assert_eq!((store.len(), store.durable_len()), (2, 1));
    let flag = |store: &DiskStore, id: &str| {
        store.entries().iter().find(|e| e.id == id).map(|e| e.durable)
    };
    assert_eq!(
        (flag(&store, &plain), flag(&store, &durable)),
        (Some(false), Some(true))
    );

    // The flag is in meta.json and survives a touch (which rewrites it).
    let meta_path = |id: &str| root.join("fmt").join(id).join(META);
    let read_meta = |id: &str| -> serde_json::Value {
        serde_json::from_slice(&fs::read(meta_path(id)).expect("meta")).expect("json")
    };
    assert_eq!(read_meta(&durable)["durable"], serde_json::Value::Bool(true));
    assert_eq!(read_meta(&plain)["durable"], serde_json::Value::Bool(false));
    store.touch(&durable);
    assert_eq!(read_meta(&durable)["durable"], serde_json::Value::Bool(true));

    // Reopening rebuilds the index with the flag.
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    assert_eq!(
        (flag(&reopened, &plain), flag(&reopened, &durable)),
        (Some(false), Some(true))
    );
    assert_eq!(reopened.durable_len(), 1);
    drop(reopened);

    // A meta file from before the field existed has no `durable` key: it is
    // an evicted session, not a durable entry.
    let mut old = read_meta(&durable);
    old.as_object_mut().expect("object").remove("durable");
    fs::write(meta_path(&durable), serde_json::to_vec(&old).expect("json"))
        .expect("write");
    let legacy = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen legacy");
    assert_eq!(flag(&legacy, &durable), Some(false));
    assert_eq!(legacy.durable_len(), 0);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_durable_cap_evicts_the_least_recently_used_durable_entry_only() {
    let root = temp_root("durable-cap");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    // An evicted session first: older than everything, yet never a victim of
    // the durable cap.
    let plain = store_fake(&mut store, &[100], 10, None).expect("plain");
    let durables: Vec<String> = (0..DURABLE_MAX_ENTRIES as u32)
        .map(|i| store_fake_durable(&mut store, &[i, i + 1], 10))
        .collect();
    assert_eq!(
        (store.len(), store.durable_len()),
        (DURABLE_MAX_ENTRIES + 1, DURABLE_MAX_ENTRIES)
    );
    // A normal store on top does not count toward the cap and evicts nothing.
    let plain2 = store_fake(&mut store, &[200], 10, None).expect("plain2");
    assert_eq!(
        (store.len(), store.durable_len()),
        (DURABLE_MAX_ENTRIES + 2, DURABLE_MAX_ENTRIES)
    );
    drop(store);

    // Make the third durable entry the least recently used one (the plain
    // entries older still) and reopen so the index carries the timestamps.
    backdate(&root, &durables[2], 60);
    backdate(&root, &plain, 600);
    backdate(&root, &plain2, 600);
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    let newest = store_fake_durable(&mut store, &[7, 7, 7], 10);
    assert_eq!(
        (store.len(), store.durable_len()),
        (DURABLE_MAX_ENTRIES + 2, DURABLE_MAX_ENTRIES)
    );
    let present = ids(&store);
    assert!(
        !present.contains(&durables[2]),
        "the least recently used durable entry goes: {present:?}"
    );
    assert!(!root.join("fmt").join(&durables[2]).exists());
    for keep in durables
        .iter()
        .filter(|id| **id != durables[2])
        .chain([&plain, &plain2, &newest])
    {
        assert!(present.contains(keep), "{keep} must survive: {present:?}");
    }
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn image_spans_round_trip_through_meta_and_old_meta_files_load_as_text() {
    let root = temp_root("images-meta");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let images = vec![
        CachedImage { start: 2, len: 4, grid_h: 2, grid_w: 8, digest: [7; 32] },
        CachedImage { start: 9, len: 1, grid_h: 2, grid_w: 2, digest: [9; 32] },
    ];
    let tokens = [1, 2, 99, 99, 99, 99, 3, 4, 5, 99, 6];
    let id = store
        .store(
            &tokens,
            &images,
            None,
            None,
            &[tokens.len()],
            &mut |w| Ok(w.write_all(&[1u8; 10])?),
            &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
        )
        .expect("store")
        .expect("kept");
    assert_eq!(store.entries()[0].images, images);
    // A span past the tokens is refused.
    let err = store
        .store(
            &tokens,
            &[CachedImage { start: 8, len: 4, grid_h: 2, grid_w: 8, digest: [0; 32] }],
            None,
            None,
            &[tokens.len()],
            &mut |w| Ok(w.write_all(&[1u8; 10])?),
            &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("extends past"), "{err}");

    // The spans are in meta.json, survive a touch and come back on reopen.
    let meta_path = root.join("fmt").join(&id).join(META);
    let read_meta = || -> serde_json::Value {
        serde_json::from_slice(&fs::read(&meta_path).expect("meta")).expect("json")
    };
    assert_eq!(read_meta()["images"].as_array().map(Vec::len), Some(2));
    assert_eq!(read_meta()["images"][0]["start"], serde_json::json!(2));
    store.touch(&id);
    assert_eq!(read_meta()["images"][1]["digest"][0], serde_json::json!(9));
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    assert_eq!(reopened.entries()[0].images, images);

    // A meta file from before the field existed (no `images`) loads as a
    // text-only lineage.
    let mut legacy = read_meta();
    legacy.as_object_mut().unwrap().remove("images");
    fs::write(&meta_path, serde_json::to_vec(&legacy).unwrap()).expect("write");
    let legacy_store =
        DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen legacy");
    assert_eq!(legacy_store.len(), 1);
    assert!(legacy_store.entries()[0].images.is_empty());
    assert_eq!(legacy_store.entries()[0].tokens, tokens);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn store_reopen_and_read_back() {
    let root = temp_root("roundtrip");
    let mut store = DiskStore::open(&root, "fmt-a", 1 << 20, 0).expect("open");
    assert!(store.is_empty());
    let id = store_fake(&mut store, &[1, 2, 3, 4], 100, Some("conv")).expect("stored");
    assert_eq!(store.len(), 1);
    assert_eq!(store.used_bytes(), 108);

    // A second store instance indexes the entry from disk.
    let store2 = DiskStore::open(&root, "fmt-a", 1 << 20, 0).expect("reopen");
    assert_eq!(store2.len(), 1);
    let e = &store2.entries()[0];
    assert_eq!(
        (
            e.id.as_str(),
            e.tokens.as_slice(),
            e.checkpoints.as_slice(),
            e.cache_key.as_deref()
        ),
        (id.as_str(), &[1, 2, 3, 4][..], &[4][..], Some("conv"))
    );
    let mut prefix = Vec::new();
    store2.open_prefix(&id).expect("prefix").read_to_end(&mut prefix).expect("read");
    assert_eq!(prefix, vec![7u8; 100]);
    let mut ckpt = Vec::new();
    store2.open_checkpoint(&id, 4).expect("ckpt").read_to_end(&mut ckpt).expect("read");
    assert_eq!(ckpt, 4u64.to_le_bytes());

    // Another format shares the root but sees nothing.
    let other = DiskStore::open(&root, "fmt-b", 1 << 20, 0).expect("open other");
    assert!(other.is_empty());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn budget_evicts_least_recently_used_and_refuses_oversize() {
    let root = temp_root("budget");
    // Entries are 308 bytes (300 of prefix, 8 of checkpoint): two fit, three do not.
    let mut store = DiskStore::open(&root, "fmt", 700, 0).expect("open");
    let a = store_fake(&mut store, &[1], 300, None).expect("a");
    let b = store_fake(&mut store, &[2], 300, None).expect("b");
    // Touch a so b becomes the oldest.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    store.touch(&a);
    let c = store_fake(&mut store, &[3], 300, None).expect("c");
    let ids: Vec<&str> = store.entries().iter().map(|e| e.id.as_str()).collect();
    assert!(
        ids.contains(&a.as_str())
            && ids.contains(&c.as_str())
            && !ids.contains(&b.as_str()),
        "{ids:?}"
    );
    assert!(!root.join("fmt").join(&b).exists());
    // Larger than the whole budget: not kept, nothing else evicted.
    assert!(store_fake(&mut store, &[4], 2000, None).is_none());
    assert_eq!(store.len(), 2);
    assert!(!root.join("fmt").join("s4").exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn corrupt_entries_are_dropped_on_open() {
    let root = temp_root("corrupt");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let a = store_fake(&mut store, &[1, 2], 10, None).expect("a");
    let b = store_fake(&mut store, &[3, 4], 10, None).expect("b");
    fs::remove_file(root.join("fmt").join(&b).join("ckpt-2.bin")).expect("remove");
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    let ids: Vec<&str> = reopened.entries().iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, vec![a.as_str()]);
    assert!(!root.join("fmt").join(&b).exists());
    // Ids continue past the ones seen on disk.
    let mut reopened = reopened;
    let c = store_fake(&mut reopened, &[5], 10, None).expect("c");
    assert_ne!(c, a);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn remove_deletes_files() {
    let root = temp_root("remove");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let a = store_fake(&mut store, &[1], 10, None).expect("a");
    assert!(root.join("fmt").join(&a).is_dir());
    store.remove(&a);
    assert!(store.is_empty());
    assert!(!root.join("fmt").join(&a).exists());
    let _ = fs::remove_dir_all(&root);
}

/// Backdates an entry's last use by rewriting its meta file.
fn backdate(root: &Path, id: &str, seconds: u64) {
    let path = root.join("fmt").join(id).join(META);
    let mut meta: Meta =
        serde_json::from_slice(&fs::read(&path).expect("meta")).expect("parse");
    meta.last_used -= seconds;
    fs::write(&path, serde_json::to_vec(&meta).expect("json")).expect("write");
}

#[test]
fn entries_expire_after_the_maximum_age() {
    let root = temp_root("ttl");
    let day = 86_400;
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 3 * day).expect("open");
    let old = store_fake(&mut store, &[1], 10, None).expect("old");
    let fresh = store_fake(&mut store, &[2], 10, None).expect("fresh");
    backdate(&root, &old, 4 * day);
    // The in-memory index still holds the old timestamp until reopened; a
    // reopen applies the age limit.
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 3 * day).expect("reopen");
    let ids: Vec<&str> = reopened.entries().iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, vec![fresh.as_str()]);
    assert!(!root.join("fmt").join(&old).exists());
    drop(reopened);
    backdate(&root, &fresh, 4 * day);
    let mut relisted =
        DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen without ttl");
    assert_eq!(relisted.len(), 1, "no age limit keeps it");
    relisted.expire();
    assert_eq!(relisted.len(), 1, "max_age 0 never expires");
    drop(relisted);
    let live = DiskStore::open(&root, "fmt", 1 << 20, 3 * day).expect("reopen");
    assert!(live.is_empty());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_maximum_age_is_read_at_each_expiry_pass() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    let root = temp_root("ttl-runtime");
    let day = 86_400;
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let old = store_fake(&mut store, &[1], 10, None).expect("old");
    backdate(&root, &old, 4 * day);
    drop(store);
    // The runtime settings' handle, 0 at first: nothing expires.
    let ttl = Arc::new(AtomicU64::new(0));
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, ttl.load(Ordering::Acquire))
        .expect("reopen")
        .with_max_age(ttl.clone());
    store.expire();
    assert_eq!(store.len(), 1);
    // A change applies at the next pass, without a reopen.
    ttl.store(5 * day, Ordering::Release);
    assert_eq!(store.max_age_secs(), 5 * day);
    store.expire();
    assert_eq!(store.len(), 1, "4 days unused is inside 5");
    ttl.store(3 * day, Ordering::Release);
    store.expire();
    assert!(store.is_empty(), "and outside 3");
    assert!(!root.join("fmt").join(&old).exists());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn an_entry_the_stored_ttl_keeps_survives_a_start_whose_flag_ttl_would_expire_it() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};
    let root = temp_root("ttl-startup");
    let day = 86_400;
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let old = store_fake(&mut store, &[1], 10, None).expect("old");
    backdate(&root, &old, 4 * day);
    drop(store);
    // The start: `--disk-cache-ttl 3d` in effect, the database (5d) not
    // applied yet, the bound far off.
    let ttl = Arc::new(AtomicU64::new(3 * day));
    let settled = Arc::new(AtomicBool::new(false));
    let gate =
        ExpiryGate::new(settled.clone(), Instant::now() + Duration::from_secs(3600));
    let mut store = DiskStore::open_gated(&root, "fmt", 1 << 20, ttl.clone(), gate)
        .expect("reopen");
    assert_eq!(store.len(), 1, "no expiry at the open");
    assert!(!store.expiring());
    store.expire();
    assert_eq!(store.len(), 1, "nor before the overrides are settled");
    // The recorder applies the stored 5d, then settles.
    ttl.store(5 * day, Ordering::Release);
    settled.store(true, Ordering::Release);
    assert!(store.expiring());
    store.expire();
    assert_eq!(store.len(), 1, "4 days unused is inside the stored 5");
    // Had the flag's TTL stayed, the first pass after settling expires it.
    ttl.store(3 * day, Ordering::Release);
    store.expire();
    assert!(store.is_empty());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_gate_that_is_never_settled_opens_at_its_bound() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::time::Instant;
    let root = temp_root("ttl-bound");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let old = store_fake(&mut store, &[1], 10, None).expect("old");
    backdate(&root, &old, 10);
    drop(store);
    let gate = ExpiryGate::new(Arc::new(AtomicBool::new(false)), Instant::now());
    let mut store =
        DiskStore::open_gated(&root, "fmt", 1 << 20, Arc::new(AtomicU64::new(5)), gate)
            .expect("reopen");
    assert!(store.is_empty(), "past its bound the gate is open, the open expires");
    store.expire();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn a_reserved_entry_is_invisible_until_committed_and_a_crash_leaves_nothing_readable() {
    let root = temp_root("reserve");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let tokens = [1u32, 2, 3];
    let reserved = store.reserve();
    let bytes = write_files(
        reserved.path(),
        &[3],
        &mut |w| Ok(w.write_all(&[7u8; 50])?),
        &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
    )
    .expect("write");
    assert_eq!(bytes, 58);
    // Written but not committed: no lookup sees it, and a restart (a crash
    // before the commit) drops the directory, which has no meta file.
    assert!(store.is_empty());
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    assert!(reopened.is_empty());
    assert!(!reserved.path().exists());

    // A committed one is indexed; an abandoned one is deleted.
    let reserved = store.reserve();
    let id = reserved.id().to_owned();
    let bytes = write_files(
        reserved.path(),
        &[3],
        &mut |w| Ok(w.write_all(&[7u8; 50])?),
        &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
    )
    .expect("write");
    let new = NewEntry {
        tokens: &tokens,
        images: &[],
        cache_key: None,
        checkpoints: &[3],
        durable: false,
        lineage: None,
    };
    assert_eq!(store.commit(reserved, &new, bytes).expect("commit"), Some(id.clone()));
    assert!(store.contains(&id));
    let abandoned = store.reserve();
    assert_ne!(abandoned.id(), id, "ids are never reused");
    fs::create_dir_all(abandoned.path()).expect("mkdir");
    let path = abandoned.path().to_path_buf();
    store.abandon(abandoned);
    assert!(!path.exists());
    assert_eq!(ids(&store), vec![id]);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn an_eviction_names_the_entry_its_size_and_whether_it_was_durable() {
    let entry = |durable| DiskEntry {
        id: "s84".into(),
        tokens: vec![0; 15_689],
        checkpoints: vec![15_689],
        bytes: 480_000_000,
        last_used: 0,
        cache_key: None,
        durable,
        images: vec![],
        lineage: None,
    };
    assert_eq!(
        eviction_line(&entry(true), "to make room"),
        "session cache: evicted s84 (15689 tokens, durable, 0.5 GB) from disk to make room"
    );
    assert_eq!(
        eviction_line(&entry(false), "to make room"),
        "session cache: evicted s84 (15689 tokens, 0.5 GB) from disk to make room"
    );
}

#[test]
fn lineage_ids_round_trip_through_meta_and_old_meta_files_load_without_one() {
    let root = temp_root("lineage-meta");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    let lineage = Lineage { id: 41, created_ms: 1_760_000_000_000 };
    let id = store
        .store(
            &[1, 2, 3],
            &[],
            None,
            Some(lineage),
            &[3],
            &mut |w| Ok(w.write_all(&[7u8; 10])?),
            &mut |pos, w| Ok(w.write_all(&(pos as u64).to_le_bytes())?),
        )
        .expect("store")
        .expect("kept");
    let lineage_of = |store: &DiskStore| {
        store.entries().iter().find(|e| e.id == id).and_then(|e| e.lineage)
    };
    assert_eq!(lineage_of(&store), Some(lineage));
    assert_eq!(store.max_lineage(), Some(41));

    // In meta.json, kept by a touch (which rewrites it) and by a reopen.
    let meta_path = root.join("fmt").join(&id).join(META);
    let read_meta = || -> serde_json::Value {
        serde_json::from_slice(&fs::read(&meta_path).expect("meta")).expect("json")
    };
    assert_eq!(read_meta()["lineage"]["id"], 41);
    store.touch(&id);
    assert_eq!(read_meta()["lineage"]["created_ms"], 1_760_000_000_000u64);
    let reopened = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen");
    assert_eq!(lineage_of(&reopened), Some(lineage));
    drop(reopened);

    // A meta file from before the field existed loads as an entry without
    // a lineage, and contributes nothing to the counter's floor.
    let mut old = read_meta();
    old.as_object_mut().expect("object").remove("lineage");
    fs::write(&meta_path, serde_json::to_vec(&old).expect("json")).expect("write");
    let legacy = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("reopen legacy");
    assert_eq!(legacy.len(), 1);
    assert_eq!(lineage_of(&legacy), None);
    assert_eq!(legacy.max_lineage(), None);
    let _ = fs::remove_dir_all(&root);
}

/// The permission bits of `path`.
fn mode(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt as _;
    fs::metadata(path).expect("metadata").permissions().mode() & 0o777
}

#[test]
fn the_tier_is_private_to_the_account_whatever_the_umask() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = temp_root("private");
    // A tier directory left by an older lily with the usual umask.
    fs::create_dir_all(root.join("fmt")).expect("mkdir");
    fs::set_permissions(root.join("fmt"), fs::Permissions::from_mode(0o755))
        .expect("chmod");
    let mut store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    assert_eq!(
        mode(&root.join("fmt")),
        0o700,
        "an existing tier directory is tightened"
    );
    let id = store_fake(&mut store, &[1, 2, 3], 10, Some("k")).expect("stored");
    let entry = root.join("fmt").join(&id);
    assert_eq!(mode(&entry), 0o700, "{}", entry.display());
    for file in [META, PREFIX, "ckpt-3.bin"] {
        assert_eq!(mode(&entry.join(file)), 0o600, "{file}");
    }
    store.touch(&id);
    assert_eq!(mode(&entry.join(META)), 0o600, "a touch keeps the meta file private");
    // A tier created from nothing: every directory lily makes is private.
    let fresh = temp_root("private-fresh");
    DiskStore::open(&fresh.join("lily").join("sessions"), "fmt", 1 << 20, 0)
        .expect("open");
    for dir in [
        fresh.join("lily"),
        fresh.join("lily/sessions"),
        fresh.join("lily/sessions/fmt"),
    ] {
        assert_eq!(mode(&dir), 0o700, "{}", dir.display());
    }
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&fresh);
}

/// A tier directory that is a link is opened as before and what it points
/// at keeps its mode: only a directory of its own is tightened.
#[test]
fn a_linked_tier_directory_keeps_its_targets_mode() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = temp_root("linked");
    let target = temp_root("linked-target");
    fs::create_dir_all(&root).expect("mkdir");
    fs::create_dir_all(&target).expect("mkdir");
    fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).expect("chmod");
    std::os::unix::fs::symlink(&target, root.join("fmt")).expect("symlink");
    let mut store =
        DiskStore::open(&root, "fmt", 1 << 20, 0).expect("the tier still opens");
    assert_eq!(mode(&target), 0o755, "the link's target is not changed");
    // The tier works through the link, its entries private as elsewhere.
    let id = store_fake(&mut store, &[1, 2, 3], 10, None).expect("stored");
    assert_eq!(mode(&target.join(&id)), 0o700);
    assert_eq!(mode(&target.join(&id).join(META)), 0o600);
    // A directory already closed to group and others is left alone too,
    // its owner's bits included.
    let closed = temp_root("closed");
    fs::create_dir_all(closed.join("fmt")).expect("mkdir");
    fs::set_permissions(closed.join("fmt"), fs::Permissions::from_mode(0o500))
        .expect("chmod");
    DiskStore::open(&closed, "fmt", 1 << 20, 0).expect("open");
    assert_eq!(mode(&closed.join("fmt")), 0o500);
    fs::set_permissions(closed.join("fmt"), fs::Permissions::from_mode(0o700))
        .expect("chmod");
    for dir in [root, target, closed] {
        let _ = fs::remove_dir_all(&dir);
    }
}

/// Opening one format makes private what an earlier lily left under the
/// root for every format, entries and files included, and nothing that is
/// not the tier's: the root may hold other things.
#[test]
fn opening_the_tier_makes_every_formats_entries_private_and_nothing_else() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = temp_root("private-all");
    let chmod = |path: &Path, mode: u32| {
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).expect("chmod");
    };
    // Two formats written by an older lily under the usual umask.
    let mut entries = Vec::new();
    for format in ["fmt", "old"] {
        let mut store = DiskStore::open(&root, format, 1 << 20, 0).expect("open");
        let id = store_fake(&mut store, &[1, 2, 3], 10, None).expect("stored");
        let entry = root.join(format).join(id);
        chmod(&root.join(format), 0o755);
        chmod(&entry, 0o755);
        for file in [META, PREFIX, "ckpt-3.bin"] {
            chmod(&entry.join(file), 0o644);
        }
        entries.push(entry);
    }
    // Not the tier's: a directory without entries, with a file and a
    // directory of its own; and a file directly under the root.
    let other = root.join("models");
    fs::create_dir_all(other.join("bin")).expect("mkdir");
    fs::write(other.join("bin/tool"), "").expect("write");
    fs::write(root.join("notes.txt"), "").expect("write");
    chmod(&other, 0o755);
    chmod(&other.join("bin"), 0o755);
    chmod(&other.join("bin/tool"), 0o755);
    chmod(&root.join("notes.txt"), 0o644);
    chmod(&root, 0o755);
    // A link to a format's directory elsewhere, and a link inside an entry.
    let outside = temp_root("private-all-outside");
    let mut elsewhere = DiskStore::open(&outside, "linked", 1 << 20, 0).expect("open");
    let linked_id = store_fake(&mut elsewhere, &[4, 5], 10, None).expect("stored");
    let linked_entry = outside.join("linked").join(linked_id);
    chmod(&outside.join("linked"), 0o755);
    chmod(&linked_entry, 0o755);
    chmod(&linked_entry.join(META), 0o644);
    std::os::unix::fs::symlink(outside.join("linked"), root.join("linked"))
        .expect("symlink");
    fs::write(outside.join("file"), "").expect("write");
    chmod(&outside.join("file"), 0o644);
    std::os::unix::fs::symlink(outside.join("file"), entries[0].join("extra.bin"))
        .expect("symlink");

    let store = DiskStore::open(&root, "fmt", 1 << 20, 0).expect("open");
    assert_eq!(store.entries().len(), 1, "the format's own entry is still there");
    for entry in &entries {
        let format_dir = entry.parent().expect("parent");
        assert_eq!(mode(format_dir), 0o700, "{}", format_dir.display());
        assert_eq!(mode(entry), 0o700, "{}", entry.display());
        for file in [META, PREFIX, "ckpt-3.bin"] {
            assert_eq!(mode(&entry.join(file)), 0o600, "{}/{file}", entry.display());
        }
    }
    assert_eq!(mode(&root), 0o755, "the root is the user's choice");
    assert_eq!(mode(&other), 0o755);
    assert_eq!(mode(&other.join("bin")), 0o755);
    assert_eq!(mode(&other.join("bin/tool")), 0o755);
    assert_eq!(mode(&root.join("notes.txt")), 0o644);
    assert_eq!(mode(&outside.join("linked")), 0o755, "a link is not followed");
    assert_eq!(mode(&linked_entry), 0o755);
    assert_eq!(mode(&linked_entry.join(META)), 0o644);
    assert_eq!(mode(&outside.join("file")), 0o644, "nor a link inside an entry");
    for dir in [root, outside] {
        let _ = fs::remove_dir_all(&dir);
    }
}
