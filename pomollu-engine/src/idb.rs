use acvus_orchestration::{BlobHash, BlobStore};
use js_sys::{Promise, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbObjectStore, IdbRequest, IdbTransaction, IdbTransactionMode};

use crate::fetch::UnsafeSend;

// ── Constants ──────────────────────────────────────────────────────

const DB_NAME: &str = "blob_storage";
const DB_VERSION: u32 = 1;
const BLOBS_STORE: &str = "blobs";
const REFS_STORE: &str = "refs";

// ── IdbBlobStore ───────────────────────────────────────────────────

pub struct IdbBlobStore {
    db: IdbDatabase,
    session_id: String,
}

// SAFETY: WASM is single-threaded. IdbDatabase is !Send+!Sync but
// there is no concurrent access.
unsafe impl Send for IdbBlobStore {}
unsafe impl Sync for IdbBlobStore {}

// ── Helpers ────────────────────────────────────────────────────────

/// Wrap an `IdbRequest` into a `Promise` and await it via `JsFuture`.
async fn idb_request(request: &IdbRequest) -> Result<JsValue, JsValue> {
    let promise = Promise::new(&mut |resolve, reject| {
        let resolve2 = resolve.clone();
        let request_ref = request.clone();
        let on_success = Closure::once(move |_: JsValue| match request_ref.result() {
            Ok(val) => {
                let _ = resolve2.call1(&JsValue::NULL, &val);
            }
            Err(e) => {
                let _ = reject.call1(&JsValue::NULL, &e);
            }
        });
        request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        on_success.forget();

        let on_error = Closure::once(move |_: JsValue| {
            let _ = resolve.call1(&JsValue::NULL, &JsValue::from_str("IdbRequest error"));
        });
        request.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    });
    JsFuture::from(promise).await
}

/// Await an `IdbTransaction` completing (oncomplete / onerror).
async fn tx_done(tx: &IdbTransaction) -> Result<(), JsValue> {
    let promise = Promise::new(&mut |resolve, reject| {
        let on_complete = Closure::once(move |_: JsValue| {
            let _ = resolve.call0(&JsValue::NULL);
        });
        tx.set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
        on_complete.forget();

        let on_error = Closure::once(move |_: JsValue| {
            let _ = reject.call1(&JsValue::NULL, &JsValue::from_str("transaction error"));
        });
        tx.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        on_error.forget();
    });
    JsFuture::from(promise).await.map(|_| ())
}

/// Parse a 64-char hex string into a `BlobHash`.
fn hash_from_hex(hex: &str) -> BlobHash {
    let mut bytes = [0u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        let hi = hex.as_bytes()[i * 2];
        let lo = hex.as_bytes()[i * 2 + 1];
        *byte = (hex_nibble(hi) << 4) | hex_nibble(lo);
    }
    BlobHash::from_bytes(bytes)
}

fn hex_nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0,
    }
}

// ── Construction ───────────────────────────────────────────────────

impl IdbBlobStore {
    /// Open (or create) the IndexedDB database and return a ready store.
    pub async fn open(session_id: String) -> Self {
        let db = UnsafeSend(async {
            let window = web_sys::window().expect("no window");
            let factory = window
                .indexed_db()
                .expect("indexed_db() failed")
                .expect("IndexedDB not available");

            let open_req = factory
                .open_with_u32(DB_NAME, DB_VERSION)
                .expect("open_with_u32 failed");

            // onupgradeneeded: create object stores if absent.
            let open_req_ref = open_req.clone();
            let on_upgrade = Closure::once(move |_: JsValue| {
                let db: IdbDatabase = open_req_ref
                    .result()
                    .expect("result() in onupgradeneeded")
                    .unchecked_into();
                let store_names = db.object_store_names();
                if !store_names.contains(BLOBS_STORE) {
                    db.create_object_store(BLOBS_STORE)
                        .expect("create blobs store");
                }
                if !store_names.contains(REFS_STORE) {
                    db.create_object_store(REFS_STORE)
                        .expect("create refs store");
                }
            });
            open_req.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
            on_upgrade.forget();

            let db_val = idb_request(&open_req).await.expect("failed to open IDB");
            let db: IdbDatabase = db_val.unchecked_into();
            db
        })
        .await;

        Self { db, session_id }
    }

    // ── Internal helpers ───────────────────────────────────────────

    fn blob_store(&self, mode: IdbTransactionMode) -> (IdbTransaction, IdbObjectStore) {
        let tx = self
            .db
            .transaction_with_str_and_mode(BLOBS_STORE, mode)
            .expect("transaction on blobs");
        let store = tx.object_store(BLOBS_STORE).expect("open blobs store");
        (tx, store)
    }

    fn refs_store(&self, mode: IdbTransactionMode) -> (IdbTransaction, IdbObjectStore) {
        let tx = self
            .db
            .transaction_with_str_and_mode(REFS_STORE, mode)
            .expect("transaction on refs");
        let store = tx.object_store(REFS_STORE).expect("open refs store");
        (tx, store)
    }

    fn ref_key(&self, name: &str) -> String {
        format!("{}:{}", self.session_id, name)
    }
}

// ── BlobStore impl ─────────────────────────────────────────────────

impl BlobStore for IdbBlobStore {
    async fn put(&mut self, data: Vec<u8>) -> BlobHash {
        let hash = BlobHash::of(&data);
        let hex_key = hash.to_string();

        UnsafeSend(async {
            let (_tx, store) = self.blob_store(IdbTransactionMode::Readwrite);
            let arr = Uint8Array::from(data.as_slice());
            let key = JsValue::from_str(&hex_key);
            let req = store.put_with_key(&arr, &key).expect("put_with_key");
            idb_request(&req).await.expect("put request failed");
        })
        .await;

        hash
    }

    async fn get(&self, hash: &BlobHash) -> Option<Vec<u8>> {
        let hex_key = hash.to_string();

        UnsafeSend(async {
            let (_tx, store) = self.blob_store(IdbTransactionMode::Readonly);
            let key = JsValue::from_str(&hex_key);
            let req = store.get(&key).expect("get request");
            let result = idb_request(&req).await.expect("get failed");

            if result.is_undefined() || result.is_null() {
                None
            } else {
                let arr: Uint8Array = result.unchecked_into();
                Some(arr.to_vec())
            }
        })
        .await
    }

    async fn remove(&mut self, hash: &BlobHash) {
        let hex_key = hash.to_string();

        UnsafeSend(async {
            let (_tx, store) = self.blob_store(IdbTransactionMode::Readwrite);
            let key = JsValue::from_str(&hex_key);
            let req = store.delete(&key).expect("delete request");
            idb_request(&req).await.expect("delete failed");
        })
        .await;
    }

    async fn ref_get(&self, name: &str) -> Option<BlobHash> {
        let ref_key = self.ref_key(name);

        UnsafeSend(async {
            let (_tx, store) = self.refs_store(IdbTransactionMode::Readonly);
            let key = JsValue::from_str(&ref_key);
            let req = store.get(&key).expect("ref_get request");
            let result = idb_request(&req).await.expect("ref_get failed");

            if result.is_undefined() || result.is_null() {
                None
            } else {
                let hex: String = result.as_string().expect("ref value should be a string");
                Some(hash_from_hex(&hex))
            }
        })
        .await
    }

    async fn ref_cas(
        &mut self,
        name: &str,
        expected: Option<BlobHash>,
        new: BlobHash,
    ) -> Result<(), Option<BlobHash>> {
        let ref_key = self.ref_key(name);
        let new_hex = new.to_string();

        UnsafeSend(async {
            let (tx, store) = self.refs_store(IdbTransactionMode::Readwrite);
            let key = JsValue::from_str(&ref_key);

            // Read current value within the same readwrite transaction.
            let req = store.get(&key).expect("ref_cas get request");
            let result = idb_request(&req).await.expect("ref_cas get failed");

            let current = if result.is_undefined() || result.is_null() {
                None
            } else {
                let hex: String = result.as_string().expect("ref value should be a string");
                Some(hash_from_hex(&hex))
            };

            if current == expected {
                let val = JsValue::from_str(&new_hex);
                let req = store.put_with_key(&val, &key).expect("ref_cas put");
                idb_request(&req).await.expect("ref_cas put failed");
                tx_done(&tx).await.expect("ref_cas tx_done");
                Ok(())
            } else {
                Err(current)
            }
        })
        .await
    }

    async fn ref_remove(&mut self, name: &str) {
        let ref_key = self.ref_key(name);

        UnsafeSend(async {
            let (_tx, store) = self.refs_store(IdbTransactionMode::Readwrite);
            let key = JsValue::from_str(&ref_key);
            let req = store.delete(&key).expect("ref_remove delete");
            idb_request(&req).await.expect("ref_remove failed");
        })
        .await;
    }

    async fn batch_put(&mut self, blobs: Vec<Vec<u8>>) -> Vec<BlobHash> {
        if blobs.is_empty() {
            return vec![];
        }

        let pairs: Vec<(BlobHash, Vec<u8>)> = blobs
            .into_iter()
            .map(|data| {
                let hash = BlobHash::of(&data);
                (hash, data)
            })
            .collect();
        let hashes: Vec<BlobHash> = pairs.iter().map(|(h, _)| *h).collect();

        UnsafeSend(async {
            let (tx, store) = self.blob_store(IdbTransactionMode::Readwrite);
            for (hash, data) in &pairs {
                let arr = Uint8Array::from(data.as_slice());
                let key = JsValue::from_str(&hash.to_string());
                store
                    .put_with_key(&arr, &key)
                    .expect("batch_put put_with_key");
            }
            tx_done(&tx).await.expect("batch_put tx_done");
        })
        .await;

        hashes
    }

    async fn batch_get(&self, hashes: &[BlobHash]) -> Vec<Option<Vec<u8>>> {
        if hashes.is_empty() {
            return vec![];
        }

        UnsafeSend(async {
            let (tx, store) = self.blob_store(IdbTransactionMode::Readonly);

            // Issue all get requests.
            let requests: Vec<IdbRequest> = hashes
                .iter()
                .map(|h| {
                    let key = JsValue::from_str(&h.to_string());
                    store.get(&key).expect("batch_get get")
                })
                .collect();

            tx_done(&tx).await.expect("batch_get tx_done");

            // Collect results after transaction completes.
            requests
                .iter()
                .map(|req| {
                    let result = req.result().expect("batch_get result");
                    if result.is_undefined() || result.is_null() {
                        None
                    } else {
                        let arr: Uint8Array = result.unchecked_into();
                        Some(arr.to_vec())
                    }
                })
                .collect()
        })
        .await
    }

    async fn batch_remove(&mut self, hashes: Vec<BlobHash>) {
        if hashes.is_empty() {
            return;
        }

        UnsafeSend(async {
            let (tx, store) = self.blob_store(IdbTransactionMode::Readwrite);
            for hash in &hashes {
                let key = JsValue::from_str(&hash.to_string());
                store.delete(&key).expect("batch_remove delete");
            }
            tx_done(&tx).await.expect("batch_remove tx_done");
        })
        .await;
    }
}

// ── IdbAssetStore ─────────────────────────────────────────────────

const ASSETS_STORE: &str = "assets";
const META_STORE: &str = "meta";

pub struct IdbAssetStore {
    db: IdbDatabase,
    db_name: String,
}

// SAFETY: WASM is single-threaded. IdbDatabase is !Send+!Sync but
// there is no concurrent access.
unsafe impl Send for IdbAssetStore {}
unsafe impl Sync for IdbAssetStore {}

impl IdbAssetStore {
    /// Open (or create) an asset IndexedDB database.
    ///
    /// DB name is passed in from the caller (e.g. `asset_{bot_id}`).
    pub async fn open(db_name: &str) -> Self {
        let db_name = db_name.to_string();
        let db_name_clone = db_name.clone();
        let db = UnsafeSend(async {
            let window = web_sys::window().expect("no window");
            let factory = window
                .indexed_db()
                .expect("indexed_db() failed")
                .expect("IndexedDB not available");

            let open_req = factory
                .open_with_u32(&db_name, 1)
                .expect("open_with_u32 failed");

            let open_req_ref = open_req.clone();
            let on_upgrade = Closure::once(move |_: JsValue| {
                let db: IdbDatabase = open_req_ref
                    .result()
                    .expect("result() in onupgradeneeded")
                    .unchecked_into();
                let store_names = db.object_store_names();
                if !store_names.contains(ASSETS_STORE) {
                    db.create_object_store(ASSETS_STORE)
                        .expect("create assets store");
                }
                if !store_names.contains(META_STORE) {
                    db.create_object_store(META_STORE)
                        .expect("create meta store");
                }
            });
            open_req.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
            on_upgrade.forget();

            let db_val = idb_request(&open_req)
                .await
                .expect("failed to open asset IDB");
            let db: IdbDatabase = db_val.unchecked_into();
            db
        })
        .await;

        Self {
            db,
            db_name: db_name_clone,
        }
    }

    pub fn db_name(&self) -> &str {
        &self.db_name
    }

    fn assets_store(&self, mode: IdbTransactionMode) -> (IdbTransaction, IdbObjectStore) {
        let tx = self
            .db
            .transaction_with_str_and_mode(ASSETS_STORE, mode)
            .expect("transaction on assets");
        let store = tx.object_store(ASSETS_STORE).expect("open assets store");
        (tx, store)
    }

    fn meta_store(&self, mode: IdbTransactionMode) -> (IdbTransaction, IdbObjectStore) {
        let tx = self
            .db
            .transaction_with_str_and_mode(META_STORE, mode)
            .expect("transaction on meta");
        let store = tx.object_store(META_STORE).expect("open meta store");
        (tx, store)
    }

    /// Check if an asset path exists (has a hash entry in the assets store).
    pub async fn exists(&self, path: &str) -> bool {
        let path = path.to_string();
        UnsafeSend(async {
            let (_tx, store) = self.assets_store(IdbTransactionMode::Readonly);
            let key = JsValue::from_str(&path);
            let req = store.get(&key).expect("asset exists request");
            let result = idb_request(&req).await.expect("asset exists failed");
            !result.is_undefined() && !result.is_null()
        })
        .await
    }

    /// Read the version counter. Returns 0 if not set.
    pub async fn version(&self) -> i64 {
        UnsafeSend(async {
            let (_tx, store) = self.meta_store(IdbTransactionMode::Readonly);
            let key = JsValue::from_str("version");
            let req = store.get(&key).expect("version get request");
            let result = idb_request(&req).await.expect("version get failed");

            if result.is_undefined() || result.is_null() {
                0
            } else {
                result.as_f64().expect("IDB version must be a number") as i64
            }
        })
        .await
    }
}
