# Permission protection levels (roadmap step 2.3)

Levels read from AOSP's framework manifest, `main` branch, on 2026-11-21: [core/res/AndroidManifest.xml](https://android.googlesource.com/platform/frameworks/base/+/main/core/res/AndroidManifest.xml). The official `Manifest.permission` reference is generated from this file; each row links to its reference entry. Early level: AOSP revision [`c231fd0`](https://android.googlesource.com/platform/frameworks/base/+/c231fd0/core/res/AndroidManifest.xml), early Android. Generated from `shortcutcheck.mrs.permissions`; do not edit by hand.

Rules: MR5 adds only current `normal` permissions that are not deprecated or removed. MR6 adds only current `dangerous` permissions that are not removed. Everything else is excluded.

| Feature | Protection level (AOSP main) | Early level (c231fd0) | Notes | Used by | Reference |
| --- | --- | --- | --- | --- | --- |
| `READ_PHONE_STATE` | dangerous | dangerous | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#READ_PHONE_STATE) |
| `READ_CONTACTS` | dangerous | dangerous | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#READ_CONTACTS) |
| `READ_SMS` | dangerous | dangerous | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#READ_SMS) |
| `CAMERA` | dangerous\|instant | dangerous | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#CAMERA) |
| `RECORD_AUDIO` | dangerous\|instant | dangerous | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#RECORD_AUDIO) |
| `READ_EXTERNAL_STORAGE` | dangerous | not defined | no effect for apps targeting API 33+ (not marked deprecated) | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#READ_EXTERNAL_STORAGE) |
| `READ_HISTORY_BOOKMARKS` | not defined | dangerous | removed; not defined as android.permission in AOSP main; only com.android.browser.permission.READ_HISTORY_BOOKMARKS remains, flagged removed | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#READ_HISTORY_BOOKMARKS) |
| `ACCESS_NETWORK_STATE` | normal\|instant | normal | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#ACCESS_NETWORK_STATE) |
| `ACCESS_WIFI_STATE` | normal | normal | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#ACCESS_WIFI_STATE) |
| `GET_TASKS` | normal | dangerous | deprecated; no longer enforced | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#GET_TASKS) |
| `ACTIVITY_RECOGNITION` | dangerous\|instant | not defined | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#ACTIVITY_RECOGNITION) |
| `INTERNET` | normal\|instant | dangerous | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#INTERNET) |
| `SEND_SMS` | dangerous | dangerous | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#SEND_SMS) |
| `CALL_PHONE` | dangerous | dangerous | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#CALL_PHONE) |
| `READ_CALL_LOG` | dangerous | not defined | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#READ_CALL_LOG) |
| `BLUETOOTH_ADMIN` | normal | dangerous | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#BLUETOOTH_ADMIN) |
| `BLUETOOTH` | normal | dangerous | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#BLUETOOTH) |
| `BODY_SENSORS` | dangerous | not defined | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#BODY_SENSORS) |
| `GET_ACCOUNTS` | dangerous | normal | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#GET_ACCOUNTS) |
| `WRITE_EXTERNAL_STORAGE` | dangerous | dangerous | no effect for apps targeting API 30+ (not marked deprecated) | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#WRITE_EXTERNAL_STORAGE) |
| `NFC` | normal | not defined | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#NFC) |
| `WRITE_CONTACTS` | dangerous | dangerous | – | MR6 | [docs](https://developer.android.com/reference/android/Manifest.permission#WRITE_CONTACTS) |
| `WRITE_SMS` | normal | dangerous | removed; permissionFlags=removed; hidden | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#WRITE_SMS) |
| `MOUNT_FORMAT_FILESYSTEMS` | signature\|privileged | dangerous | – | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#MOUNT_FORMAT_FILESYSTEMS) |
| `CHANGE_NETWORK_STATE` | normal | dangerous | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#CHANGE_NETWORK_STATE) |
| `CHANGE_WIFI_STATE` | normal | dangerous | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#CHANGE_WIFI_STATE) |
| `REORDER_TASKS` | normal | dangerous | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#REORDER_TASKS) |
| `WAKE_LOCK` | normal\|instant | dangerous | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#WAKE_LOCK) |
| `REBOOT` | signature\|privileged | signature | – | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#REBOOT) |
| `KILL_BACKGROUND_PROCESSES` | normal | not defined | – | MR5 | [docs](https://developer.android.com/reference/android/Manifest.permission#KILL_BACKGROUND_PROCESSES) |
| `INSTALL_PACKAGES` | signature\|privileged | signatureOrSystem | – | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#INSTALL_PACKAGES) |
| `REQUEST_INSTALL_PACKAGES` | signature\|appop | not defined | – | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#REQUEST_INSTALL_PACKAGES) |
| `INJECT_EVENTS` | signature | signature | – | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#INJECT_EVENTS) |
| `SYSTEM_ALERT_WINDOW` | signature\|setup\|appop\|installer\|pre23\|development | dangerous | – | excluded | [docs](https://developer.android.com/reference/android/Manifest.permission#SYSTEM_ALERT_WINDOW) |

**11 permissions for MR5, 14 for MR6, 9 excluded.** At fit time each MR also drops permissions held by fewer than 10 training apps, and MR5 keeps only those more common in training goodware than training malware.
