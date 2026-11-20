"""Official Android protection levels for Dada's permission features (roadmap step 2.3).

Source of truth: the ``android:protectionLevel`` of each ``<permission>`` element in AOSP's
framework manifest, ``frameworks/base/core/res/AndroidManifest.xml`` on the ``main`` branch.
The public ``Manifest.permission`` reference pages are generated from this file. Levels were
read on the date in ``RETRIEVED``. For transparency, the level in an early AOSP revision
(``c231fd0``) is recorded too: many permissions that are "normal" today were "dangerous" in
early Android, which is reported as a threat to validity.

Only these rules decide what MR5 and MR6 may add:

* MR5 may add only permissions whose current base level is ``normal`` and that are neither
  deprecated nor removed.
* MR6 may add only permissions whose current base level is ``dangerous`` and that are not removed.
* Anything else (signature-level, privileged, app-op, or not defined in the framework) is
  excluded from both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Base = Literal["normal", "dangerous", "signature", "undefined"]

PREFIX = "android.permission."
SOURCE_URL = (
    "https://android.googlesource.com/platform/frameworks/base/+/main/core/res/AndroidManifest.xml"
)
EARLY_SOURCE_URL = (
    "https://android.googlesource.com/platform/frameworks/base/+/c231fd0/core/res/"
    "AndroidManifest.xml"
)
RETRIEVED = "2026-11-21"
REFERENCE_URL = "https://developer.android.com/reference/android/Manifest.permission#{name}"


@dataclass(frozen=True)
class PermissionLevel:
    name: str  # without the "android.permission." prefix
    level: str  # exact protectionLevel string in AOSP main ("" if not defined)
    early_level: str  # protectionLevel in AOSP c231fd0 ("" if not defined there)
    deprecated: bool = False
    removed: bool = False
    note: str = ""

    @property
    def base(self) -> Base:
        if not self.level:
            return "undefined"
        first = self.level.split("|")[0]
        if first in ("normal", "dangerous"):
            return first  # type: ignore[return-value]
        return "signature"

    @property
    def column(self) -> str:
        return PREFIX + self.name

    @property
    def mr5_eligible(self) -> bool:
        return self.base == "normal" and not self.deprecated and not self.removed

    @property
    def mr6_eligible(self) -> bool:
        return self.base == "dangerous" and not self.removed

    @property
    def reference(self) -> str:
        return REFERENCE_URL.format(name=self.name)


# The 34 android.permission.* features in Dada v3.1, in Dada's column order.
LEVELS: tuple[PermissionLevel, ...] = (
    PermissionLevel("READ_PHONE_STATE", "dangerous", "dangerous"),
    PermissionLevel("READ_CONTACTS", "dangerous", "dangerous"),
    PermissionLevel("READ_SMS", "dangerous", "dangerous"),
    PermissionLevel("CAMERA", "dangerous|instant", "dangerous"),
    PermissionLevel("RECORD_AUDIO", "dangerous|instant", "dangerous"),
    PermissionLevel(
        "READ_EXTERNAL_STORAGE",
        "dangerous",
        "",
        note="no effect for apps targeting API 33+ (not marked deprecated)",
    ),
    PermissionLevel(
        "READ_HISTORY_BOOKMARKS",
        "",
        "dangerous",
        removed=True,
        note="not defined as android.permission in AOSP main; only "
        "com.android.browser.permission.READ_HISTORY_BOOKMARKS remains, flagged removed",
    ),
    PermissionLevel("ACCESS_NETWORK_STATE", "normal|instant", "normal"),
    PermissionLevel("ACCESS_WIFI_STATE", "normal", "normal"),
    PermissionLevel("GET_TASKS", "normal", "dangerous", deprecated=True, note="no longer enforced"),
    PermissionLevel("ACTIVITY_RECOGNITION", "dangerous|instant", ""),
    PermissionLevel("INTERNET", "normal|instant", "dangerous"),
    PermissionLevel("SEND_SMS", "dangerous", "dangerous"),
    PermissionLevel("CALL_PHONE", "dangerous", "dangerous"),
    PermissionLevel("READ_CALL_LOG", "dangerous", ""),
    PermissionLevel("BLUETOOTH_ADMIN", "normal", "dangerous"),
    PermissionLevel("BLUETOOTH", "normal", "dangerous"),
    PermissionLevel("BODY_SENSORS", "dangerous", ""),
    PermissionLevel("GET_ACCOUNTS", "dangerous", "normal"),
    PermissionLevel(
        "WRITE_EXTERNAL_STORAGE",
        "dangerous",
        "dangerous",
        note="no effect for apps targeting API 30+ (not marked deprecated)",
    ),
    PermissionLevel("NFC", "normal", ""),
    PermissionLevel("WRITE_CONTACTS", "dangerous", "dangerous"),
    PermissionLevel(
        "WRITE_SMS", "normal", "dangerous", removed=True, note="permissionFlags=removed; hidden"
    ),
    PermissionLevel("MOUNT_FORMAT_FILESYSTEMS", "signature|privileged", "dangerous"),
    PermissionLevel("CHANGE_NETWORK_STATE", "normal", "dangerous"),
    PermissionLevel("CHANGE_WIFI_STATE", "normal", "dangerous"),
    PermissionLevel("REORDER_TASKS", "normal", "dangerous"),
    PermissionLevel("WAKE_LOCK", "normal|instant", "dangerous"),
    PermissionLevel("REBOOT", "signature|privileged", "signature"),
    PermissionLevel("KILL_BACKGROUND_PROCESSES", "normal", ""),
    PermissionLevel("INSTALL_PACKAGES", "signature|privileged", "signatureOrSystem"),
    PermissionLevel("REQUEST_INSTALL_PACKAGES", "signature|appop", ""),
    PermissionLevel("INJECT_EVENTS", "signature", "signature"),
    PermissionLevel(
        "SYSTEM_ALERT_WINDOW", "signature|setup|appop|installer|pre23|development", "dangerous"
    ),
)

BY_COLUMN = {p.column: p for p in LEVELS}


def mr5_columns() -> list[str]:
    return [p.column for p in LEVELS if p.mr5_eligible]


def mr6_columns() -> list[str]:
    return [p.column for p in LEVELS if p.mr6_eligible]


def _uses(p: PermissionLevel) -> str:
    if p.mr5_eligible:
        return "MR5"
    if p.mr6_eligible:
        return "MR6"
    return "excluded"


def render_markdown() -> str:
    lines = [
        "# Permission protection levels (roadmap step 2.3)",
        "",
        f"Levels read from AOSP's framework manifest, `main` branch, on {RETRIEVED}: "
        f"[core/res/AndroidManifest.xml]({SOURCE_URL}). The official `Manifest.permission` "
        "reference is generated from this file; each row links to its reference entry. "
        f"Early level: AOSP revision [`c231fd0`]({EARLY_SOURCE_URL}), early Android. Generated "
        "from `shortcutcheck.mrs.permissions`; do not edit by hand.",
        "",
        "Rules: MR5 adds only current `normal` permissions that are not deprecated or removed. "
        "MR6 adds only current `dangerous` permissions that are not removed. Everything else is "
        "excluded.",
        "",
        "| Feature | Protection level (AOSP main) | Early level (c231fd0) | Notes | Used by "
        "| Reference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for p in LEVELS:
        notes = "; ".join(
            n
            for n in (
                "deprecated" if p.deprecated else "",
                "removed" if p.removed else "",
                p.note,
            )
            if n
        )
        lines.append(
            f"| `{p.name}` | {p.level or 'not defined'} | {p.early_level or 'not defined'} "
            f"| {notes or '–'} | {_uses(p)} | [docs]({p.reference}) |"
        )
    n5, n6 = len(mr5_columns()), len(mr6_columns())
    lines += [
        "",
        f"**{n5} permissions for MR5, {n6} for MR6, {len(LEVELS) - n5 - n6} excluded.** "
        "At fit time each MR also drops permissions held by fewer than 10 training apps, and "
        "MR5 keeps only those more common in training goodware than training malware.",
    ]
    return "\n".join(lines) + "\n"
