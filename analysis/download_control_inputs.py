"""Download a registry-pinned control bundle into a fresh directory.

The parent-authored analysis/CONTROL_INPUT_BUNDLE.json must supply url (HTTPS),
sha256 (exactly 64 hexadecimal characters), and archive_name (a ZIP basename).
No release URL or digest is guessed when that registry is absent.

Run from the public code root:
    python -B -m analysis.download_control_inputs
    python -B -m analysis.download_control_inputs --archive /path/to/bundle.zip

The default destination is analysis.paths.DATA_DIR/deep_control_inputs. Downloads
are retained alongside the destination using archive_name; an existing ZIP is
never overwritten (use --archive to verify/reuse it). No files are deleted, even
on failure. Failed extraction may leave an incomplete new directory, but its
package_complete.json is written only after all checks succeed. Choose a fresh
destination for another attempt; there is no force or resume mode.

Only the standard library and the local standard-library-only paths module are
used. This verifies transport/file integrity, not fit completion or convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import ssl
import stat
import sys
from typing import BinaryIO
import unicodedata
import urllib.parse
import urllib.request
import zipfile

from analysis.paths import ANALYSIS_ROOT, DATA_DIR


REGISTRY_PATH = ANALYSIS_ROOT / "CONTROL_INPUT_BUNDLE.json"
DEFAULT_DESTINATION = DATA_DIR / "deep_control_inputs"
HASH_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
WINDOWS_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9\u00b9\u00b2\u00b3]|LPT[1-9\u00b9\u00b2\u00b3])\Z", re.IGNORECASE)
SEAL_NAME = "package_complete.json"
SUMS_NAME = "SHA256SUMS"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest_value(value: object) -> str:
    require(isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None,
            "SHA256 must be exactly 64 hexadecimal characters, without prefixes or whitespace.")
    return value.lower()


def stream_digest(stream: BinaryIO) -> str:
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def json_object(raw: bytes) -> dict:
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
    require(isinstance(value, dict), "Expected a JSON object.")
    return value


def https_url(value: object) -> str:
    require(isinstance(value, str) and bool(value) and not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value),
            "Registry/redirect URL must be a nonempty HTTPS URL without whitespace.")
    parsed = urllib.parse.urlsplit(value)
    require(parsed.scheme == "https" and bool(parsed.hostname) and parsed.username is None and
            parsed.password is None and not parsed.fragment,
            "Only HTTPS URLs without credentials or fragments are allowed.")
    # Accessing port also validates malformed/non-numeric/out-of-range ports.
    _ = parsed.port
    return value


def relative_name(name: str, *, directory: bool = False) -> tuple[str, ...]:
    require(isinstance(name, str) and bool(name) and "\\" not in name and
            not name.startswith("/") and not PureWindowsPath(name).drive and
            not any(ord(c) < 32 or ord(c) == 127 for c in name),
            f"Unsafe archive/checksum path: {name!r}")
    body = name[:-1] if directory and name.endswith("/") else name
    parts = tuple(body.split("/"))
    for part in parts:
        require(bool(part) and part not in (".", "..") and part == part.rstrip(" .") and
                not any(c in '<>:"|?*' for c in part) and
                WINDOWS_DEVICE.fullmatch(part.split(".", 1)[0]) is None,
                f"Unsafe archive/checksum path: {name!r}")
    return parts


def path_key(parts: tuple[str, ...]) -> str:
    return "/".join(unicodedata.normalize("NFC", part).casefold() for part in parts)


def load_registry(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError("Control bundle registry is missing. The parent must publish CONTROL_INPUT_BUNDLE.json with the actual URL and ZIP hash.")
    value = json_object(path.read_bytes())
    require(all(key in value for key in ("url", "sha256", "archive_name")),
            "Registry requires url, sha256, and archive_name.")
    url, digest = https_url(value["url"]), digest_value(value["sha256"])
    name = value["archive_name"]
    require(isinstance(name, str) and len(relative_name(name)) == 1 and name.lower().endswith(".zip"),
            "archive_name must be a safe ZIP basename, not a path.")
    return {"url": url, "sha256": digest, "archive_name": name}


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_archive(url: str, target: Path) -> None:
    https_url(url)
    if os.path.lexists(target):
        raise FileExistsError("Download cache already exists; use --archive to verify it or select another destination parent.")
    context = ssl.create_default_context()
    opener = urllib.request.build_opener(HTTPSOnlyRedirect(), urllib.request.HTTPSHandler(context=context))
    request = urllib.request.Request(url, headers={"User-Agent": "single-cell-dr-control-inputs"})
    with opener.open(request, timeout=60) as response:
        https_url(response.geturl())
        with target.open("xb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)


def fresh_destination(path: Path) -> Path:
    path = Path(path).expanduser()
    if os.path.lexists(path):
        raise FileExistsError("Destination already exists, including empty directories or links. Choose a new directory; nothing is overwritten or deleted.")
    resolved = path.resolve()
    if os.path.lexists(resolved):
        raise FileExistsError("Resolved destination already exists. Choose a new directory.")
    return resolved


def contained_path(root: Path, parts: tuple[str, ...]) -> Path:
    target = root.joinpath(*parts)
    resolved = target.resolve()
    require(root in resolved.parents and resolved != root,
            "Archive member resolves outside the destination.")
    require(not root.is_symlink() and root.resolve() == root, "Destination changed into a link.")
    return target


def inspect_members(bundle: zipfile.ZipFile, root: Path) -> tuple[dict[str, zipfile.ZipInfo], list[tuple[str, ...]]]:
    files, directories, seen = {}, [], {}
    for member in bundle.infolist():
        require(member.orig_filename == member.filename, "Unsafe NUL-truncated ZIP member.")
        parts = relative_name(member.filename, directory=member.is_dir())
        key = path_key(parts)
        require(key not in seen, f"Duplicate/Windows-aliased ZIP member: {member.filename!r}")
        seen[key] = member.is_dir()
        require(not (member.flag_bits & 1), "Encrypted ZIP members are not supported.")
        mode = member.external_attr >> 16
        kind = stat.S_IFMT(mode)
        require(kind in (0, stat.S_IFREG, stat.S_IFDIR) and not (member.external_attr & 0x400),
                "ZIP symlinks, reparse points, and special files are forbidden.")
        require((kind != stat.S_IFDIR or member.is_dir()) and
                (kind != stat.S_IFREG or not member.is_dir()) and
                (not (member.external_attr & 0x10) or member.is_dir()),
                "ZIP file/directory attributes disagree with the member name.")
        contained_path(root, parts)
        if member.is_dir():
            directories.append(parts)
        else:
            files[member.filename] = member
    for key in seen:
        parts = key.split("/")
        for count in range(1, len(parts)):
            require(seen.get("/".join(parts[:count])) is not False,
                    "ZIP file/directory prefix conflict.")
    require(SUMS_NAME in files and SEAL_NAME in files, "Bundle lacks SHA256SUMS or package_complete.json.")
    return files, directories


def verify_internal(bundle: zipfile.ZipFile, files: dict[str, zipfile.ZipInfo]) -> tuple[dict[str, str], bytes]:
    seal_bytes = bundle.read(files[SEAL_NAME])
    seal = json_object(seal_bytes)
    require(type(seal.get("schema_version")) is int and seal["schema_version"] == 1 and
            seal.get("status") == "complete", "Invalid or incomplete package seal.")
    sums_bytes = bundle.read(files[SUMS_NAME])
    require(hashlib.sha256(sums_bytes).hexdigest() == digest_value(seal.get("sha256sums_sha256")),
            "SHA256SUMS seal mismatch.")
    checksums, seen = {}, set()
    for line in sums_bytes.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
        require(match is not None, "Malformed SHA256SUMS line.")
        digest, name = match.groups()
        key = path_key(relative_name(name))
        require(key not in seen, "Duplicate/Windows-aliased checksum entry.")
        require(name not in (SUMS_NAME, SEAL_NAME), "Checksums must not include the checksum file or its seal.")
        checksums[name] = digest.lower()
        seen.add(key)
    require(set(checksums) == set(files) - {SUMS_NAME, SEAL_NAME},
            "Checksum inventory does not exactly cover all bundle payload files.")
    for name, expected in sorted(checksums.items()):
        with bundle.open(files[name]) as stream:
            require(stream_digest(stream) == expected, f"Internal file checksum mismatch: {name}")
    checksums[SUMS_NAME] = hashlib.sha256(sums_bytes).hexdigest()
    return checksums, seal_bytes


def install_archive(archive: Path, expected_sha256: str, destination: Path) -> Path:
    root = fresh_destination(destination)
    expected_sha256 = digest_value(expected_sha256)
    # Hash and read the very same open archive, not a path reopened after verification.
    with Path(archive).open("rb") as archive_stream:
        require(stream_digest(archive_stream) == expected_sha256, "Archive SHA256 mismatch; nothing extracted.")
        archive_stream.seek(0)
        with zipfile.ZipFile(archive_stream) as bundle:
            files, directories = inspect_members(bundle, root)
            checksums, seal_bytes = verify_internal(bundle, files)
            archive_stream.seek(0)
            require(stream_digest(archive_stream) == expected_sha256, "Archive changed during preflight.")
            root.parent.mkdir(parents=True, exist_ok=True)
            root.mkdir(exist_ok=False)
            for parts in sorted(directories):
                contained_path(root, parts).mkdir(parents=True, exist_ok=True)
            for name, expected in sorted(checksums.items()):
                target = contained_path(root, relative_name(name))
                target.parent.mkdir(parents=True, exist_ok=True)
                # Recheck containment after parent creation; no ZIP modes/links are applied.
                target = contained_path(root, relative_name(name))
                digest = hashlib.sha256()
                with bundle.open(files[name]) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                require(digest.hexdigest() == expected, f"File changed during extraction: {name}")
            archive_stream.seek(0)
            require(stream_digest(archive_stream) == expected_sha256, "Archive changed during extraction; new directory is incomplete.")
            marker = contained_path(root, (SEAL_NAME,))
            with marker.open("xb") as output:
                output.write(seal_bytes)
    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH,
                        help="Parent-authored registry with the actual HTTPS URL, ZIP hash and name.")
    parser.add_argument("--archive", type=Path, help="Verify/extract a local ZIP; no network request.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DESTINATION,
                        help="New nonexistent directory; defaults to analysis/data/deep_control_inputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    destination = fresh_destination(args.output_dir)
    registry = load_registry(args.registry)
    archive = args.archive
    if archive is None:
        archive = destination.parent / registry["archive_name"]
        require(archive != destination, "Destination conflicts with the retained download ZIP filename.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {registry['archive_name']} with HTTPS certificate verification.")
        download_archive(registry["url"], archive)
        print(f"ZIP retained at {archive}")
    root = install_archive(archive, registry["sha256"], destination)
    print(f"Verified archive, internal checksums and package seal; extracted to {root}")
    print("File integrity is not verification of scientific convergence or workflow completion.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as exc:
        print(f"Control input installation stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
