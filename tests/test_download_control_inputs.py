"""Local integrity/security fixtures only; never contact a release or run fits."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import ssl
import stat
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import warnings
import zipfile

from analysis import download_control_inputs as downloader


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FixtureResponse(io.BytesIO):
    def __init__(self, raw: bytes, url: str = "https://example.invalid/fixture.zip"):
        super().__init__(raw)
        self.url = url

    def geturl(self) -> str:
        return self.url


class ControlInputDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="control-input-fixtures-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.destination = self.root / "installed"
        network = patch.object(downloader.urllib.request, "build_opener",
                               side_effect=AssertionError("Network forbidden in fixture tests"))
        self.network_guard = network.start()
        self.addCleanup(network.stop)

    def make_bundle(self, payloads=None, *, sums=None, seal_updates=None, extra_entries=(), zip64=False):
        payloads = payloads if payloads is not None else [
            ("controls/fixture.npy", b"fixture bytes, not scientific output\x00\xff"),
            ("README.md", b"Local test fixture only.\n")]
        if sums is None:
            sums = "".join(f"{digest(data)}  {name.filename if isinstance(name, zipfile.ZipInfo) else name}\n"
                           for name, data in payloads if not
                           (name.is_dir() if isinstance(name, zipfile.ZipInfo) else name.endswith("/"))).encode("utf-8")
        seal = {"schema_version": 1, "status": "complete", "sha256sums_sha256": digest(sums)}
        seal.update(seal_updates or {})
        path = self.root / "fixture.zip"
        with warnings.catch_warnings(), zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as bundle:
            warnings.simplefilter("ignore", UserWarning)  # Deliberate duplicate-member fixtures.
            for name, data in [*payloads, ("SHA256SUMS", sums),
                               ("package_complete.json", encoded(seal)), *extra_entries]:
                if zip64:
                    info = name if isinstance(name, zipfile.ZipInfo) else zipfile.ZipInfo(name)
                    with bundle.open(info, "w", force_zip64=True) as stream:
                        stream.write(data)
                else:
                    bundle.writestr(name, data)
        return path, digest(path.read_bytes())

    def registry(self, archive_hash, **updates):
        value = {"url": "https://example.invalid/fixture.zip", "sha256": archive_hash,
                 "archive_name": "fixture.zip"}
        value.update(updates)
        path = self.root / "registry.json"
        path.write_bytes(encoded(value))
        return path

    def assert_rejected(self, path, archive_hash, message):
        with self.assertRaisesRegex(ValueError, message):
            downloader.install_archive(path, archive_hash, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertFalse((self.root / "escaped").exists())

    def test_valid_archive_is_extracted_byte_identically(self):
        path, archive_hash = self.make_bundle()
        original = path.read_bytes()
        result = downloader.install_archive(path, archive_hash.upper(), self.destination)
        self.assertEqual(result, self.destination.resolve())
        self.assertEqual((result / "controls/fixture.npy").read_bytes(),
                         b"fixture bytes, not scientific output\x00\xff")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(json.loads((result / "package_complete.json").read_bytes())["status"], "complete")
        self.network_guard.assert_not_called()

    def test_local_archive_cli_never_downloads(self):
        path, archive_hash = self.make_bundle()
        registry = self.registry(archive_hash)
        with contextlib.redirect_stdout(io.StringIO()):
            downloader.main(["--registry", str(registry), "--archive", str(path),
                             "--output-dir", str(self.destination)])
        self.assertTrue((self.destination / "package_complete.json").is_file())
        self.network_guard.assert_not_called()

    def test_zip64_stored_entries_match_public_packaging_format(self):
        path, archive_hash = self.make_bundle(zip64=True)
        downloader.install_archive(path, archive_hash, self.destination)
        self.assertTrue((self.destination / "package_complete.json").is_file())

    def test_download_cli_with_mocked_transport_retains_verified_zip(self):
        path, archive_hash = self.make_bundle()
        registry = self.registry(archive_hash)
        destination = self.root / "downloaded" / "installed"
        with patch.object(downloader.urllib.request, "build_opener") as build, \
                contextlib.redirect_stdout(io.StringIO()):
            build.return_value.open.return_value = FixtureResponse(path.read_bytes())
            downloader.main(["--registry", str(registry), "--output-dir", str(destination)])
        self.assertEqual((destination.parent / "fixture.zip").read_bytes(), path.read_bytes())
        self.assertTrue((destination / "package_complete.json").is_file())

    def test_registry_is_required_even_for_local_archives(self):
        path, _ = self.make_bundle()
        with self.assertRaises(FileNotFoundError):
            downloader.main(["--registry", str(self.root / "absent.json"), "--archive", str(path),
                             "--output-dir", str(self.destination)])
        self.assertFalse(self.destination.exists())
        self.network_guard.assert_not_called()

    def test_registry_fields_are_required_and_digest_is_exact_hex(self):
        registry = self.root / "registry.json"
        for missing in ("url", "sha256", "archive_name"):
            value = {"url": "https://example.invalid/fixture.zip", "sha256": "a" * 64,
                     "archive_name": "fixture.zip"}
            value.pop(missing)
            registry.write_bytes(encoded(value))
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "requires"):
                downloader.load_registry(registry)
        for invalid in (None, 42, "", "g" * 64, "a" * 63, "a" * 65, "a" * 64 + "\n", "sha256:" + "a" * 64):
            with self.subTest(digest=invalid), self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                downloader.load_registry(self.registry(invalid))
        self.assertEqual(downloader.load_registry(self.registry("A" * 64))["sha256"], "a" * 64)

    def test_registry_rejects_unsafe_urls_and_archive_names(self):
        for url in ("http://example.invalid/a.zip", "file:///tmp/a.zip", "https:///a.zip",
                    "https://user:password@example.invalid/a.zip", "https://example.invalid/a.zip#fragment",
                    "https://example.invalid/a.zip\n", "https://example.invalid:bad/a.zip"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                downloader.load_registry(self.registry("a" * 64, url=url))
        for name in ("../a.zip", "folder/a.zip", "C:a.zip", r"folder\a.zip", "/a.zip",
                     "NUL.zip", "a.zip ", "a.zip:stream", "not-a-zip", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                downloader.load_registry(self.registry("a" * 64, archive_name=name))

    def test_registry_rejects_duplicate_keys(self):
        path = self.root / "registry.json"
        path.write_bytes(b'{"url":"https://example.invalid/a.zip","url":"https://example.invalid/b.zip"}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            downloader.load_registry(path)

    def test_outer_hash_tamper_is_rejected_before_extraction(self):
        path, archive_hash = self.make_bundle()
        with path.open("ab") as stream:
            stream.write(b"tamper")
        self.assert_rejected(path, archive_hash, "Archive SHA256 mismatch")

    def test_internal_file_tamper_with_matching_outer_hash_is_rejected(self):
        sums = f"{digest(b'original')}  controls/fixture.npy\n".encode("ascii")
        path, archive_hash = self.make_bundle([("controls/fixture.npy", b"tampered")], sums=sums)
        self.assert_rejected(path, archive_hash, "Internal file checksum mismatch")

    def test_package_seal_tamper_is_rejected(self):
        path, archive_hash = self.make_bundle(seal_updates={"sha256sums_sha256": "b" * 64})
        self.assert_rejected(path, archive_hash, "seal mismatch")
        for changes in ({"status": "ongoing"}, {"schema_version": True}, {"schema_version": 2},
                        {"sha256sums_sha256": "not-hex"}):
            with self.subTest(changes=changes):
                path, archive_hash = self.make_bundle(seal_updates=changes)
                self.assert_rejected(path, archive_hash, "Invalid|64 hexadecimal")

    def test_missing_seal_or_checksums_is_rejected(self):
        for missing in ("SHA256SUMS", "package_complete.json"):
            path = self.root / "fixture.zip"
            with zipfile.ZipFile(path, "w") as bundle:
                bundle.writestr("data.bin", b"fixture")
                if missing == "SHA256SUMS":
                    bundle.writestr("package_complete.json", encoded({"status": "complete"}))
                else:
                    bundle.writestr("SHA256SUMS", b"")
            with self.subTest(missing=missing):
                self.assert_rejected(path, digest(path.read_bytes()), "lacks")

    def test_checksum_inventory_requires_exact_unique_coverage(self):
        valid = f"{digest(b'fixture')}  data.bin\n"
        for sums in (b"", b"not a checksum\n", valid.encode() * 2,
                     (valid + f"{digest(b'x')}  extra.bin\n").encode(),
                     f"{digest(b'fixture')}  ../escaped\n".encode(),
                     f"{digest(b'fixture')}  SHA256SUMS\n".encode(),
                     f"{digest(b'fixture')}  package_complete.json\n".encode()):
            with self.subTest(sums=sums):
                path, archive_hash = self.make_bundle([("data.bin", b"fixture")], sums=sums)
                self.assert_rejected(path, archive_hash, "coverage|cover|Malformed|Duplicate|Unsafe|must not include")

    def test_checksum_paths_cannot_traverse_or_use_windows_aliases(self):
        for name in (r"..\escaped", "C:escaped", "/escaped", "safe/../escaped", "data.bin "):
            with self.subTest(name=name):
                sums = f"{digest(b'fixture')}  {name}\n".encode()
                path, archive_hash = self.make_bundle([("data.bin", b"fixture")], sums=sums)
                self.assert_rejected(path, archive_hash, "Unsafe")
        sums = f"{digest(b'fixture')}  data.bin\n{digest(b'fixture')}  DATA.BIN\n".encode()
        path, archive_hash = self.make_bundle([("data.bin", b"fixture")], sums=sums)
        self.assert_rejected(path, archive_hash, "Duplicate")

    def test_posix_windows_and_backslash_traversal_are_rejected(self):
        names = ("../escaped", "safe/../../escaped", "/escaped", "C:/escaped", "C:escaped",
                 r"..\escaped", r"safe\..\escaped", r"\\server\share\escaped", "//server/share/escaped",
                 "safe//escaped", "safe/./escaped", "./escaped", "safe/file:stream",
                 "safe/file.", "safe/file ", "CON.txt", "LPT1/data.bin")
        for name in names:
            with self.subTest(name=name):
                path, archive_hash = self.make_bundle([(name, b"unsafe fixture")])
                self.assert_rejected(path, archive_hash, "Unsafe")

    def test_duplicate_and_case_aliased_members_are_rejected(self):
        for names in (("data.bin", "data.bin"), ("DATA/a.bin", "data/A.BIN"),
                      ("caf\u00e9.bin", "cafe\u0301.bin")):
            with self.subTest(names=names):
                path, archive_hash = self.make_bundle([(name, b"fixture") for name in names])
                self.assert_rejected(path, archive_hash, "Duplicate")

    def test_symlinks_reparse_points_and_special_files_are_rejected(self):
        for mode, low_flags in ((stat.S_IFLNK | 0o777, 0), (stat.S_IFIFO | 0o600, 0),
                                (stat.S_IFREG | 0o600, 0x400)):
            member = zipfile.ZipInfo("link.bin")
            member.create_system = 3
            member.external_attr = (mode << 16) | low_flags
            with self.subTest(mode=mode, low_flags=low_flags):
                path, archive_hash = self.make_bundle([(member, b"../escaped")])
                self.assert_rejected(path, archive_hash, "symlinks, reparse points, and special files")

    def test_file_directory_and_case_aliased_prefix_conflicts_are_rejected(self):
        for payloads in ([('controls', b'file'), ('controls/data.bin', b'child')],
                         [('CONTROLS', b'file'), ('controls/data.bin', b'child')],
                         [('controls/', b''), ('controls', b'file')]):
            with self.subTest(payloads=payloads):
                path, archive_hash = self.make_bundle(payloads)
                self.assert_rejected(path, archive_hash, "prefix conflict|Duplicate")

    def test_explicit_directory_entries_are_supported(self):
        path, archive_hash = self.make_bundle([("controls/", b""), ("controls/data.bin", b"fixture")])
        downloader.install_archive(path, archive_hash, self.destination)
        self.assertEqual((self.destination / "controls/data.bin").read_bytes(), b"fixture")

    def test_existing_destinations_are_untouched(self):
        path, archive_hash = self.make_bundle()
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            downloader.install_archive(path, archive_hash, self.destination)
        sentinel = self.destination / "keep.txt"
        sentinel.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            downloader.install_archive(path, archive_hash, self.destination)
        self.assertEqual(sentinel.read_bytes(), b"preserve")
        with self.assertRaises(FileExistsError):
            downloader.install_archive(path, archive_hash, path)
        self.network_guard.assert_not_called()

    def test_extraction_failure_retains_files_without_completion_seal(self):
        path, archive_hash = self.make_bundle()
        original_archive = path.read_bytes()
        original_open = Path.open

        def interrupted_open(target, mode="r", *args, **kwargs):
            if target == self.destination / "controls/fixture.npy" and mode == "xb":
                raise OSError("fixture disk failure")
            return original_open(target, mode, *args, **kwargs)

        with patch.object(Path, "open", interrupted_open), self.assertRaisesRegex(OSError, "fixture disk failure"):
            downloader.install_archive(path, archive_hash, self.destination)
        self.assertTrue((self.destination / "README.md").is_file())
        self.assertTrue((self.destination / "SHA256SUMS").is_file())
        self.assertFalse((self.destination / "package_complete.json").exists())
        self.assertEqual(path.read_bytes(), original_archive)
        with self.assertRaises(FileExistsError):
            downloader.install_archive(path, archive_hash, self.destination)

    def test_existing_destination_link_is_not_followed(self):
        path, archive_hash = self.make_bundle()
        try:
            self.destination.symlink_to(self.root / "absent-target", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("OS does not permit fixture symlinks")
        with self.assertRaises(FileExistsError):
            downloader.install_archive(path, archive_hash, self.destination)
        self.assertFalse((self.root / "absent-target").exists())

    def test_default_paths_and_no_force_switch(self):
        args = downloader.parse_args([])
        self.assertEqual(args.registry, downloader.ANALYSIS_ROOT / "CONTROL_INPUT_BUNDLE.json")
        self.assertEqual(args.output_dir, downloader.DATA_DIR / "deep_control_inputs")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            downloader.parse_args(["--force"])

    def test_download_uses_verified_tls_and_retains_zip(self):
        target = self.root / "download.zip"
        response = FixtureResponse(b"fixture transport bytes")
        with patch.object(downloader.urllib.request, "build_opener") as build:
            build.return_value.open.return_value = response
            downloader.download_archive("https://example.invalid/fixture.zip", target)
        handlers = build.call_args.args
        handler = next(h for h in handlers if isinstance(h, urllib.request.HTTPSHandler))
        self.assertEqual(handler._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(handler._context.check_hostname)
        self.assertTrue(any(isinstance(h, downloader.HTTPSOnlyRedirect) for h in handlers))
        self.assertEqual(target.read_bytes(), b"fixture transport bytes")

    def test_existing_download_is_not_overwritten_or_fetched(self):
        target = self.root / "download.zip"
        target.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            downloader.download_archive("https://example.invalid/fixture.zip", target)
        self.assertEqual(target.read_bytes(), b"preserve")
        self.network_guard.assert_not_called()

    def test_redirects_cannot_downgrade_https(self):
        handler = downloader.HTTPSOnlyRedirect()
        request = urllib.request.Request("https://example.invalid/fixture.zip")
        for url in ("http://example.invalid/a.zip", "file:///tmp/a.zip", "ftp://example.invalid/a.zip"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, "Found", {}, url)
        redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://example.invalid/asset.zip")
        self.assertEqual(redirected.full_url, "https://example.invalid/asset.zip")

    def test_unexpected_non_https_response_is_rejected(self):
        target = self.root / "download.zip"
        with patch.object(downloader.urllib.request, "build_opener") as build:
            build.return_value.open.return_value = FixtureResponse(b"fixture", "http://example.invalid/unsafe.zip")
            with self.assertRaises(ValueError):
                downloader.download_archive("https://example.invalid/fixture.zip", target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
