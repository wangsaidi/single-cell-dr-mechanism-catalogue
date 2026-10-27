"""Synthetic byte-integrity checks; no private inputs, network, fits or scoring."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile

from analysis import download_control_inputs as bundle
from analysis.scripts import score_deep_configuration_controls as scorer


CODE_ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(shutil.which("git"), "Git is required for actual attribute/filter checks")
class AttributeTests(unittest.TestCase):
    def repository(self, root):
        (root / ".gitattributes").write_bytes((CODE_ROOT / ".gitattributes").read_bytes())
        subprocess.run(["git", "init", "--quiet", "--template="], cwd=root,
                       check=True, capture_output=True, timeout=20)

    def test_only_new_hash_locked_sources_and_outputs_disable_text_conversion(self):
        locked = ["data/source_data/deep_controls/fit/history.csv",
                  "data/source_data/reference_attribution/design.json",
                  "data/source_data/control_plot_data_inventory.json",
                  "data/source_data/control_plot_data_schema.json",
                  *[f"outputs/supplementary_figures/Supplementary_Fig_S{n}_fixture.svg"
                    for n in (18, 19, 20)]]
        established = ["README.md", "FILE_MANIFEST.csv", "analysis/ARCHIVE_SHA256SUMS.txt",
                       "data/source_data/generated/fixture.csv", "data/source_data/calibration/fixture.csv",
                       "outputs/supplementary_figures/Supplementary_Fig_S17_fixture.svg"]
        names = locked + established
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.repository(root)
            result = subprocess.run(["git", "check-attr", "-z", "--stdin", "text"],
                                    input="\0".join(names) + "\0", cwd=root,
                                    capture_output=True, text=True, check=True, timeout=20)
        values = result.stdout.rstrip("\0").split("\0")
        attributes = {values[i]: values[i + 2] for i in range(0, len(values), 3)}
        self.assertEqual(attributes, {**dict.fromkeys(locked, "unset"), **dict.fromkeys(established, "auto")})

    def test_windows_and_linux_git_filters_preserve_new_source_bytes(self):
        payload = b"name,value\r\nfixture,0.125\r\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.repository(root)
            raw = subprocess.run(["git", "hash-object", "--no-filters", "--stdin"], input=payload,
                                 cwd=root, capture_output=True, check=True, timeout=20).stdout
            canonical_old = subprocess.run(["git", "hash-object", "--no-filters", "--stdin"],
                                           input=payload.replace(b"\r\n", b"\n"), cwd=root,
                                           capture_output=True, check=True, timeout=20).stdout
            for autocrlf in ("true", "false"):
                for name in ("data/source_data/deep_controls/fixture.csv",
                             "data/source_data/reference_attribution/fixture.csv"):
                    with self.subTest(autocrlf=autocrlf, name=name):
                        filtered = subprocess.run(
                            ["git", "-c", f"core.autocrlf={autocrlf}", "hash-object", f"--path={name}", "--stdin"],
                            input=payload, cwd=root, capture_output=True, check=True, timeout=20).stdout
                        self.assertEqual(filtered, raw)
                old = subprocess.run(["git", "-c", f"core.autocrlf={autocrlf}", "hash-object",
                                      "--path=data/source_data/generated/fixture.csv", "--stdin"],
                                     input=payload, cwd=root, capture_output=True, check=True, timeout=20).stdout
                self.assertEqual(old, canonical_old)


class BundleIntegrityTests(unittest.TestCase):
    def archive(self, root, payload=None, *, corrupt_payload=False):
        payload = payload if payload is not None else {"controls/fixture.csv": b"id,value\r\na,1\r\n"}
        checksums = "".join(f"{digest(value)}  {name}\n" for name, value in sorted(payload.items())).encode("utf-8")
        seal = json.dumps({"schema_version": 1, "status": "complete",
                           "sha256sums_sha256": digest(checksums)}).encode("ascii")
        path = root / "fixture.zip"
        with zipfile.ZipFile(path, "w") as stream:
            for name, data in payload.items():
                stream.writestr(name, data + (b"corrupt" if corrupt_payload else b""))
            stream.writestr("SHA256SUMS", checksums)
            stream.writestr("package_complete.json", seal)
        return path

    def test_install_preserves_payload_bytes_and_original_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root)
            original = archive.read_bytes()
            destination = root / "installed"
            self.assertEqual(bundle.install_archive(archive, digest(original), destination), destination)
            self.assertEqual((destination / "controls/fixture.csv").read_bytes(), b"id,value\r\na,1\r\n")
            self.assertEqual(archive.read_bytes(), original)
            self.assertEqual(json.loads((destination / "package_complete.json").read_text())["status"], "complete")

    def test_wrong_archive_hash_creates_no_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root)
            with self.assertRaisesRegex(ValueError, "Archive SHA256 mismatch"):
                bundle.install_archive(archive, "0" * 64, root / "installed")
            self.assertFalse((root / "installed").exists())

    def test_internal_hash_mismatch_creates_no_destination_or_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, corrupt_payload=True)
            with self.assertRaisesRegex(ValueError, "Internal file checksum mismatch"):
                bundle.install_archive(archive, digest(archive.read_bytes()), root / "installed")
            self.assertFalse((root / "installed").exists())

    def test_case_aliased_or_traversal_archive_is_rejected_before_extraction(self):
        payloads = [{"controls/A.csv": b"a", "controls/a.csv": b"b"}, {"../escape.csv": b"a"}]
        for payload in payloads:
            with self.subTest(names=list(payload)), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = self.archive(root, payload)
                with self.assertRaises(ValueError):
                    bundle.install_archive(archive, digest(archive.read_bytes()), root / "installed")
                self.assertFalse((root / "installed").exists())
                self.assertFalse((root / "escape.csv").exists())

    def test_existing_destination_is_not_overwritten_or_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root)
            destination = root / "installed"
            destination.mkdir()
            sentinel = destination / "keep.txt"
            sentinel.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                bundle.install_archive(archive, digest(archive.read_bytes()), destination)
            self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_registry_does_not_guess_urls_and_rejects_insecure_or_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry.json"
            with self.assertRaises(FileNotFoundError):
                bundle.load_registry(registry)
            registry.write_text('{"url":"https://example.invalid/a.zip","url":"https://example.invalid/b.zip"}', encoding="ascii")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                bundle.load_registry(registry)
        for url in ("http://example.invalid/a.zip", "https://user:pass@example.invalid/a.zip"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                bundle.https_url(url)


class PortableInputHashTests(unittest.TestCase):
    def metadata(self, original, portable):
        counts = digest(b"synthetic count fixture")
        return {"input_file_sha256": digest(original), "portable_input_file_sha256": digest(portable),
                "original_counts_file_sha256": counts,
                "input_serialization": {"source_npz_sha256": digest(original), "portable_npz_sha256": digest(portable),
                    "counts_hvg_sha256": counts, "numeric_X_npy_byte_identical": True,
                    "object_id_payloads_deserialized": False, "training_inputs_modified": False, "allow_pickle": False}}

    def test_original_training_hash_is_preserved_and_portable_file_hash_is_checked(self):
        original, portable = b"original serialization fixture", b"portable serialization fixture"
        meta = self.metadata(original, portable)
        before = copy.deepcopy(meta)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.npz"
            path.write_bytes(portable)
            self.assertEqual(scorer.validate_control_input_hash(path, meta), digest(portable))
            self.assertEqual(meta, before)
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                scorer.validate_control_input_hash(path, meta)

    def test_missing_or_unsafe_derivative_provenance_is_rejected(self):
        original, portable = b"original fixture", b"portable fixture"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.npz"
            path.write_bytes(portable)
            for key, value in (("training_inputs_modified", True), ("object_id_payloads_deserialized", True),
                               ("allow_pickle", True), ("numeric_X_npy_byte_identical", False)):
                meta = self.metadata(original, portable)
                meta["input_serialization"][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "provenance"):
                    scorer.validate_control_input_hash(path, meta)


if __name__ == "__main__":
    unittest.main()
