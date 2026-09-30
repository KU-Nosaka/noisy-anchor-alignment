"""Small raw-byte and cohort tests; no face data or downloaded checkpoint required."""
from pathlib import Path
import hashlib
import gc
import io as bytes_io
import json
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

import celeba_inputs as celeba
import celeba_public_svd_identity100_runner as base
import notebook_api as api
import vgg_context
import vgg_io as vgg
import vgg_master


class Reporter:
    def emit(self, *args, **kwargs):
        pass


def metadata(root, txt=True):
    root.mkdir(parents=True, exist_ok=True)
    if txt:
        (root / "list_attr_celeba.txt").write_text(
            "4\nSmiling Male\n000001.jpg 1 -1\n000002.jpg -1 -1\n"
            "000003.jpg -1 1\n000004.jpg 1 1\n")
    else:
        (root / "list_attr_celeba.csv").write_text(
            "image_id,Smiling,Male\n000001.jpg,1,-1\n000002.jpg,-1,-1\n"
            "000003.jpg,-1,1\n000004.jpg,1,1\n")
    (root / "identity_CelebA.txt").write_text(
        "000001.jpg 2\n000002.jpg 2\n000003.jpg 5\n000004.jpg 5\n")


class RawPreparationTests(unittest.TestCase):
    def test_official_txt_csv_have_identical_row_and_label_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata(root / "txt")
            metadata(root / "csv", txt=False)
            txt, _, _ = celeba.load_metadata(root / "txt", expected_images=4, expected_identities=2)
            csv, _, _ = celeba.load_metadata(root / "csv", expected_images=4, expected_identities=2)
            pd.testing.assert_frame_equal(txt, csv)
            self.assertEqual(txt.label.tolist(), [1, 0, 0, 1])
            self.assertEqual(txt.source_index.tolist(), [0, 1, 2, 3])
            self.assertEqual(txt.identity.tolist(), [2, 2, 5, 5])

    def test_metadata_rejects_missing_duplicate_and_nonbinary_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata(root)
            path = root / "identity_CelebA.txt"
            path.write_text("000001.jpg 2\n000001.jpg 2\n000003.jpg 5\n000004.jpg 5\n")
            with self.assertRaisesRegex(RuntimeError, "one-to-one"):
                celeba.load_metadata(root, expected_images=4, expected_identities=2)
            metadata(root)
            path = root / "list_attr_celeba.txt"
            path.write_text(path.read_text().replace("000001.jpg 1 -1", "000001.jpg 0 -1"))
            with self.assertRaisesRegex(RuntimeError, "-1/\\+1"):
                celeba.load_metadata(root, expected_images=4, expected_identities=2)

    def test_actual_crop_pixels_and_cache_corruption_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / "raw", Path(directory) / "cache"
            root.mkdir(); output.mkdir()
            pixels = np.arange(80 * 120 * 3, dtype=np.uint16).reshape(80, 120, 3).astype(np.uint8)
            Image.fromarray(pixels).save(root / "000001.jpg", quality=95)
            raw, receipt = celeba.decode_pixels(root, ["000001.jpg"], output, "fixed", Reporter())
            with Image.open(root / "000001.jpg") as image:
                expected = np.asarray(image.convert("RGB").crop((20, 0, 100, 80)).resize(
                    (64, 64), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
            np.testing.assert_array_equal(raw[0], expected)
            del raw
            replay, second = celeba.decode_pixels(root, ["000001.jpg"], output, "fixed", Reporter())
            self.assertEqual(receipt, second)
            del replay
            with (output / "celeba_rgb64.npy").open("r+b") as stream:
                stream.seek(-1, 2); stream.write(b"x")
            with self.assertRaisesRegex(RuntimeError, "Committed.*differ"):
                celeba.decode_pixels(root, ["000001.jpg"], output, "fixed", Reporter())

    def test_raw_source_hash_detects_changed_jpeg_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jpg").write_bytes(b"original")
            first = celeba.verify_raw_images(root, ["a.jpg"], Reporter())
            (root / "a.jpg").write_bytes(b"modified")
            second = celeba.verify_raw_images(root, ["a.jpg"], Reporter())
            self.assertNotEqual(first["ordered_files_sha256"], second["ordered_files_sha256"])

    def test_vgg_downloaded_bytes_checked_without_external_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset").write_bytes(b"raw-data")
            pin = {"path": "asset", "size": 8, "sha256": hashlib.sha256(b"raw-data").hexdigest()}
            path = root / "pins.json"
            path.write_text(json.dumps({"assets": [pin]}))
            self.assertEqual(vgg.checked_sources(root, path), {"asset": pin})
            (root / "asset").write_bytes(b"bad-data")
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                vgg.checked_sources(root, path)

    def test_maad_ternary_labels_tokens_and_training_eligibility(self):
        frame = pd.DataFrame({"Filename": ["n000002/0001_01.jpg"] * 3,
                              "Identity": ["2"] * 3, "Smiling": [-1, 0, 1]})
        ids, exceptions = vgg.validate_maad_rows(frame)
        self.assertEqual(ids.tolist(), ["n000002"] * 3)
        self.assertEqual(exceptions, [])
        malformed = frame.copy(); malformed.loc[0, "Smiling"] = 2
        with self.assertRaisesRegex(RuntimeError, "encoding"):
            vgg.validate_maad_rows(malformed)
        wrong = frame.copy(); wrong.loc[0, "Identity"] = "3"
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            vgg.validate_maad_rows(wrong)
        records = [{"numeric_identity": i, "official_split": "train", "defined": 51,
                    "positive": 1, "negative": 50, "undefined": 100} for i in range(5)]
        records[1]["defined"] = 50
        records[2]["positive"] = 0
        records[3]["negative"] = 0
        records[4]["official_split"] = "test"
        self.assertEqual(vgg_master.eligible_identity_ids(records), {0})

    def test_streaming_vgg_archive_decodes_only_requested_pixels_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root, out = Path(directory) / "raw", Path(directory) / "cache"
            (root / "data").mkdir(parents=True); out.mkdir()
            image = Image.fromarray(np.random.default_rng(9).integers(0, 255, (90, 100, 3), dtype=np.uint8))
            stream = bytes_io.BytesIO(); image.save(stream, format="JPEG")
            encoded = stream.getvalue(); name = "n000002/0001_01.jpg"
            archive = root / "data/vggface2_train.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo("train/" + name); member.size = len(encoded)
                tar.addfile(member, bytes_io.BytesIO(encoded))
            source = {"size": archive.stat().st_size, "sha256": vgg.file_sha256(archive)}
            box = [10, 10, 60, 50]
            raw, index, receipts = vgg.decode_candidates(root, out, {name}, {name: box}, source, Reporter())
            expected, geometry = vgg.crop_rgb64(encoded, box)
            np.testing.assert_array_equal(raw[index[name]], expected)
            self.assertTrue(receipts[name]["valid"])
            self.assertEqual(receipts[name]["source_image_sha256"], hashlib.sha256(encoded).hexdigest())
            del raw
            replay, _, _ = vgg.decode_candidates(root, out, {name}, {name: box}, source, Reporter())
            del replay
            with self.assertRaisesRegex(RuntimeError, "cache differs"):
                vgg.decode_candidates(root, out, {name}, {name: box}, {**source, "sha256": "0" * 64}, Reporter())

    def test_fixed_config_is_complete_and_does_not_read_prior_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = api.build_study_config("celeba", root / "raw", root / "cache", root / "output")
            spec = json.loads(path.read_text())
            self.assertEqual(spec["participants"], [2, 5, 10, 20, 50])
            self.assertEqual(spec["planned_training_fits"], 1097)
            self.assertEqual(spec["noise_scale_upper_bounds"], {"anchor": .08, "private": 3.})
            self.assertEqual(spec["max_proposals_per_family"], 1500)
            self.assertFalse(any(key in spec for key in ["prepared_cache_root", "design_source", "auditor_checkpoint_source"]))
            self.assertEqual(path, api.build_study_config("celeba", root / "raw", root / "cache", root / "output"))
            with self.assertRaisesRegex(ValueError, "isolated"):
                api.study_spec("celeba", root / "raw", root / "raw/cache", root / "output")
            with self.assertRaisesRegex(RuntimeError, "prepare_dataset first"):
                api.load_context("celeba", root / "raw", root / "cache", root / "context", device="cpu")


class PreparedContextTests(unittest.TestCase):
    def test_synthetic_400d_raw_context_has_disjoint_splits_and_fixed_1000_lineups(self):
        """Exercise actual adapter/galleries with synthetic fresh generated memmaps."""
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cache = root / "cache"; (cache / "basis").mkdir(parents=True)
            raw = np.lib.format.open_memmap(cache / "pixels.npy", mode="w+", dtype=np.uint8,
                                            shape=(600, 3, 64, 64))
            for index in range(600):
                raw[index] = index % 256
            raw.flush()
            reduced = np.lib.format.open_memmap(cache / "reduced.npy", mode="w+", dtype=np.float32,
                                                shape=(600, 400))
            reduced[:] = np.arange(600)[:, None] % 256 / 255.
            reduced.flush()
            F = torch.zeros(12288, 400); F[:400] = torch.eye(400)
            basis_path = cache / "basis/vgg_public_basis.pt"
            torch.save({"F": F, "singular_values": torch.ones(400)}, basis_path)
            assignments = []
            for participant in range(2):
                for j in range(100):
                    identity = participant * 100 + j
                    assignments.append({"participant": participant,
                        "split": "train" if j < 70 else "val" if j < 85 else "test",
                        "identity": identity, "reserved_source": identity * 3,
                        "reserved_filename": f"{identity}_reference.jpg",
                        "protocol_filenames": [f"{identity}_a.jpg", f"{identity}_b.jpg"],
                        "protocol_labels": [0, 1], "protocol_sources": [identity * 3 + 1, identity * 3 + 2]})
            manifest = {"version": "celeba-synthetic", "config": {"seed": 20260713, "pmax": 50,
                        "identities_per_client": 100}, "assignments": assignments, "sources": {},
                        "code_sha256": {"frozen_base": vgg.file_sha256(base.__file__)},
                        "raw_data": {"file": "pixels.npy", "bytes": (cache / "pixels.npy").stat().st_size,
                                     "sha256": vgg.file_sha256(cache / "pixels.npy")},
                        "projection": {"file": "reduced.npy", "bytes": (cache / "reduced.npy").stat().st_size,
                                       "sha256": vgg.file_sha256(cache / "reduced.npy")},
                        "basis": {"basis_file_sha256": vgg.file_sha256(basis_path), "uncentered": True},
                        "cache_validation": "synthetic fresh byte hashes"}
            manifest["manifest_hash"] = vgg.canonical_hash(manifest)
            vgg.atomic_json(cache / "master_manifest.json", manifest)
            vgg_master.verify_master_artifacts(cache, manifest, Reporter())
            del raw, reduced
            ctx = vgg_context.load_frozen_data(root / "raw", 2, api.HERE, root / "context", cache)
            self.assertEqual(ctx.query_reduced.shape, (100, 400))
            self.assertEqual(ctx.data.references.raw_uint8.shape, (200, 3, 64, 64))
            self.assertEqual(len(ctx.linkage.cross_galleries), 100)
            self.assertEqual(sum(len(galleries) for galleries in ctx.linkage.cross_galleries), 1000)
            split_ids = [set(ctx.data.split(name).identities.tolist()) for name in ("train", "val", "test")]
            self.assertFalse(split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2])
            for identity, groups in zip(ctx.linkage.query_identities, ctx.linkage.cross_galleries):
                for gallery in groups:
                    self.assertEqual(len(set(gallery)), 10)
                    self.assertIn(("reference", identity), gallery)
            again = vgg_context.load_frozen_data(root / "raw", 2, api.HERE, root / "other", cache)
            self.assertEqual(ctx.linkage.manifest_hash, again.linkage.manifest_hash)
            del ctx, again
            gc.collect()  # Release Windows file-backed memmaps before TemporaryDirectory cleanup.


if __name__ == "__main__":
    unittest.main()
