"""Synthetic scientific/integrity tests; never download data or model weights."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import casia_linkage as cl


class ConstantEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.eval(); self.seen = []

    def forward(self, batch):
        self.seen.append(batch.detach().clone())
        return torch.ones(len(batch), 512, device=batch.device)


def gallery_fixture():
    ids = np.arange(10, 20)
    galleries = np.stack([np.roll(np.arange(10), i) for i in range(10)])[None]
    return np.asarray([10]), ids, galleries


class CASIALinkageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_single_winner_ties_are_ten_percent_not_all_successes(self):
        qi, ri, galleries = gallery_fixture()
        q = torch.zeros(1, 512); q[:, 0] = 1
        r = q.repeat(10, 1)
        score = cl.score_embeddings(q, r, qi, ri, galleries)
        self.assertEqual(score['top1'], .1)
        self.assertEqual(sum(score['episode_successes'][0]), 1)
        self.assertEqual(score['chance_top1'], .1)
        self.assertEqual(score['ci95_identity_cluster'], [.1, .1])

    def test_distinct_embeddings_match_genuine_through_all_permutations(self):
        qi, ri, galleries = gallery_fixture()
        r = torch.eye(512)[:10]
        score = cl.score_embeddings(r[:1], r, qi, ri, galleries)
        self.assertEqual(score['top1'], 1.)
        self.assertEqual(score['episode_ranks'], [[1] * 10])
        with self.assertRaisesRegex(RuntimeError, 'normalized'):
            cl.score_embeddings(r[:1] * 2, r, qi, ri, galleries)

    def test_rgb64_facenet_preprocessing_and_batching(self):
        model = ConstantEmbedder()
        images = torch.stack([torch.zeros(3, 64, 64, dtype=torch.uint8),
                              torch.full((3, 64, 64), 255, dtype=torch.uint8)])
        outputs = cl.embeddings(model, images, device='cpu', batch_size=1)
        self.assertEqual([x.shape for x in model.seen], [torch.Size([1, 3, 160, 160])] * 2)
        self.assertTrue(torch.equal(model.seen[0], torch.full_like(model.seen[0], -127.5 / 128.)))
        self.assertTrue(torch.equal(model.seen[1], torch.full_like(model.seen[1], 127.5 / 128.)))
        self.assertTrue(torch.allclose(outputs.norm(dim=1), torch.ones(2)))
        with self.assertRaisesRegex(RuntimeError, 'frozen/eval'):
            cl.embeddings(model.train(), images, device='cpu')

    def test_decoding_keeps_exact_cached_query_coordinates(self):
        f = torch.zeros(12288, 400); f[:400] = torch.eye(400)
        raw = torch.full((2, 3, 64, 64), 255, dtype=torch.uint8)
        reduced = torch.zeros(2, 400); reduced[:, 1] = .123456789
        actual = cl.decode(raw, f, reduced)
        expected = (reduced @ f.T).reshape(2, 3, 64, 64)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, cl.decode(raw, f)))
        self.assertEqual(actual.flatten(1)[0, 0], 0.)
        self.assertEqual(actual.flatten(1)[0, 1], reduced[0, 1])

    def test_rejects_centered_or_nonorthogonal_basis(self):
        f = torch.zeros(12288, 400); f[:400] = torch.eye(400)
        with self.assertRaisesRegex(RuntimeError, 'uncentered'):
            cl.validate_basis(f, {'uncentered': False})
        with self.assertRaisesRegex(RuntimeError, 'orthonormal'):
            cl.validate_basis(f * 2, {'uncentered': True})
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            cl.validate_basis(f, {'uncentered': True, 'basis_matrix_sha256': 'wrong'})

    def test_gallery_and_source_image_isolation(self):
        qi, ri, galleries = gallery_fixture()
        queries = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
        references = torch.ones(10, 3, 64, 64, dtype=torch.uint8)
        qnames, rnames = ['query'], [f'ref{i}' for i in range(10)]
        cl.validate_inputs(queries, references, qi, ri, galleries, qnames, rnames)
        altered = galleries.copy(); altered[0, 0, 0] = altered[0, 0, 1]
        with self.assertRaisesRegex(RuntimeError, 'repeats'):
            cl.validate_inputs(queries, references, qi, ri, altered, qnames, rnames)
        with self.assertRaisesRegex(RuntimeError, 'image leakage'):
            cl.validate_inputs(queries, references, qi, ri, galleries, ['ref0'], rnames)
        references[0] = queries[0]
        with self.assertRaisesRegex(RuntimeError, 'Identical RGB64'):
            cl.validate_inputs(queries, references, qi, ri, galleries, qnames, rnames)

    def test_bootstrap_resamples_identities_and_is_reproducible(self):
        values = [0., .2, .5, 1.]
        seed = cl.named_seed(20260713, 'linkage-ci', 'cross_image_identity')
        rng = np.random.default_rng(seed)
        # Independent direct implementation of the original identity bootstrap.
        draws = np.asarray(values)[rng.integers(0, 4, size=(10000, 4))].mean(axis=1)
        expected = np.quantile(draws, [.025, .975]).tolist()
        self.assertEqual(cl.identity_bootstrap(values, seed), expected)
        self.assertEqual(cl.identity_bootstrap(values, seed), cl.identity_bootstrap(values, seed))

    def test_full_results_csv_resume_and_tamper_detection(self):
        qi, ri, galleries = gallery_fixture()
        f = torch.zeros(12288, 400); f[:400] = torch.eye(400)
        model = ConstantEmbedder()
        arguments = dict(queries=torch.zeros(1, 3, 64, 64, dtype=torch.uint8),
            references=torch.ones(10, 3, 64, 64, dtype=torch.uint8), query_ids=qi,
            reference_ids=ri, gallery_indices=galleries, query_names=['query'],
            reference_names=[f'ref{i}' for i in range(10)], basis=f,
            basis_metadata={'uncentered': True, 'basis_matrix_sha256': cl.tensor_sha(f)},
            model=model, auditor_metadata={'pretrained': 'casia-webface', 'frozen': True,
                'checkpoint_sha256': cl.CHECKPOINT_SHA256, 'state_sha256': cl.STATE_SHA256},
            provenance={'dataset': 'synthetic'}, device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            result = cl.evaluate_audit(**arguments, out_dir=directory)
            self.assertEqual(set(result['metrics']), {'raw_raw', 'decoded_decoded', 'decoded_raw'})
            self.assertEqual(result['metrics']['decoded_decoded']['top1'], .1)
            self.assertFalse(result['utility_training_executed'])
            calls = len(model.seen)
            self.assertEqual(cl.evaluate_audit(**arguments, out_dir=directory), result)
            self.assertEqual(len(model.seen), calls)
            path = Path(directory) / 'per_identity.csv'
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['decoded_decoded_successes'], '1')
            self.assertEqual(rows[0]['decoded_decoded_episodes'], '10')
            path.write_text('tampered')
            with self.assertRaisesRegex(RuntimeError, 'artifact differs'):
                cl.evaluate_audit(**arguments, out_dir=directory)

    def test_official_checkpoint_pin_if_local_asset_is_available(self):
        path = Path(__file__).parent / 'model_assets' / cl.CHECKPOINT_NAME
        if not path.exists():
            self.skipTest('Checkpoint download is not part of the unit test')
        self.assertEqual(path.stat().st_size, cl.CHECKPOINT_BYTES)
        self.assertEqual(cl.file_sha(path), cl.CHECKPOINT_SHA256)
        state = torch.load(path, map_location='cpu', weights_only=True)
        digest = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            digest.update(name.encode()); digest.update(tensor.contiguous().numpy().tobytes())
        self.assertEqual(digest.hexdigest(), cl.STATE_SHA256)
        self.assertEqual(state['logits.weight'].shape, (10575, 512))


if __name__ == '__main__':
    unittest.main()
