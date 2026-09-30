"""Fresh parameter-bank algebra, immutable replay, and bounded-memory checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import context_common as common
import data_pipeline as dp

import core_protocols as core
import celeba_public_svd_identity100_runner as base


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fresh_full_design_shared_across_datasets_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, receipt, _ = common.fresh_design(core, base, Path(directory)/'a', 20260928)
            self.assertEqual(payload['anchor'].shape, (1600,400))
            self.assertEqual(len(payload['rotations']), 50)
            self.assertTrue(torch.allclose(payload['anchor'].T @ payload['anchor'],
                                          torch.eye(400,dtype=torch.float64),atol=1e-10,rtol=0))
            self.assertLess(float(payload['anchor'].mean(0).abs().max()),1e-10)
            # Distinct dataset outputs must receive exactly the same fresh bank.
            other, second, _ = common.fresh_design(core, base, Path(directory)/'b', 20260928)
            self.assertEqual(receipt['tensor_hashes'], second['tensor_hashes'])
            replay, third, _ = common.fresh_design(core, base, Path(directory)/'a', 20260928)
            self.assertEqual(receipt, third)
            with self.assertRaisesRegex(RuntimeError,'identity/checksum'):
                common.fresh_design(core, base, Path(directory)/'a', 20260929)
            # Knowing a legitimate common secret needs no fitted colluder rank.
            x = torch.randn(5,400,dtype=torch.float64)
            o, psi = payload['rotations'][0],payload['translations'][0]
            self.assertTrue(torch.allclose(core.invert_known_gdp(x@o+psi,o,psi).reduced,x,atol=1e-10,rtol=1e-10))
            # Public anchor correspondence independently recovers the private view.
            a,o,psi=payload['anchor'],payload['rotations'][1],payload['translations'][1]
            recovered=core.attack_pa_op(a,a@o+psi,x@o+psi).reduced
            self.assertTrue(torch.allclose(recovered,x,atol=1e-10,rtol=1e-10))

    def test_private_rms_slices_without_full_double_copy(self):
        value=torch.randn(23,400)
        class Lazy:
            def __len__(self): return len(value)
            def __getitem__(self,index):
                if index.stop-index.start>5: raise AssertionError('unbounded access')
                return value[index]
        self.assertAlmostEqual(common.private_rms(Lazy(),5),float(value.double().square().mean().sqrt()),places=14)

    def test_dataset_noise_namespaces_and_immutable_quota_configuration(self):
        config={'version':'test','seed':20260928,'dataset':'celeba'}
        dp.configure(config); first=dp.VERSION
        dp.configure({**config,'dataset':'vggface2'})
        self.assertNotEqual(first,dp.VERSION)
        for change in ({'sampling_mode':'fixed100'}, {'conditions_per_bin_per_family':19},
                       {'linkage_bin_edges_percent':[0,15,30,45,60,100]}, {'max_proposals_per_family':2000}):
            with self.assertRaises(RuntimeError): dp.configure({**config,**change})


if __name__=='__main__': unittest.main()
