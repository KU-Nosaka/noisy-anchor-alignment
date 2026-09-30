import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import attack_comparison as audit
import core_protocols as core
import notebook_api
from spectral import spectral_align


class AttackAlgebraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, p=3, r=24, h=5):
        gen=torch.Generator().manual_seed(9017)
        g=torch.randn((r,h),dtype=torch.float64,generator=gen)
        a,_=torch.linalg.qr(g-g.mean(0),mode='reduced')
        rotations=[core.haar_orthogonal(h,generator=gen) for _ in range(p)]
        shifts=[torch.randn(h,dtype=torch.float64,generator=gen) for _ in range(p)]
        q=torch.randn((7,h),dtype=torch.float64,generator=gen)
        b=torch.stack([a@o+psi for o,psi in zip(rotations,shifts)])
        return a,rotations,shifts,q,b,gen

    def test_all_three_noiseless_recover_true_queries(self):
        a,o,psi,q,b,_=self.fixture()
        result,diagnostics=audit.recover_attacks(a,b,q@o[1]+psi[1],o[0])
        for name in audit.ATTACKS:
            torch.testing.assert_close(result[name],q,rtol=1e-11,atol=1e-11)
        self.assertEqual(diagnostics['MP']['numerical_rank'],q.shape[1])

    def test_matches_frozen_attack_orientation_and_spectral_initializer(self):
        a,o,psi,q,b,gen=self.fixture()
        b=b+.045*torch.randn(b.shape,dtype=torch.float64,generator=gen)
        y=q@o[1]+psi[1]
        recovered,_=audit.recover_attacks(a,b,y,o[0])
        torch.testing.assert_close(recovered['MP'],core.attack_pa_mp(a,b[1],y).reduced,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(recovered['OP'],core.attack_pa_op(a,b[1],y).reduced,rtol=1e-10,atol=1e-10)
        centered=b-b.mean(1,keepdim=True)
        new,_=spectral_align(centered)
        old,diagnostics=core.gpm_align(list(centered),raw_anchor=a,max_iters=0,n_random_starts=0)
        torch.testing.assert_close(new,torch.stack(old),rtol=0,atol=0)
        self.assertEqual(diagnostics.iterations,0)
        expected=core.attack_pa_alignment_calibration(a,b[1],y,new[1],[o[0]],[new[0]]).reduced
        torch.testing.assert_close(recovered['AM'],expected,rtol=1e-10,atol=1e-10)

    def test_am_is_invariant_to_common_right_gauge(self):
        a,o,psi,q,b,gen=self.fixture()
        b=b+.08*torch.randn(b.shape,dtype=torch.float64,generator=gen)
        rs,_=spectral_align(b-b.mean(1,keepdim=True))
        gauge=core.haar_orthogonal(q.shape[1],generator=gen)
        torch.testing.assert_close(audit.am_rotation(o[0],rs),audit.am_rotation(o[0],rs@gauge),rtol=1e-12,atol=1e-12)

    def test_mp_uses_inverse_not_transpose_for_nonorthogonal_estimate(self):
        a,o,psi,q,b,_=self.fixture()
        m=torch.tensor([[2.,.3,0,0,0],[0,1.4,.1,0,0],[0,0,.5,0,0],[0,0,0,1.,.2],[0,0,0,0,.8]],dtype=torch.float64)
        b[1]=a@m+psi[1]; y=q@m+psi[1]
        recovered,diag=audit.recover_attacks(a,b,y,o[0])
        torch.testing.assert_close(recovered['MP'],q,rtol=1e-11,atol=1e-11)
        self.assertGreater(float(torch.linalg.norm(recovered['OP']-q)),.1)
        self.assertFalse(diag['MP']['rank_truncated'])

    def test_pinv_default_cutoff_and_rank_deficiency_are_recorded(self):
        a,o,psi,q,b,_=self.fixture()
        m=torch.diag(torch.tensor([2.,1.,.5,1e-17,0.],dtype=torch.float64))
        b[1]=a@m+psi[1]; y=q@m+psi[1]
        recovered,diag=audit.recover_attacks(a,b,y,o[0])
        expected=q.clone();expected[:,3:]=0
        torch.testing.assert_close(recovered['MP'],expected,rtol=1e-10,atol=1e-10)
        self.assertEqual(diag['MP']['rtol'],5*torch.finfo(torch.float64).eps)
        self.assertEqual(diag['MP']['atol'],0)
        self.assertEqual(diag['MP']['numerical_rank'],3)
        self.assertTrue(diag['MP']['rank_truncated'])
        self.assertEqual(diag['MP']['effective_cutoff'],diag['MP']['sigma_max']*diag['MP']['rtol'])


class PairingAndScoringTests(unittest.TestCase):
    def test_schedule_is_exactly500_unconditional_paired_draws(self):
        first=audit.make_schedule();second=audit.make_schedule()
        self.assertEqual(first,second)
        self.assertEqual(len(first['draws']),500)
        self.assertEqual([r['draw'] for r in first['draws']],list(range(500)))
        self.assertTrue(all(0<r['scale']<.08 for r in first['draws']))
        self.assertEqual(len({s for r in first['draws'] for s in r['noise_seeds']}),5000)
        self.assertFalse(first['adaptive_selection']);self.assertFalse(first['bin_acceptance'])
        self.assertTrue(first['paired_across_attacks']);self.assertTrue(first['paired_across_datasets'])
        self.assertNotEqual(first,audit.make_schedule(20260931))

    def test_identical_condition_reuses_noise_but_next_draw_is_fresh(self):
        a,o,psi,q,b,_=AttackAlgebraTests().fixture(p=10)
        bank={'anchor':a,'rotations':o,'translations':psi};schedule=audit.make_schedule()
        x,h=audit.gaussian_anchors(bank,schedule['draws'][0]);y,j=audit.gaussian_anchors(bank,schedule['draws'][0])
        z,k=audit.gaussian_anchors(bank,schedule['draws'][1])
        torch.testing.assert_close(x,y,rtol=0,atol=0);self.assertEqual(h,j)
        self.assertNotEqual(h,k);self.assertFalse(torch.equal(x,z))
        self.assertEqual(len(set(h)),10)

    def test_ties_use_single_original_argsort_winner(self):
        q=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float32)
        refs=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],dtype=torch.float32)
        galleries=torch.tensor([[[1,0,2]],[[0,2,1]]])
        true=torch.tensor([[1],[1]])
        result=audit.gallery_success(q,refs,galleries,true)
        expected=[]
        for i in range(2):
            scores=torch.stack([refs[int(j)] for j in galleries[i,0]])@q[i]
            expected.append([int(torch.argsort(scores,descending=True)[0])])
        self.assertEqual(result['winner_positions'],expected)
        self.assertEqual(result['trial_successes'][0][0],int(expected[0][0]==1))
        self.assertEqual(result['trials'],2)

    def test_resume_rejects_changed_conditions_or_corrupted_results(self):
        condition=audit.make_schedule()['draws'][0]
        result={'trial_successes':[[1]*10 for _ in range(100)],'correct_per_identity':[10]*100,
                'trials':1000,'correct_trials':1000,'linkage_percent':100.,'winner_positions':[[0]*10 for _ in range(100)]}
        record={'version':audit.VERSION,'dataset':'celeba','schedule_hash':'schedule','training_performed':False,
                'identity_hash':'frozen','condition':condition,'attacks':{name:copy.deepcopy(result) for name in audit.ATTACKS}}
        record['record_sha256']=audit.canonical(record)
        audit.validate_record(record,'frozen',condition)
        with self.assertRaises(RuntimeError):audit.validate_record(record,'changed',condition)
        wrong=copy.deepcopy(condition);wrong['scale']+=1e-6
        with self.assertRaises(RuntimeError):audit.validate_record(record,'frozen',wrong)
        bad=copy.deepcopy(record);bad['attacks']['MP']['linkage_percent']=99.
        with self.assertRaises(RuntimeError):audit.validate_record(bad,'frozen',condition)
        bad['record_sha256']=audit.canonical({k:v for k,v in bad.items() if k!='record_sha256'})
        with self.assertRaises(RuntimeError):audit.validate_record(bad,'frozen',condition)
        with self.assertRaises(RuntimeError):audit.validate_record(record,'frozen',condition,schedule_hash='other')
        bad=copy.deepcopy(record);bad['attacks']['AM']['winner_positions'][0][0]=1
        bad['record_sha256']=audit.canonical({k:v for k,v in bad.items() if k!='record_sha256'})
        with self.assertRaises(RuntimeError):audit.validate_record(bad,'frozen',condition,genuine_positions=torch.zeros(100,10,dtype=torch.int64))

    def test_immutable_record_and_input_lock_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'record.json';record=audit.commit_record(path,{'value':1})
            before=path.read_bytes()
            with self.assertRaises(RuntimeError):audit.commit_record(path,{'value':2})
            self.assertEqual(path.read_bytes(),before)
            lock=Path(temp)/'identity.json';audit.lock_json(lock,{'input':'a'});audit.lock_json(lock,{'input':'a'})
            with self.assertRaises(RuntimeError):audit.lock_json(lock,{'input':'b'})


class RawRunnerResumeTests(unittest.TestCase):
    """Exercise partial commits/resume with synthetic rows and a fake auditor."""

    def fixture(self):
        a, rotations, shifts, _, _, gen = AttackAlgebraTests().fixture(p=10)
        query = torch.randn((100, 5), dtype=torch.float64, generator=gen)
        F = torch.zeros((12288, 5), dtype=torch.float32); F[:5] = torch.eye(5)
        refs = torch.zeros((1000, 3, 64, 64), dtype=torch.uint8)
        refs.flatten(1)[:, :5] = torch.randint(1, 255, (1000, 5), generator=gen, dtype=torch.uint8)
        galleries = torch.arange(10).repeat(100, 10, 1)
        genuine = torch.zeros((100, 10), dtype=torch.long)
        metadata = {'pretrained': 'casia-webface', 'frozen': True,
            'checkpoint_sha256': audit.casia.CHECKPOINT_SHA256, 'state_sha256': audit.casia.STATE_SHA256}
        source = {'genuine_positions': genuine.tolist(), 'context_input_hash': 'synthetic-generated-context'}
        bank = {'anchor': a, 'rotations': rotations, 'translations': shifts}
        return F, query, refs, bank, galleries, genuine, metadata, source

    def test_partial_run_resumes_exact_schedule_and_keeps_committed_bytes(self):
        inputs = self.fixture()
        api = SimpleNamespace(prepare_dataset=mock.Mock(),
                              load_context=mock.Mock(return_value=SimpleNamespace(face_model=object())))

        def embedding(model, pixels, **kwargs):
            return torch.nn.functional.normalize(pixels.flatten(1)[:, :5], dim=1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; data.mkdir()
            output = root / 'results'
            with mock.patch.object(notebook_api, 'prepare_dataset', api.prepare_dataset), \
                 mock.patch.object(notebook_api, 'load_context', api.load_context), \
                 mock.patch.object(audit, '_context_inputs', return_value=inputs), \
                 mock.patch.object(audit.casia, 'embeddings', side_effect=embedding):
                first = audit.run_attack_comparison('celeba', data, root/'cache', output,
                                                     device='cpu', max_new_draws=1)
                self.assertEqual(first['completed_draws'], 1)
                record_path = output/'study/records/draw_0000.json'
                before = record_path.read_bytes()
                zero_before = (output/'study/zero_control.json').read_bytes()
                second = audit.run_attack_comparison('celeba', data, root/'cache', output,
                                                      device='cpu', max_new_draws=1)
                self.assertEqual(second['completed_draws'], 2)
                self.assertEqual(record_path.read_bytes(), before)
                self.assertEqual((output/'study/zero_control.json').read_bytes(), zero_before)
                self.assertEqual(audit.read(output/'study/schedule.json'), audit.make_schedule())
                self.assertFalse((output/'study/completion.json').exists())
                self.assertTrue(second['zero_control_excluded'])
                self.assertFalse(second['training_performed'])
                with self.assertRaises(RuntimeError): audit.inspect_results(output)
                with self.assertRaises(RuntimeError):
                    audit.run_attack_comparison('celeba', data, root/'cache', output,
                                                 device='cpu', seed=20260931, max_new_draws=0)
                self.assertEqual(record_path.read_bytes(), before)
                saved = audit.read(output/'study/summary.json')
                saved['statistics']['OP']['mean_linkage_percent'] += 1
                audit.atomic_json(output/'study/summary.json', saved)
                with self.assertRaises(RuntimeError): audit.inspect_results(output, require_complete=False)
            self.assertEqual(api.prepare_dataset.call_count, 3)
            self.assertEqual(api.load_context.call_count, 3)

    def test_float_winner_indices_are_rejected_even_if_checksum_recomputed(self):
        result = {'trial_successes': [[1]*10 for _ in range(100)], 'correct_per_identity': [10]*100,
                  'trials': 1000, 'correct_trials': 1000, 'linkage_percent': 100.,
                  'winner_positions': [[0.]*10 for _ in range(100)]}
        condition = audit.make_schedule()['draws'][0]
        record = {'version': audit.VERSION, 'dataset': 'celeba', 'schedule_hash': 'schedule',
                  'training_performed': False, 'identity_hash': 'input', 'condition': condition,
                  'attacks': {name: copy.deepcopy(result) for name in audit.ATTACKS}}
        record['record_sha256'] = audit.canonical(record)
        with self.assertRaises(RuntimeError): audit.validate_record(record, 'input', condition)


if __name__=='__main__':unittest.main()
