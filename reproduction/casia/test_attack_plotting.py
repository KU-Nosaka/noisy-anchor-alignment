import unittest

import numpy as np

import attack_plotting as plotting
import attack_comparison as attack


class PairedSplineTests(unittest.TestCase):
    def test_paired_intervals_preserve_constant_difference_between_attacks(self):
        x = np.array([row['scale'] for row in attack.make_schedule()['draws']])
        baseline = 18 + 120*x + .3*np.sin(np.arange(500))
        y = np.column_stack([baseline, baseline+2, baseline+7])
        fit = plotting.paired_spline(x, y, resamples=32)
        self.assertEqual(fit['estimate'].shape, (201,3))
        np.testing.assert_allclose(fit['bootstrap_means'][:,2]-fit['bootstrap_means'][:,0], 7, atol=1e-12)
        np.testing.assert_allclose(fit['lower'][:,2]-fit['lower'][:,0], 7, atol=1e-8)
        np.testing.assert_allclose(fit['upper'][:,2]-fit['upper'][:,0], 7, atol=1e-8)

    def test_zero_control_or_nonfinite_scores_are_rejected(self):
        x = np.array([row['scale'] for row in attack.make_schedule()['draws']])
        y = np.full((500,3), 20.)
        x[0] = 0.
        with self.assertRaises(RuntimeError): plotting.paired_spline(x,y,resamples=2)
        x[0] = .03; y[0,0] = np.nan
        with self.assertRaises(RuntimeError): plotting.paired_spline(x,y,resamples=2)


if __name__ == '__main__': unittest.main()
