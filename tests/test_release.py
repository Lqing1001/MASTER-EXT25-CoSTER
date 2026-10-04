import unittest
import numpy as np
import torch
from CoSTER import CoSTER,daily_zscore
from investment.backtest import test_longshort,test_longonly_and_hac
from evaluation.statistics import moving_block_interval,hac_mean_test
class ReleaseTests(unittest.TestCase):
 def test_accounting(self):test_longshort();test_longonly_and_hac()
 def test_bootstrap(self):
  self.assertTrue(np.allclose(moving_block_interval(np.full(60,.02)),[.02,.02]))
  self.assertTrue(np.isfinite(hac_mean_test(np.arange(60.))['p_value']))
 def test_full_model(self):
  torch.set_num_threads(1);torch.manual_seed(9);m=CoSTER().eval();x=torch.randn(12,8,221);x[:,:,158:]=x[:1,:,158:].clone()
  with torch.no_grad():y=m(x,True)
  self.assertEqual(tuple(y['scores'].shape),(12,));self.assertTrue(torch.isfinite(y['scores']).all())
  self.assertTrue(torch.allclose(y['scores'],.5*daily_zscore(y['master'])+.5*daily_zscore(y['continuous'])))
  self.assertEqual(sum(p.numel() for p in m.continuous.parameters()),402792)
  self.assertIn('time_position',m.continuous.base.base.state_dict())
if __name__=='__main__':unittest.main()
