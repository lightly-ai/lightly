import unittest

import torch

from lightly.models.modules.heads import SwaVPrototypes

class TestSwavFreezingPrototypes(unittest.TestCase):
    def test_swav_frozen_prototypes(self, device: str = "cpu", seed=0):
        criterion = torch.nn.L1Loss()
        linear_layer = torch.nn.Linear(8, 8, bias=False)
        prototypes = SwaVPrototypes(8, 8, 2)
        optimizer = torch.optim.SGD(prototypes.parameters(), lr=0.01)
        torch.manual_seed(seed)
        in_features = torch.rand(4, 8, device="cpu")
        target_features = torch.ones(4, 8, device="cpu")
        for step in range(4):
            out_features = linear_layer(in_features)
            out_features = prototypes.forward(out_features, step)
            loss = criterion(out_features, target_features)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step == 0:
                loss0 = loss
            if step <= 2:
                self.assertEqual(loss, loss0)
            if step > 2:
                self.assertNotEqual(loss, loss0) 
        
            
        
            
        