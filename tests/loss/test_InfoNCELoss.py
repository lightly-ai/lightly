import torch 
from lightly.loss.directclr_loss import InfoNCELoss
import unittest
import torchvision

class InfoNCETest(unittest.TestCase):
    def setUp(self):
        resnet = torchvision.models.resnet50(pretrained = False)
        self.model = torch.nn.Sequential(*list(resnet.children)[:-1])
        self.dim = 360
        self.loss = InfoNCELoss(dim = self.dim)

    def test_infonce_forward(self, seed = 42):
        torch.manual_seed(seed = seed)
        viewOne = torch.randn(1,3,256,256)
        viewTwo = torch.randn(1,3,256,256)
        z1 = self.model(viewOne)
        z2 = self.model(viewTwo)
        loss = self.loss(z1, z2)
    
    def test_infonce_backward(self, seed=42):
        torch.manual_seed(seed=seed)
        out0 = torch.randn(1,3,256,256)
        out1 = torch.rand(1,3,256,256)
        out0 = self.model(out0)
        out1 = self.model(out1)
        #criterion = self.loss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        loss = self.loss(out0, out1)
        loss.backward()
        optimizer.step()

