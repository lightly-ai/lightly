import itertools
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lightly.loss import dino_loss
from lightly.loss import DINOLoss


class FacebookDINOLoss(nn.Module):
    """Copy paste from the original DINO paper. We use this to verify our
    implementation.

    The only change from the original code is that distributed training is no
    longer assumed.

    Source: https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/main_dino.py#L363
    
    """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class TestDINOLoss(unittest.TestCase):

    def generate_output_nested(self, batch_size=2, n_views=3, out_dim=4, seed=0):
        """Generates list of embeddings nested by image and view.
        
        Example output:
            torch.Tensor([
                [img0_view0, img0_view1],
                [img1_view0, img1_view1],
            ])

        """
        torch.manual_seed(seed)
        out = []
        for _ in range(batch_size):
            views = [torch.rand(out_dim) for _ in range(n_views)]
            out.append(torch.stack(views))
        return torch.stack(out)

    def generate_output_flat(self, batch_size=2, n_views=3, out_dim=4, seed=0):
        """Generates flat list of embeddings with the same order as the
        output of DINOCollateFunction.

        Example output:
            torch.Tensor([img0_view0, img0_view1, img1_view0, img1_view1])
        
        """
        torch.manual_seed(seed)
        out = []
        for _ in range(batch_size):
            for _ in range(n_views):
                out.append(torch.rand(out_dim))
        return torch.stack(out)

    def generate_output_facebook(self, batch_size=2, n_views=3, out_dim=4, seed=0):
        """Generates list of embeddings with in the same order as the reference
        DINO implementation from Facebook.
        
        Example output:
            torch.Tensor([img0_view0, img1_view0, img0_view1, img1_view1])
        
        """
        torch.manual_seed(seed)
        nested = self.generate_output_nested(batch_size, n_views, out_dim, seed)
        out = []
        for v in range(n_views):
            for b in range(batch_size):
                out.append(nested[b][v])
        return torch.stack(out)

    def test_concat_student_outputs(self):
        batch_size = 2
        out_dim = 4

        nested = self.generate_output_nested(batch_size=batch_size, out_dim=out_dim)
        expected = nested

        # test single input vector
        student_out = self.generate_output_flat(batch_size=batch_size, out_dim=out_dim)
        concat = dino_loss._concat_student_outputs(batch_size, out_dim, student_out)
        assert torch.all(concat == expected)

        # test list of input vectors
        global_views = torch.stack([
            *nested[0][:2],
            *nested[1][:2],
        ])
        local_views = torch.stack([
            *nested[0][2:],
            *nested[1][2:],
        ])
        concat = dino_loss._concat_student_outputs(
            batch_size,
            out_dim,
            [global_views, local_views],
        )
        assert torch.all(concat == expected)

    def test_dino_loss_equal_to_original(self):

        def test(
            batch_size=3,
            n_global=2, # number of global views
            n_local=6,  # number of local views
            out_dim=4,
            warmup_teacher_temp=0.04,
            teacher_temp=0.04,
            warmup_teacher_temp_epochs=30,
            student_temp=0.1,
            center_momentum=0.9,
            epoch=0,
            n_epochs=100,
        ): 
            """Runs test with the given input parameters."""
            with self.subTest(
                f'batch_size={batch_size}, n_global={n_global}, '
                f'n_local={n_local}, out_dim={out_dim}, '
                f'warmup_teacher_temp={warmup_teacher_temp}, '
                f'teacher_temp={teacher_temp}, '
                f'warmup_teacher_temp_epochs={warmup_teacher_temp_epochs}, '
                f'student_temp={student_temp}, '
                f'center_momentum={center_momentum}, epoch={epoch}, '
                f'n_epochs={n_epochs}'
            ):
                our_loss_fn = DINOLoss(
                    out_dim=out_dim,
                    warmup_teacher_temp=warmup_teacher_temp,
                    teacher_temp=teacher_temp,
                    warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
                    student_temp=student_temp,
                    center_momentum=center_momentum,
                )
                
                fb_loss_fn = FacebookDINOLoss(
                    out_dim=out_dim,
                    ncrops=n_global + n_local,
                    teacher_temp=teacher_temp,
                    warmup_teacher_temp=warmup_teacher_temp,
                    warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
                    nepochs=n_epochs,
                    student_temp=student_temp,
                    center_momentum=center_momentum,
                )

                our_teacher_out = self.generate_output_flat(
                    batch_size=batch_size,
                    n_views=n_global,
                    out_dim=out_dim,
                    seed=0,
                )
                our_student_out = self.generate_output_flat(
                    batch_size=batch_size,
                    n_views=n_global + n_local,
                    out_dim=out_dim, 
                    seed=1,
                )
                fb_teacher_out = self.generate_output_facebook(
                    batch_size,
                    n_views=n_global,
                    out_dim=out_dim, 
                    seed=0,
                )
                fb_student_out = self.generate_output_facebook(
                    batch_size=batch_size,
                    n_views=n_global + n_local,
                    out_dim=out_dim, 
                    seed=1,
                )
                our_loss = our_loss_fn(
                    teacher_out=our_teacher_out, 
                    student_out=our_student_out, 
                    epoch=epoch,
                    n_views_teacher=n_global,
                )
                fb_loss = fb_loss_fn(
                    student_output=fb_student_out, 
                    teacher_output=fb_teacher_out, 
                    epoch=epoch
                )
                assert torch.allclose(our_loss_fn.center, fb_loss_fn.center)
                assert torch.allclose(our_loss, fb_loss)

        def test_all(**kwargs):
            """Tests all combinations of the input parameters"""
            parameters = []
            for name, values in kwargs.items():
                parameters.append([(name, value) for value in values])
            # parameters = [
            #   [(param1, val11), (param1, val12), ..],
            #   [(param2, val21), (param2, val22), ..],
            #   ...
            # ]

            for params in itertools.product(*parameters):
                # params = [(param1, value1), (param2, value2), ...]
                test(**dict(params))
        
        # test input sizes
        test_all(
            batch_size=np.arange(1,4),
            n_local=np.arange(1, 4),
            out_dim=np.arange(1, 4),
        )
        # test teacher temp warmup
        test_all(
            warmup_teacher_temp=[0.01, 0.04, 0.07],
            teacher_temp=[0.01, 0.04, 0.07],
            warmup_teacher_temp_epochs=[0, 1, 10],
            epoch=[0, 1, 10, 20],
        )
        # test other params
        test_all(
            student_temp=[0.05, 0.1, 0.2],
            center_momentum=[0.5, 0.9, 0.95],
        )
