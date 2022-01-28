from typing import Union, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

def _concat_student_outputs(
    batch_size: int,
    out_dim: int,
    student_out: Union[torch.Tensor, List[torch.Tensor]],
) -> torch.Tensor:
    """Concatenates multiple outputs from the student model into a single 
    tensor.
    
    Args:
        batch_size:
            The batch size B where B = number of images.
        out_dim:
            The dimensions of the model output.
        student_out:
            A single tensor with shape (B * V, D) or a list of tensors where
            every tensor can have a different V.

    Returns:
        A tensor with shape (B, V_sum, D) where V_sum is the sum of all
        V in the input.
    
    """
    if isinstance(student_out, torch.Tensor):
        student_out = [student_out]
    
    student_out = [x.reshape(batch_size, -1, out_dim) for x in student_out]
    student_out = torch.cat(student_out, dim=1)
    return student_out


class DINOLoss(nn.Module):
    """
    Implementation of the loss described in 'Emerging Properties in 
    Self-Supervised Vision Transformers'. [0]

    This implementation follows the code published by the authors. [1]
    It supports global and local image crops. A linear warmup schedule for the
    teacher temperature is implemented to stabilize training at the beginning.
    Centering is applied to the teacher output to avoid model collapse.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        out_dim: 
            Dimension of the model output.
        warmup_teacher_temp:
            Initial value of the teacher temperature. Should be decreased if the
            training loss does not decrease.
        teacher_temp:
            Final value of the teacher temperature after linear warmup. Values
            above 0.07 result in unstable behavior in most cases. Can be
            slightly increased to improve performance during finetuning.
        warmup_teacher_temp_epochs:
            Number of epochs for the teacher temperature warmup.
        student_temp:
            Temperature of the student.
        center_momentum:
            Momentum term for the center calculation.

    Examples:

        >>> # initialize loss function
        >>> loss_fn = DINOLoss(128)
        >>>
        >>> # generate views from some images img1 and img2
        >>> # t = some random image transformation function
        >>> views = torch.stack([t(img1), t(img1), t(img2), t(img2)])
        >>>
        >>> # embed the views with a student and a teacher model
        >>> student_out = student(views)
        >>> teacher_out = teacher(views)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(student_out, teacher_out, epoch=0)

    """
    def __init__(
        self, 
        out_dim: int,
        warmup_teacher_temp: float = 0.04, 
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30, 
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.linspace(
            start=warmup_teacher_temp, 
            stop=teacher_temp,
            num=warmup_teacher_temp_epochs,
        )

    def forward(
        self, 
        teacher_out: torch.Tensor, 
        student_out: Union[torch.Tensor, List[torch.Tensor]], 
        epoch: int,
        n_views_teacher: int = 2,
    ):
        """Cross-entropy between softmax outputs of the teacher and student 
        networks.

        Input tensors are assumed to have shape (B * V, D) where
        B = number of images in the batch, V = number of views per image, and
        D = the output dimension. The number of views per image can differ 
        between the teacher and student models and we refer to them with V_t 
        and V_s respectively.

        Args:
            teacher_out:
                Output of the teacher model with shape (B * V_t, D).
            student_out:
                Output of the student model. Either a single tensor with shape
                (B * V_s, D) or a list of tensors where each tensor can have a
                different V_s. Note that the first B * V_t entries in the tensor
                are assumed to be generated from the same views as the entries
                in teacher_out.
            epoch:
                The current training epoch.
            n_views_teacher:
                The number of views per image in the teacher_out tensor. This
                corresponds to V_t and is by default 2 as the teacher only sees
                the two global views in the standard DINO implementation.
       
        """
        batch_size = teacher_out.shape[0] // n_views_teacher
        out_dim = teacher_out.shape[1]

        # get teacher temperature
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.teacher_temp_schedule[epoch]
        else:
            teacher_temp = self.teacher_temp

        # teacher centering and sharpening
        teacher_center = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)

        # convert from (B*V_t, D) to (B, V_t, D) shape
        teacher_center = teacher_center.reshape(batch_size, n_views_teacher, out_dim)
        
        # convert from list of (B * V_s, D) to a single (B, sum(V_s), D) tensor
        student_out = _concat_student_outputs(
            batch_size,
            out_dim,
            student_out,
        )
        n_views_student = student_out.shape[1]

        student_out = F.log_softmax(student_out / self.student_temp, dim=-1)
        
        # we want to calculate the following loss:
        #
        # loss = 0
        # for t in range(n_views_teacher):
        #     for s in range(n_views_student):
        #         if t == s:
        #             # skip if teacher and student got same view     
        #             continue 
        #         loss += -torch.sum(teacher_out[:, t] * student_out[:, s])
        #
        # instead of a for loop we use the equivalent einsum operation with
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = out_dim
        loss = -torch.einsum('btd,bsd->ts', teacher_center, student_out)
        
        # ignore entries where the teacher and student got the same view
        loss.fill_diagonal_(0)
        
        # take average of loss
        loss = loss.sum() / (n_views_teacher * (n_views_student - 1) * batch_size)

        self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        """Moving average update of the center used for the teacher output.

        Args:
            teacher_out:
                Output from the teacher model with shape (B*V, D).

        """
        batch_center = torch.mean(teacher_out, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)