# This example requires the following dependencies to be installed:
# pip install lightly[timm]

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

# LeWM is a latent world model. It predicts the embedding of the next frame
# from the embeddings of the past frames and the actions that were taken, and
# it never reconstructs pixels. The encoder is trained from pixels together
# with the predictor. SIGReg keeps the embeddings close to an isotropic
# Gaussian, which is what prevents collapse, so no stop-gradient, no teacher
# network and no exponential moving average are needed.

from __future__ import annotations

import torch
from timm.models.vision_transformer import vit_tiny_patch16_224
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import Dataset

from lightly.loss import LeWMLoss
from lightly.models.modules import (
    ActionEncoder,
    LatentDynamicsPredictor,
    LeWMProjectionHead,
)

# Number of frames in one training clip. The predictor sees the first
# num_frames - 1 frames and predicts the embedding of the following frame.
num_frames = 4
image_size = 64
embed_dim = 192
action_dim = 2


class MovingSquareTrajectories(Dataset):
    """Clips of a square that the action pushes around a canvas.

    This stands in for an offline dataset of trajectories so that the example
    runs anywhere. The next frame is fully determined by the current frame and
    the action, which is exactly the structure a world model has to discover.

    Replace this with your own recorded trajectories. Any dataset works as long
    as an item is a clip of frames of shape (T, C, H, W) and the actions of
    shape (T, action_dim) that were taken in those frames.
    """

    def __init__(
        self,
        num_clips: int = 512,
        num_frames: int = 4,
        image_size: int = 64,
        square_size: int = 12,
        max_speed: float = 8.0,
        seed: int = 0,
    ) -> None:
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.image_size = image_size
        self.square_size = square_size
        self.max_speed = max_speed
        self.seed = seed
        self.max_position = image_size - square_size

    def __len__(self) -> int:
        return self.num_clips

    def _render(self, position: Tensor) -> Tensor:
        frame = torch.zeros(3, self.image_size, self.image_size)
        top, left = position.round().long().tolist()
        frame[:, top : top + self.square_size, left : left + self.square_size] = 1.0
        return frame

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        position = torch.rand(2, generator=generator) * self.max_position
        actions = (
            torch.rand(self.num_frames, 2, generator=generator) - 0.5
        ) * self.max_speed

        frames = []
        for t in range(self.num_frames):
            frames.append(self._render(position))
            # actions[t] is the action taken at frame t, so it leads to the
            # frame at t + 1. The loss below relies on this alignment.
            position = (position + actions[t]).clamp(0, self.max_position)
        return torch.stack(frames), actions


class LeWM(Module):
    def __init__(
        self,
        image_size: int = 64,
        embed_dim: int = 192,
        action_dim: int = 2,
        num_frames: int = 4,
    ) -> None:
        super().__init__()
        # Any image encoder works here. LeWM trains it from scratch, but a
        # frozen pretrained encoder can be used instead without changing the
        # predictor or the loss.
        self.backbone = vit_tiny_patch16_224(
            pretrained=False,
            img_size=image_size,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.projection_head = LeWMProjectionHead(
            input_dim=self.backbone.num_features,
            output_dim=embed_dim,
        )
        self.action_encoder = ActionEncoder(action_dim=action_dim, output_dim=embed_dim)
        self.predictor = LatentDynamicsPredictor(
            num_frames=num_frames,
            input_dim=embed_dim,
            hidden_dim=384,
            depth=4,
            num_heads=6,
        )

    def encode(self, frames: Tensor) -> Tensor:
        """Encode frames of shape (B, T, C, H, W) into embeddings (B, T, D)."""
        batch_size, num_steps = frames.shape[:2]
        features = self.backbone(frames.flatten(0, 1))
        emb = self.projection_head(features)
        return emb.unflatten(0, (batch_size, num_steps))

    def predict(self, emb: Tensor, actions: Tensor) -> Tensor:
        """Predict the next embedding for every frame in the context."""
        return self.predictor(emb, action_emb=self.action_encoder(actions))


model = LeWM(
    image_size=image_size,
    embed_dim=embed_dim,
    action_dim=action_dim,
    num_frames=num_frames,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = MovingSquareTrajectories(num_frames=num_frames, image_size=image_size)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

criterion = LeWMLoss(lambda_param=0.1)

# Move loss to correct device because it also contains parameters.
criterion = criterion.to(device)

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

epochs = 10
num_batches = len(dataloader)

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0.0
    for frames, actions in dataloader:
        frames = frames.to(device)
        actions = actions.to(device)

        emb = model.encode(frames)
        # Teacher forcing: the predictor reads the observed embeddings and
        # predicts the embedding one step ahead.
        predicted = model.predict(emb[:, :-1], actions[:, :-1])

        loss = criterion(
            predicted=predicted,
            target=emb[:, 1:],
            embeddings=emb,
        )
        total_loss += loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / num_batches
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
