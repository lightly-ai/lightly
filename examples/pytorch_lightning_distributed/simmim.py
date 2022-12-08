import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data.collate import MAECollateFunction # Same collate as MAE
from lightly.models import utils
from lightly.models.modules import masked_autoencoder


class SimMIM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        decoder_dim = vit.hidden_dim
        vit = torchvision.models.vit_b_32(pretrained=False)
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # same backbone as MAE
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit) 

        # the decoder is a simple linear layer
        self.decoder = nn.Linear(vit.hidden_dim, vit.patch_size ** 2 * 3)

        # L1 loss as paper suggestion
        self.criterion = nn.L1Loss()
        

    def forward_encoder(self, images, batch_size, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        tokens = self.backbone.images_to_tokens(images, prepend_class_token=True)
        tokens_masked = utils.mask_at_index(tokens, idx_mask , self.mask_token)
        return self.backbone.encoder(tokens_masked)

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)

    def training_step(self, batch, batch_idx):
        images, _, _ = batch

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        
        # Encoding...
        x_encoded = self.forward_encoder(images, batch_size, idx_mask)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)

        # Decoding...
        x_out = self.forward_decoder(x_encoded_masked)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_out, target)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim


model = SimMIM()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# we ignore object detection annotations by setting target_transform to return 0
pascal_voc = torchvision.datasets.VOCDetection(
    "/home/ubuntu/datasets/pascal_voc", download=False, target_transform=lambda t: 0
)
dataset = LightlyDataset.from_torch_dataset(pascal_voc)
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = MAECollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

gpus = torch.cuda.device_count()

# Train with DDP on multiple gpus. Distributed sampling is also enabled with 
# replace_sampler_ddp=True.
trainer = pl.Trainer(
    max_epochs=10, 
    gpus=gpus,
    strategy='ddp',
    replace_sampler_ddp=True,
)
trainer.fit(model=model, train_dataloaders=dataloader)
