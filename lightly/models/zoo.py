""" Lightly Model Zoo """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

ZOO = {

    'resnet-9/simclr/d16/w0.0625':
    'https://storage.googleapis.com/models_boris/whattolabel-resnet9-simclr-d16-w0.0625-i-ce0d6bd9.pth',

    'resnet-9/simclr/d16/w0.125':
    'https://storage.googleapis.com/models_boris/whattolabel-resnet9-simclr-d16-w0.125-i-7269c38d.pth',

    'resnet-18/simclr/d16/w1.0':
    'https://storage.googleapis.com/models_boris/whattolabel-resnet18-simclr-d16-w1.0-i-58852cb9.pth',

    'resnet-18/simclr/d32/w1.0':
    'https://storage.googleapis.com/models_boris/whattolabel-resnet18-simclr-d32-w1.0-i-085d0693.pth',

    'resnet-34/simclr/d16/w1.0':
    'https://storage.googleapis.com/models_boris/whattolabel-resnet34-simclr-d16-w1.0-i-6e80d963.pth',

    'resnet-34/simclr/d32/w1.0':
    'https://storage.googleapis.com/models_boris/whattolabel-resnet34-simclr-d32-w1.0-i-9f185b45.pth'

}

def checkpoints():
    """Returns the Lightly model zoo as a list of checkpoints.

    Checkpoints:
        ResNet-9:
            SimCLR with width = 0.0625 and num_ftrs = 16
        ResNet-9:
            SimCLR with width = 0.125 and num_ftrs = 16
        ResNet-18:
            SimCLR with width = 1.0 and num_ftrs = 16
        ResNet-18:
            SimCLR with width = 1.0 and num_ftrs = 32
        ResNet-34:
            SimCLR with width = 1.0 and num_ftrs = 16
        ResNet-34:
            SimCLR with width = 1.0 and num_ftrs = 32

    Returns:
        A list of available checkpoints as URLs.

    """
    return [item for key, item in ZOO.items()]

