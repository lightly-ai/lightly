from typing import Dict, Optional, Any

import pytorch_lightning.callbacks as cb


class CustomModelCheckpoint(cb.ModelCheckpoint):
    """Custom implementation of the Pytorch Lightning ModelCheckpoint.

    Attributes:
        checkpoint_fmt:
            String which determines the format of the checkpoint name.
            Default leads to, e.g.

            >>> epoch_10.ckpt
    """

    def __init__(self, checkpoint_fmt: str = 'lightly_epoch_{epoch}.ckpt'):
        # use default initialization to prevent compatability
        # issues in case pytorch_lightning changes attributes
        super(CustomModelCheckpoint, self).__init__()
        self.checkpoint_fmt = checkpoint_fmt

    def format_checkpoint_name(
        self, epoch: int, metrics: Dict[str, Any], ver: Optional[int] = None
    ) -> str:
        """Formats the format string to an actual checkpoint name.

        Args:
            epoch:
                Training epoch of the checkpoint.

        """
        # use custom template to prevent the = in the checkpoint name
        return self.checkpoint_fmt.format(epoch=epoch)
