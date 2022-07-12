from collections import OrderedDict
import torch
import lightly

def load_ckpt(ckpt_path, model_name='resnet-18', model_width=1):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    state_dict = OrderedDict()
    for key, value in ckpt['state_dict'].items():
        if ('projection_head' in key) or ('backbone.7' in key):
            continue
        state_dict[key.replace('model.backbone.', '')] = value
    
    resnet = lightly.models.ResNetGenerator(name=model_name, width=model_width)
    model = torch.nn.Sequential(
        lightly.models.batchnorm.get_norm_layer(3, 0),
        *list(resnet.children())[:-1],
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        raise RuntimeError(
            f'It looks like you tried loading a checkpoint from a model that is not a {model_name} with width={model_width}! '
            f'Please set model_name and model_width to the lightly.model.name and lightly.model.width parameters from the '
            f'configuration you used to run Lightly. The configuration from a Lightly worker run can be found in output_dir/config/config.yaml'
        )
    return model

# loading the model
model = load_ckpt('output_dir/lightly_epoch_1.ckpt')


# example usage
image_batch = torch.rand(16, 3, 224, 224)
out = model(image_batch)
print(out.shape) # prints: torch.Size([16, 512, 28, 28])


# creating a classifier from the pre-trained model
num_classes = 10
classifier = torch.nn.Sequential(
    model,
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(1),
    torch.nn.Linear(512, num_classes) # use 2048 instead of 512 for resnet-50
)

out = classifier(image_batch)
print(out.shape) # prints: torch.Size(16, 10)
