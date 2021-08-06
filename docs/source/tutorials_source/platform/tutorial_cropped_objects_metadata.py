import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightly.active_learning.utils import BoundingBox
from lightly.data import LightlyDataset
from lightly.data.lightly_subset import LightlySubset
from lightly.utils.cropping.crop_image_by_bounding_boxes import crop_dataset_by_bounding_boxes_and_save

DATASET_PATH = "../../datasets/retail_SKU110k/SKU_110k_val_100/valid/images"
CATEGORY_NAMES_PATH = "../../datasets/retail_SKU110k/SKU_110k_val_100/coco_stuff_91_category_names.txt"
OUTPUT_DIR = "../../datasets/retail_SKU110k/SKU_110k_val_100/valid/cropped_images"

x_size = 2048
y_size = 2048
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((x_size, y_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
dataset = LightlyDataset(DATASET_PATH, transform=transform)
dataset = LightlySubset(dataset, dataset.get_filenames()[:2])

model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
#model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
model.eval()



dataloader = DataLoader(dataset, batch_size=2)
predictions = []
with torch.no_grad():
    for x, _, _ in tqdm(dataloader):
        pred = model(x)
        predictions.append(pred)

predictions = [i for sublist in predictions for i in sublist]

class_indices_list_list = [list(prediction["labels"]) for prediction in predictions]
bounding_boxes_list_list = []
for prediction in predictions:
    bounding_boxes_list = []
    for box in prediction["boxes"]:
        x0 = box[0] / x_size
        y0 = box[1] / y_size
        x1 = box[2] / x_size
        y1 = box[3] / y_size
        bounding_boxes_list.append(BoundingBox(x0, y0, x1, y1))
    bounding_boxes_list_list.append(bounding_boxes_list)
confidences_list_list = [list(prediction["scores"]) for prediction in predictions]



cropped_images_list_list = crop_dataset_by_bounding_boxes_and_save(dataset, OUTPUT_DIR, bounding_boxes_list_list, class_indices_list_list)

i=0