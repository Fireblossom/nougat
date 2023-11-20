import os
import json
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from nougat.model import NougatModel
from nougat.utils.checkpoint import get_checkpoint
import torch
from vision_utils.engine import train_one_epoch, evaluate
import vision_utils.utils as utils
from nougat.transforms import train_transform, test_transform


class LATEXRainbowDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "imgs"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "boxes"))))

    def __getitem__(self, idx):
        # load images and boxes
        img_path = os.path.join(self.root, "imgs", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        boxes_path = os.path.join(self.root, "boxes", self.boxes[idx])
        boxes = json.load(open(boxes_path))
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx # torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        self.transforms = train_transform
        if self.transforms is not None:
            transformed = self.transforms(img, target)
            img = transformed["image"]
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0,4), dtype=torch.float32)
        return img, target

    def __len__(self):
        return len(self.imgs)

def init_swin_model_from_nougat(num_classes):
    checkpoint = get_checkpoint(model_tag="0.1.0-base")
    nougat_encoder = NougatModel.from_pretrained(checkpoint).encoder
    nougat_encoder.reshape = True

    backbone = BackboneWithFPN(nougat_encoder, {}, in_channels_list=[256, 512, 1024, 1024], out_channels=256)
    # IntermediateLayerGetter backbone cannnot access subsubmodules
    backbone.body = nougat_encoder

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=(((0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0),
                        (0.5, 1.0, 2.0)))
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=672,
        max_size=896,
    )

    # test forward
    """model.eval()
    x = [torch.rand(3, 896, 672), torch.rand(3, 896, 672)]
    model(x)"""
    return model


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = LATEXRainbowDataset('PennFudanPed', train_transform)
    dataset_test = LATEXRainbowDataset('PennFudanPed', test_transform)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = init_swin_model_from_nougat(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()