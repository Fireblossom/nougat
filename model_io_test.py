from mmdet.apis import init_detector, inference_detector
from nougat.model import NougatModel
from nougat.utils.checkpoint import get_checkpoint

checkpoint = get_checkpoint(model_tag="0.1.0-base")
nougat_model = NougatModel.from_pretrained(checkpoint)

#config_file = 'conifg_mmdet.py'

#model = init_detector(config_file, device='cpu')  # or device='cuda:0'
#inference_detector(model, 'demo/demo.jpg')

from torchinfo import summary
import torch

summary(nougat_model.encoder, input_size=(2, 3, 896, 672), device='cpu')
#summary(model.backbone, input_size=(2, 3, 896, 672), device='cpu')
t = nougat_model.encoder(torch.zeros([2, 3, 896, 672]), reshape=True)
pass