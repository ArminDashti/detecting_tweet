import numpy as np
import os
from PIL import Image
import json
import torchvision
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from detection.engine import evaluate, train_one_epoch
from detection import utils


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
  
  class twt(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "boxes"))))
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        box_path = os.path.join(self.root, "boxes", self.boxes[idx])
        img = Image.open(img_path).convert("RGB")
        with open(box_path) as f:
            _json = json.load(f)
        box_point = _json['shapes'][0]['points']
        box_point_list = []
        box_point_list += box_point[0]
        box_point_list += box_point[1]
        box_point_list = np.reshape(box_point_list, (1, 4))
        label = int(_json['shapes'][0]['label'])
        box_point = torch.as_tensor(box_point_list, dtype=torch.float32)
        _label = [[]]
        _label[0] = 1
        image_id = torch.tensor([idx])
        masks = torch.tensor(0)
        area = (box_point[:, 3] - box_point[:, 1]) * (box_point[:, 2] - box_point[:, 0])
        iscrowd = torch.ones((1), dtype=torch.int64)

        label = torch.as_tensor(_label, dtype=torch.int64)
        target = {'boxes':box_point, 'labels':label,
                  'image_id':image_id, 'area':box_point, 'iscrowd':iscrowd}
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_dir = "C:\\Users\\Armin\\Desktop\\twitter_dataset"
num_classes = 2
dataset = twt(dataset_dir, transforms=get_transform(train=True))
dataset_test = twt(dataset_dir, transforms=get_transform(train=False))

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-3])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-3:])
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)
