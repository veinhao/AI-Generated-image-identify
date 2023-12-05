from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# custom_dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.classes = ['fake_train', 'real_train']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 从文件夹名中加载所有图像文件，并为它们分配标签
        self.image_paths = []
        self.image_labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.main_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.image_labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.image_labels[idx]
        image = Image.open(image_path).convert('RGB')  # 确保图像为RGB格式
        if self.transform:
            image = self.transform(image)
        return image, label
