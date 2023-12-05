from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
from config import DataConfig, ModelConfig, TrainConfig
from data.dataloader import create_data_loaders
from model.fix_VIT import ViTForBinaryClassification
from model.model import ClassModel
from train.Trainer import Trainer
from model.resnet18 import RES18
from model.resnet34 import resnet34


image_test_path = 'data/dataset/image-test/cat.jpg'
local = '/home/nano/Downloads/hugface/vit'

if __name__ == '__main__':
    # 配置
    datac = DataConfig()
    modelc = ModelConfig()
    trainc = TrainConfig()

    # 数据
    train_loader = None
    val_loader = None
    test_loader = None
    if 0 < datac.VALIDATION_SPLIT < 1:  # train val
        train_loader, val_loader = create_data_loaders(datac)
    elif datac.VALIDATION_SPLIT == 0:  # train
        train_loader = create_data_loaders(datac)
    elif datac.VALIDATION_SPLIT == 1:                              # VAL
        val_loader = create_data_loaders(datac)
    elif datac.VALIDATION_SPLIT == -1:
        test_loader = create_data_loaders(datac)

    # 模型
    # model = ViTForBinaryClassification(modelc)
    model = ClassModel(modelc)
    # 训练
    trainer = Trainer(model, train_loader, val_loader, test_loader, trainc)
    if 0 <= datac.VALIDATION_SPLIT <= 1:
        trainer.train()
    else:
        trainer.test_speed_per_image()
    # trainer.test_speed_per_image()
    # print(1)

