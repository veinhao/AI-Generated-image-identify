import os
import torch
from torch import nn, optim
from tqdm import tqdm
from torchmetrics import F1Score
import time
import numpy as np
from torchvision import transforms
from PIL import Image

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = config.CRITERION
        self.optimizer = config.OPT(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.config = config
        self.f1_score = F1Score(num_classes=2, average='weighted', task='binary').to(config.DEVICE)
        self.last_f1 = -1
        self.training_set = config.DATASET
        self.load = config.IS_LOAD
        self.device = self.config.DEVICE

        self.F1_record = []
        # self.minimizer = eval(config.MINIMIZER)

    def train_epoch(self):
        if self.config.VALIDATION_SPLIT == 1:
            return

        # training mode
        self.model.train()

        train_loss = 0.0
        correct = 0
        total = 0
        self.f1_score.reset()  # 重置F1-score计算器

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)  # + a*ssim
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.f1_score.update(predicted, targets)
            f1 = self.f1_score.compute()

            # 更新进度条
            pbar.set_description(
                f"Loss: {train_loss / (batch_idx + 1):.3f} | "
                f"Acc: {100. * correct / total:.3f}% | "
                f"F1: {f1:.3f}"
            )

            # 在所有批次完成后，获取整个周期的平均F1-score
        final_f1 = self.f1_score.compute()
        train_loss = train_loss / total
        train_acc = 100. * correct / total
        f1 = final_f1

        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | F1-score: {f1: .3f} | ")
        return f1

    def validate_epoch(self):

        if self.config.VALIDATION_SPLIT == 0:
            return

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        self.f1_score.reset()

        with torch.no_grad():
            # Wrap `self.val_loader` with `tqdm` for a progress bar
            for inputs, targets in tqdm(self.val_loader, desc='Validation Progress'):
                inputs, targets = inputs.to(self.config.DEVICE), targets.to(self.config.DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                self.f1_score.update(predicted, targets)
            f1_val = self.f1_score.compute()

        val_loss = val_loss / total
        val_acc = 100. * correct / total

        print(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}% | F1-val-score: {f1_val: .3f} \n")
        return f1_val

    def test_speed_per_image(self):
        if self.load:
            self.load_checkpoint()
        self.model.eval()

        # 图像处理流程
        preprocess = transforms.Compose([
            # 添加必要的预处理步骤，例如：
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        total_time = 0.0
        total_images = 0
        start_time = time.time()

        folder_path = self.config.DATAPATH
        # 打开文件用于写入
        t1 = time.localtime()
        save_directory = f'output/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        file = f'山东中医药大学_王文浩_{t1.tm_hour}{t1.tm_min}_score.txt'
        file_path = os.path.join(save_directory, file)
        with open(file_path, 'w') as file:
            with torch.no_grad():
                for image_name in tqdm(sorted(os.listdir(folder_path)), desc='Testing Speed Per Image'):
                    if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        image_path = os.path.join(folder_path, image_name)
                        image = Image.open(image_path)
                        input_tensor = preprocess(image)
                        input_batch = input_tensor.unsqueeze(0).to(self.config.DEVICE)

                        # 开始计时

                        # 进行预测
                        outputs = self.model(input_batch)
                        # print(outputs)
                        # 结束计时并累加

                        # total_time += t
                        # total_images += 1

                        # 获取预测标签
                        # _, predicted = torch.max(outputs, 1)
                        _, predicted = outputs.max(1)
                        # 写入图片名和预测标签
                        file.write(f"{image_name} {predicted[0].item()}\n")

            total_time = time.time() - start_time
            # 计算每秒可以处理的图像数量并写入文件
            # images_per_second = total_images / total_time
            file.write(
                f"测试程序运行总时间: {total_time:.2f} 秒. \n")

        print(f"测试程序运行总时间: {total_time:.2f} \n")


    def train(self):

        if self.load:
            self.load_checkpoint()

        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch: {epoch + 1}/{self.config.EPOCHS} | ")

            f1_tr = self.train_epoch()

            f1_val = self.validate_epoch()

            if f1_val:
                f1 = f1_val
            else:
                f1 = f1_tr

            if self.last_f1 == -1 or self.last_f1 <= f1:
                if self.config.VALIDATION_SPLIT < 1:  # 只进行验证不保留参数
                    if 'VITH-RES34' in self.config.MODEL_VERSION[self.config.MODEL_VERSION_INDEX]:
                        self.save_checkpoint_vit_res(epoch, f1)
                    else:
                        self.save_checkpoint(epoch, f1)

                    self.last_f1 = f1

            self.F1_record.append(f1)
        self.save_f1()

    def save_checkpoint(self, epoch, f1_score):
        print("\n==== 正在保存模型的最佳参数 ====")
        # 只提取全连接层的状态字典
        fc_state_dict = {k: v for k, v in self.model.model_head.state_dict().items() if 'fc' or 'classifier' in k}
        state = {
            'epoch': epoch,
            'model_state': fc_state_dict,
            'optimizer_state': self.optimizer.state_dict(),
        }
        t = time.localtime()
        f1_4 = np.round(f1_score.item(), 4)
        # 创建以数据集和模型版本命名的文件夹路径
        directory_name = f"{self.config.DATASET}--{self.config.MODEL_VERSION[self.config.MODEL_VERSION_INDEX]}"
        # 确保该文件夹存在
        save_directory = os.path.join(self.config.SAVE_PATH, directory_name)
        os.makedirs(save_directory, exist_ok=True)
        # 创建文件路径
        filename = f"dataset_{self.training_set}-best-f1_score_{f1_4}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}.pth"
        path = os.path.join(save_directory, filename)
        # 保存状态字典
        torch.save(state, path)

        print(f"\n==== 保存完成：{path} ====")

    def save_checkpoint_vit_res(self, epoch, f1_score):
        print("\n==== 正在保存模型的最佳参数 ====")
        # 获取MSA和模型头部的状态字典
        msa_state_dict = {k: v for k, v in self.model.msa.state_dict().items()}
        model_head_state_dict = {k: v for k, v in self.model.model_head.state_dict().items()}

        resnet34_state_dict = {k: v for k, v in self.model.model_b.state_dict().items()}

        state = {
            'epoch': epoch,
            'msa_state': msa_state_dict,
            'model_head_state': model_head_state_dict,
            'optimizer_state': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'resnet34': resnet34_state_dict
        }

        t = time.localtime()
        f1_4 = np.round(f1_score.item(), 5)
        # 创建以数据集和模型版本命名的文件夹路径
        directory_name = f"{self.config.DATASET}--{self.config.MODEL_VERSION[self.config.MODEL_VERSION_INDEX]}"
        # 确保该文件夹存在
        save_directory = os.path.join(self.config.SAVE_PATH, directory_name)
        os.makedirs(save_directory, exist_ok=True)
        # 创建文件路径
        filename = f"dataset_{self.training_set}-best-f1_score_{f1_4}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}.pth"
        path = os.path.join(save_directory, filename)
        # 保存状态字典
        torch.save(state, path)

        print(f"\n==== 保存完成：{path} ====")


    def save_f1(self):
        t = time.localtime()
        filename = f"dataset_{self.training_set}-best-f1_score-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}:{t.tm_min}.pth"
        log_path = os.path.join(self.config.LOG_PATH, filename)

        # 保存F1分数
        torch.save(self.F1_record, log_path)

    def load_checkpoint(self):
        print("\n==== 正在加载检查点 ====")

        # 遍历所有检查点路径
        for load_path in self.config.LOAD_PATH:
            # 检查文件是否存在
            if not os.path.isfile(load_path):
                print(f"警告：文件 '{load_path}' 不存在，跳过此检查点。")
                continue

            # 加载检查点数据
            checkpoint = torch.load(load_path, map_location=self.device)

            if 'resnet34' in checkpoint:
                self.model.model_b.load_state_dict(checkpoint['resnet34'], strict=True)

            # 更新模型的MSA和模型头部状态
            if 'msa_state' in checkpoint and 'model_head_state' in checkpoint:
                self.model.msa.load_state_dict(checkpoint['msa_state'], strict=True)
                self.model.model_head.load_state_dict(checkpoint['model_head_state'], strict=True)

            # 如果优化器的结构在检查点中存在，也加载优化器的状态
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            if 'model_state' in checkpoint:
                self.model.model_head.load_state_dict(checkpoint['model_state'], strict=True)

            print(f"模型的状态已从 '{load_path}' 中加载。")

        print("\n==== 所有检查点加载完毕 ====")

    def adjust_learning_rate(self, epoch):
        # 这里是一个简单的学习率衰减示例，您可以根据需要调整
        new_lr = self.config.LEARNING_RATE * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

