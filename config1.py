# config.py
from torch import optim

from model.model_head.SVM import LinearSVM
from train.util.lion import Lion
from model.model_head.FC4 import FCHead
from torch import nn
from model.model_head.ATTENTION import SimpleSelfAttention
from train.util.svm_hinge_loss import hinge_loss
from model.fix_VIT import ViTForBinaryClassification
from model.resnet34 import resnet34


# TODO 将经常变更的参数注明


class DataConfig:

    ################################################################################
    # 必填:修改这里需要对应dataconfig里的DATAPATH '/home/nano/Downloads/baidu/test/'
    DATAPATH = '/home/nano/Downloads/baidu/Identification_AI_Generated_Datasets'

    VALIDATION_SPLIT = 0.1
    ################################################################################

    BATCH_SIZE = 64
    IMG_SIZE = 224
    SHUFFLE = True
    RANDOM_SEED = 42
    NUM_WORKERS = 4


class ModelConfig:

    ################################################################################
    MODEL_NAME = '/home/nano/Downloads/hugface/vit'
    # RESNET34
    # RESNET34 = '/home/nano/Downloads/checkpoint/resNet34.pth'
    MODEL_HEAD = FCHead  # FCHead connect with MODEL_VERSION
    ################################################################################
    DEVICE = 'cuda'
    PRETRAINED = True
    NUM_CLASSES = 2  # 对于二分类问题
    HIDDEN_SIZE = 1280  # [1, 257, 1280] 1792

    MODEL = ''  # 'two path msa'
    NUM_FEATURE = 1792
    NUM_HEAD = 8

    MODELA = ViTForBinaryClassification
    # MODELB = resnet34


class TrainConfig:
    ################################################################################
    EPOCHS = 20
    OPT = optim.Adam  # 可选optim.Adam; Lion
    SAVE_PATH = 'data/checkpoint/'
    IS_LOAD = False  # 是否加载全连接层的checkpoint
    LOAD_PATH = [
                  #'/home/nano/Downloads/checkpoint/resNet34.pth',
                 # '/home/nano/PycharmProjects/AI-Generated-image-identify/data/checkpoint/test--VITH-RES34-FCh512-FC512256-FC256128-FC12864-FC642-Adam/dataset_test-best-f1_score_0.0-11-14-16:11.pth'
                 ]
    DATASET = 'test'  # 必填:修改这里需要对应dataconfig里的DATAPATH
    MODEL_VERSION = {
        0: 'VITH-FCh512-FC512128-FC1282-Adam',  # FC 全连接层
        1: 'VITH-FCh512-FC512128-FC1282-OPTlion',
        2: 'VITH-FCh512-FC512256-FC256128-FC12864-FC642-Adam',
        3: 'VITH-SVM-Adam',
        4: 'VITH-ATTENTION-ADAM',
        5: 'RES18-FC512256-FC25664-FC642-Adam',
        6: 'RES34-Adam',
        7: 'VITH-RES34-FCh512-FC512256-FC256128-FC12864-FC642-Adam'
    }
    MODEL_VERSION_INDEX = 2
    LOG_PATH = 'data/log/'

    # onlyFC = False  # only load full connection checkpoint
    ################################################################################

    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    DEVICE = 'cuda'

    VALIDATION_SPLIT = DataConfig.VALIDATION_SPLIT

    CRITERION = nn.CrossEntropyLoss()  # 当ModelConfig 的 MODEL_HEAD使用SVM时，必须使用hinge_loss

    # RESNET34 = LOAD_PATH[0]

