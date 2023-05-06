from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode, _OFF_WHITE, _KEYPOINT_THRESHOLD
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import random
import matplotlib.pyplot as plt

def plot_samples(dataser_name , n=1):
    dataset_custom= DatasetCatalog.get(dataser_name)
    dataset_custom_metadata= MetadataCatalog.get(dataser_name)

    for s in random.sample(dataset_custom, n):
        img=cv2.imread(s["file_name"])
        v= Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v= v.draw_dataset_dict(s)
        plt.figure(figsize=(2,4))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path ,checkpoint_url,train_dataset_name,test_dataset_name , num_classes, device, output_dir):
    cfg= get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS= model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERWS =2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 2800
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = "mps"
    cfg.OUTPUT_DIR = output_dir

    return  cfg


def on_image (image_path, perdictor ):
    im = cv2.imread(image_path)







