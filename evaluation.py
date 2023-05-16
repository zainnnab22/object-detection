from multiprocessing import freeze_support
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from detectron2.utils.logger import setup_logger
import tensorflow as tf

# from test import cfg
setup_logger()
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

from detectron2.engine import DefaultPredictor
import os
import pickle

from unit import *

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = "./output1/object_detection1"
num_classes = 3
device = "cuda"

train_dataset_name="lp_train"
train_images_path= "train500"
train_json_annot_path ="train.json"

test_dataset_name="lp_test"
test_images_path="test94"
test_json_annot_path= "test94.json"

cfg_save_path="ISG_cfg.pickle"


register_coco_instances(name=train_dataset_name, metadata={}, json_file=train_json_annot_path,
                        image_root=train_images_path)

register_coco_instances(name=test_dataset_name, metadata={}, json_file=test_json_annot_path,
                        image_root=test_images_path)


# plot_samples(dataser_name=train_dataset_name,n=2)


def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device,
                        output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # trainer.train()

    #config = tf.ConfigProto(device_count={'GPU': 0})

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("lp_test", tasks=["bbox","segm"], distributed=False, output_dir="./output1")
    val_loader = build_detection_test_loader(cfg, "lp_test")

    with tf.device('/cpu:0'):
        print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == '__main__':
    main()
# if __name__ == '__main__':
#     freeze_support()
