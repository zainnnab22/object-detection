from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
import os
import pickle

from unit import *
cfg_save_path="OD_cfg.pickle"


with open(cfg_save_path,'rb') as f:
    cfg = pickle.load(f)


cfg.MODEL.WEIGHTS= os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =0.5

predictor= DefaultPredictor(cfg)

image_path="./outputs/coco16.jpg"
on_image(image_path, predictor)

# im = cv2.imread(image_path)
# outputs=  predictor(im)
# v=Visualizer(im[ :, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
# v=v.draw_instance_predictions(outputs["instances"].to("cpu"))
#
# cv2.imshow("result", v.get_image()[:, :, ::-1])
# cv2.waitKey(0)  # waits until a key is pressed
# cv2.destroyAllWindows()
