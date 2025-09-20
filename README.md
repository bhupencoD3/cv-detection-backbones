# Object Detection Experiments Repository

This repository documents a series of experiments on object detection models, using both pretrained and custom training workflows. The work covers YOLOv8, Faster R-CNN (pretrained COCO weights), and a custom-trained Faster R-CNN on the Hard Hat Workers dataset. The aim is to provide reproducible pipelines for training, evaluation, and analysis of object detection tasks.

---

## 1. YOLOv8 – Pretrained Model Inference

* Notebook: `YOLOv8_COCO_Pretrained/yolov8_pretrained.ipynb`
* Model: YOLOv8 pretrained on MS COCO
* Tasks performed:

  * Loaded pretrained YOLOv8 model from Ultralytics
  * Performed inference on sample images
  * Visualized detection outputs with bounding boxes and confidence scores
* Observations:

  * Detected common COCO objects reliably
  * Very fast inference speed compared to two-stage models

---

## 2. YOLOv8 – Custom Training on Helmet Dataset

* Notebook: `YOLOv8_Custom_Training/yolov8_custom_training.ipynb`
* Dataset: Hard Hat Workers (Roboflow), containing classes `head`, `helmet`, `person`
* Tasks performed:

  * Prepared dataset in YOLOv8 format
  * Configured YOLOv8 training with custom classes
  * Trained on Kaggle GPU runtime
  * Evaluated model performance using precision, recall, and mAP metrics
* Observations:

  * Model converged rapidly within a few epochs
  * Demonstrated strong performance on helmet detection task
  * Dataset imbalance impacted per-class performance

---

## 3. Faster R-CNN – Pretrained COCO Weights

* Notebook: `FasterRCNN_COCO_Trained_Model/fasterrcnn_coco_pretrained.ipynb`
* Model: Faster R-CNN with ResNet-50 FPN backbone, pretrained on MS COCO
* Tasks performed:

  * Loaded pretrained Faster R-CNN from torchvision
  * Ran inference on sample images
  * Visualized bounding box predictions
* Observations:

  * Achieved strong detection quality across COCO classes
  * Slower inference compared to YOLOv8, but more accurate bounding boxes

---

## 4. Faster R-CNN – Custom Training (Detectron2, Hard Hat Workers Dataset)

* Notebook: `FasterRCNN_COCO_Trained_Model/custom-training-detectron2.ipynb`
* Objective: Train Faster R-CNN (ResNet-50 FPN backbone) on the Hard Hat Workers dataset using Detectron2
* Dataset: Hard Hat Workers (Roboflow), COCO annotation format, 3 classes (`head`, `helmet`, `person`)
* Tasks performed:

  * Registered dataset splits (`train`, `val`, `test`) with Detectron2’s COCO API
  * Visualized training samples with bounding boxes
  * Configured Faster R-CNN R50-FPN (3x) pretrained on COCO
  * Fine-tuned for 1000 iterations using Detectron2’s `DefaultTrainer`
  * Saved model checkpoints and evaluated on test split with `COCOEvaluator`
* Observations:

  * Training loss dropped consistently (from \~1.6 → \~0.5)
  * Evaluation showed strong detection of `helmet` and `head` categories
  * `Workers` class not detected due to missing annotations in test split
  * Training completed in \~7 minutes on Kaggle T4 GPU

---

## Conclusion

This repository presents end-to-end workflows for both single-stage (YOLOv8) and two-stage (Faster R-CNN) detectors. Pretrained inference, custom dataset fine-tuning, and evaluation pipelines are all included. The experiments demonstrate trade-offs between inference speed and accuracy, as well as the impact of dataset quality and labeling consistency on performance.

At this stage, the repository is considered complete, with reproducible notebooks for object detection workflows using both YOLO and Faster R-CNN.


---

## License

This project is licensed under the MIT License.

---

## Author

**Bhupen**

* [LinkedIn](https://www.linkedin.com/in/bhupenparmar/)
* [GitHub](https://github.com/bhupencoD3)