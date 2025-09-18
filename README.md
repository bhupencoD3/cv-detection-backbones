# CV Detection Backbones

This repository documents my learning process with object detection backbones. The work focuses on running experiments with pretrained and trainable models using PyTorch, Detectron2, and Ultralytics. Each notebook represents a step in understanding how modern detection models are structured, how to use them for inference or training, and how to compare their outputs across frameworks.

---

## Contents So Far

### 1. YOLOv11 – Pretrained Inference

* Notebook: `getting_started_with_yolov11.ipynb`
* Objective: Run YOLOv11 using Ultralytics with pretrained weights.
* Tasks performed:

  * Loaded the pretrained YOLOv11 model.
  * Ran inference on sample images.
  * Observed detection outputs and bounding box quality.
* Notes: This step is about familiarizing with Ultralytics’ interface and the overall pipeline of running a modern YOLO model.

### 2. Faster R-CNN – Pretrained Inference

* Notebook: `fasterrcnn-pretrained.ipynb`
* Objective: Understand Faster R-CNN architecture by running pretrained inference.
* Tasks performed:

  * Used torchvision’s pretrained Faster R-CNN model.
  * Ran inference on images and inspected detected classes and bounding boxes.
* Notes: This marks the starting point with classical region-based detectors, contrasting with YOLO’s single-shot approach.

---

## Planned Work

The repository will gradually include training experiments and evaluations with the following models:

* Faster R-CNN (trained on custom datasets)
* RetinaNet
* SSD
* Full YOLO family (YOLOv8, YOLOv11, etc.)
* Other state-of-the-art detectors as part of comparative analysis

Each addition will include scripts/notebooks for training or inference, dataset details, and evaluation metrics (such as mAP, IoU).

---

## Technical Details

The implementation relies on the following:

* Python 3
* PyTorch
* Detectron2
* Ultralytics YOLO

---

## Purpose of This Repository

This is a structured learning repository. The aim is not to provide production-ready implementations but to document the step-by-step process of:

* Running inference with pretrained detection models.
* Training selected models on datasets.
* Recording and comparing results across frameworks.

---

## License

This repository is licensed under the MIT License.

---

## Author

Bhupen
[LinkedIn](https://www.linkedin.com/in/bhupenparmar/) | [GitHub](https://github.com/bhupencoD3)
