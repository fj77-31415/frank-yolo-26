# frank-yolo-26
YOLO Model, training and test instructions, etc.

For statistics, analysis, basic overview, and issues I ran into, see below:
(Final Presentation)[https://docs.google.com/presentation/d/1GAdOyTbnxtWnAKf4xW30D_5jbgR6zHZnz4NtlhCe0-s/edit?usp=sharing]

To set up Python virtual environment:

`python3 -m venv .venv`

`source .venv/bin/activate`

Install ultralytics:

`pip install ultralytics`

(Not my actual workflow. See ipynb for actual details) But to run a basic train for fun (in Kaggle; don't use runpod credits), run:
python3 yolo-starter.py

Interpreting post-training summary:
- Box(...) is metrics for bounding-box detections
- P (Precision) = TP / (TP + FP). Of all predicted boxes, how many are correct 
  (above the IoU threshold and correct class). High P = few false positives.
  - TP: True Positive
  - FP: False Positive
  - FN: False Negative
- R (Recall) = TP / (TP + FN). Of all ground-truth (GT) objects, how many were found. 
  High R = few missed objects
  - GT: labeled/true annotations in dataset (bounding boes + class labels), used as 
    reference when evaluating predictions
- mAP50 = mean Average Precision at IoU >= 0.50. AP is computed from the 
  precision–recall curve for each class; mAP50 is the average of those APs 
  across classes using IoU=0.5 (a relatively lenient match threshold).
  - IoU (Intersection over Union): area(overlap of predicted box & ground-truth 
    box) / area(union of the 2 boxes)
    - helps decide whether prediction matches a GT (e.g. IoU >= 0.5 for AP@50)
- mAP50-95 = mean AP averaged over multiple IoU thresholds 0.50:0.05:0.95 (i.e., 
  0.50, 0.55, …, 0.95). This is the COCO-style stricter/robust metric that 
  penalizes poor localization more heavily.

- Example: Box 0.05 0.22 0.0977 0.0511 means precision≈5%, recall≈22%, 
  AP@0.5≈9.8%, AP@0.5:0.95≈5.1% for that class.

- Low P, high R → many detections but lots of false positives (model is permissive)
- High P, low R → few false positives but many misses (model is conservative).
- mAP50 >> mAP50-95 → localization is coarse (works at loose IoU but fails at tighter IoUs).
- Use mAP50-95 as the main overall quality metric; mAP50 is useful for quick/lenient checks.


For running Google Colab in VS Code:
- "Select Kernel" in top right
- Select Another Kernel > Colab > New Colab Server > GPU > Python3


To run inference on some set of images, comment this section out:

```
results = model.predict(
    source="path/to/images_or_folder",
    imgsz=640,
    conf=0.25,
    save=True
)
```

Or straight from terminal command:

`python -m ultralytics.yolo detect predict model=yolo11n.pt source="path/to/images" --imgsz 640 --conf 0.25 --save`


## File Overview ###
### docs.txt
Just repeat of what's above here. My personal notes of 
### img_folder_cleaner.py
Script to clean images in a given zip file. Removes 0-byte imgs, jsons, etc. and packages them in a new zip.
### post-train.py
Runs final validation metrics against test dataset; verifies model accuracy before spitting out weights to be used for inference.
### remove_backgrounds.py
Given a folders of training images and labels, this script removes background images (ones without labels). Can be useful if there are lots of images without the labels we care about (humans and tents).
Background images are useful in training since they let model learn what's NOT a human, but sometimes too many images are present and don't add up to much of a difference, especially when we'd rather over-classify than under-classify.
### run_sahi.py
Executes Sliced Automatic Hyper-Inference (SAHI). Cuts raw drone phots into smaller tiles at runtime. Runs finetuned.pt on each tile, merges overlapping bounding boxes to prevent duplicate classifications, and outputs the final full-rez image with target bounding boxes. 
### slice_dataset.py
Takes folder of training images and labels, splits them into smaller image chunks for SAHI, and gives a new folder of datasets of chunks of the original images. Use t
### yolo-starter.py
Starter program with basic training functions I used to get the hang of things.
### temp-runpod-script.ipynb
My personal notes on how to train models in runpod. Should be idiot-proof but dm me if something is unclear. Hyperparameters deduced thru trial-and error and with help of Gemini. Hyperparam. explanations given in comments there, these are likely quite good and doubt it's cost-effective to further adjust them. 
### best.pt
DO NOT USE. (unless you have good reason to). This is best model BEFORE FINETUNING; can possibly use this as base for future fine-tuning and fine-tuning experiments. 
### finetuned.pt
USE THIS. (unless you have good reason not to). This was finetuned on our own prop images of mannequins laying in brown dirt (as close as possible to irl environment in OK). Ensured that this model recognizes outlines of relatively small objects of interest regardless of color contrast. Trigger-happy with labeling tents as humans but that's preferable to the alternative.

## My Training Workflow
1) Used Roboflow to manually annotate the images that we took. Sorted into Train, Val, and Test folders. Used standard box, not polygon/auto-classify, since I wanted it to be as accurate as possible and YOLO uses boxes, not polygons. 
(My Dataset)[https://app.roboflow.com/franks-workspace-8dch5/customaerialpics/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true]
2) Ran slicing script to cut master images into tiles
3) Deleted bunch of empty background tiles
4) Duplicated tiles with humans/tents in them many times to make sure we can balance out the huge base-trained images
5) Fine-tuned YOLO model on the sliced dataset
6) Ran SAHI inference on new high-rez images
