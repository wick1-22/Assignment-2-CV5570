# CV5570 Assignment 2: Object Detection on ACID

This repository contains a three-class object-detection experiment using the ACID dataset. It compares Faster R-CNN, RetinaNet, and RT-DETR-L for Dozer, Dump Truck, and Excavator detection.

## Contents

```text
src/
  data_tools.py       Dataset checks, split creation, and plots
  train_models.py     Training, validation metrics, examples, and comparison plot
  build_report.py     PDF and Markdown report builder
notebooks/
  assignment2_colab.ipynb   Colab workflow
outputs/
  dataset_analysis/   Dataset summary, split files, and figures
  model_runs/         Per-model metrics and training outputs
reports/               Assignment report and editable Markdown source
```

Model weights and the ACID images are not stored in Git. The notebook reads the dataset from Google Drive and saves training outputs there.

## Dataset

The assignment dataset is the ACID three-class subset (Excavator, Dozer, and Dump Truck). Download it from the [ACID Google Drive link](https://drive.google.com/uc?id=1Qg_X5FygUMBRTcVFPb0s1fQP-20f8n8O). If Google Drive asks for access, sign in with an account that has permission to view the file. Keep the dataset outside Git; the notebook expects it under `MyDrive/Assignment 2 CV5570/ACID_3classes/3classes/` after extraction.

## Run the experiment in Colab

1. Put the repository ZIP and the ACID three-class dataset under `MyDrive/Assignment 2 CV5570/`.
2. Open `notebooks/assignment2_colab.ipynb`, select a GPU runtime, and run the setup and dataset-analysis cells.
3. Run the Faster R-CNN, RetinaNet, and RT-DETR cells. Each model saves its own results under `assignment2_run/outputs/model_runs/`.
4. If the Faster R-CNN validation example board is missing, run only its checkpoint-recovery cell. It uses `best.pth` to recreate the examples and leaves the saved metrics unchanged. Do not rerun training if the model results already exist.

The PDF and Markdown report are generated from saved outputs in the repository, not in Colab.

The split is seeded (42) and uses 70% training, 20% validation, and 10% test images. Validation metrics are used for the model comparison; the test split is held out.

## Local dataset analysis and report generation

For dataset analysis, install the packages in `requirements.txt`, then run:

```bash
python src/data_tools.py --annotations PATH/TO/3classes.json --images PATH/TO/3classes --output outputs/dataset_analysis
```

To generate the report locally after copying the saved Drive outputs into this repository:

```bash
python src/build_report.py --root .
```

Training uses PyTorch/TorchVision and Ultralytics. For local GPU training, install a PyTorch/TorchVision pair compatible with the available CUDA version before installing the remaining requirements.

## Results and metrics

Each completed model writes `metrics.json`, a loss history and curve, a validation example board, and its training configuration. The comparison reports mAP at IoU 0.5 and averaged over IoU 0.50 to 0.95, class-wise AP, parameter count, validation FPS, and training time. AP is calculated with 101-point interpolated precision-recall. Model weights and the dataset are kept in Drive and are not included in Git.

Faster R-CNN uses SGD with learning rate 0.005. RetinaNet uses learning rate 0.0005 after the initial 0.005 run produced non-finite losses. Both use momentum 0.9 and weight decay 0.0005. RT-DETR uses the optimizer settings saved by Ultralytics. The three models share the saved split, 640-pixel image size, 10-epoch budget, and disabled augmentation.

The report builder uses saved metrics, plots, loss histories, and qualitative example boards; rebuilding the report does not retrain the models. The course policy on AI assistance and disclosure should be followed when preparing the final submission.

## References

- Ren et al., 'Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,' NeurIPS 2015. https://papers.nips.cc/paper_files/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html
- Lin et al., 'Focal Loss for Dense Object Detection,' ICCV 2017. https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_Dense_Object_Detection_ICCV_2017_paper.pdf
- Zhao et al., 'DETRs Beat YOLOs on Real-time Object Detection,' CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html
