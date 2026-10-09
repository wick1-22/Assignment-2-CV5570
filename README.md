# CV5570 Assignment 2: Object Detection on ACID

This repository contains a three-class object-detection experiment using the ACID dataset. It compares Faster R-CNN, RetinaNet, and RT-DETR-L for Dozer, Dump Truck, and Excavator detection.

## Contents

```text
src/
  data_tools.py       Dataset checks, split creation, and plots
  train_models.py     Training, validation metrics, examples, and comparison plot
  build_report.py     PDF report builder
notebooks/
  assignment2_colab.ipynb   Colab workflow
outputs/
  dataset_analysis/   Dataset summary, split files, and figures
  model_runs/         Per-model metrics and training outputs
reports/               Assignment report
```

Model weights and the ACID images are not stored in Git. The notebook reads the dataset from Google Drive and saves training outputs there.

## Dataset

The assignment dataset is the ACID three-class subset (Excavator, Dozer, and Dump Truck). Download it from the [ACID Google Drive link](https://drive.google.com/uc?id=1Qg_X5FygUMBRTcVFPb0s1fQP-20f8n8O) and extract it in your Google Drive. The notebook expects `3classes.json` and its images in `ACID_3classes/3classes/` beneath the Drive workspace folder. In the notebook's **Drive and file paths** cell, change `DRIVE_BASE` once if you use a different workspace folder; later cells reuse the paths derived from it. Keep the dataset outside Git.

## Run the experiment in Colab

1. Download and extract the ACID dataset into the Drive workspace folder, using the folder structure described above.
2. Open the [notebook in Google Colab](https://colab.research.google.com/github/wick1-22/Assignment-2-CV5570/blob/main/notebooks/assignment2_colab.ipynb) and connect to a runtime.
3. In the **Drive and file paths** cell, change `DRIVE_BASE` only if your Drive workspace folder has a different location or name. The notebook clones this repository automatically; you do not need to download or upload a repository ZIP.
4. Run the setup and dataset-analysis cells. To reproduce the experiments, run each model's training cell. Results are saved under `assignment2_run/outputs/model_runs/` in Drive. Skip training cells when using the saved completed results.
5. If Faster R-CNN's example board is missing but its checkpoint exists, run only the checkpoint-recovery cell; it recreates the examples without retraining.

The repository ZIP available from GitHub's **Code → Download ZIP** contains a snapshot of the tracked repository files. It does not include the dataset or model weights. This Colab workflow clones the repository directly, so no repository ZIP is needed. The final PDF report is included in `reports/`.

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
