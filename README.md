# RANZCR Catheter and Line Position Challenge

Files and codes with the 71st place solution to the [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification).


## Summary

Hospital patients can have catheters and tubes inserted during their admission. If tubes are placed incorrectly, serious health complications can occur later. Using deep learning helps to automate early detection of malpositioned tubes based on X-ray images, which allows to reduce the workload of clinicians and prevent treatment delays.

This project works with a dataset of 30,083 high-resolution chest X-ray images. The images have 11 binary labels indicating normal, borderline or abnormal placement of endotracheal tubes, nasogastric tubes, central venous catheters and swan ganz catheters.

I develop an ensemble of 5+2 CNN models implemented in `PyTorch`. My solution reaches the test AUC of 0.971 and places 71st out of 1,549 competing teams. The diagram below overviews the ensemble. The detailed summary of the solution is provided [this writeup](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226664).

![ensemble](https://i.postimg.cc/c4cPcXng/ranzcr.png)


## Project structure

The project has the following structure:
- `codes/`: `.py` modules with training, inference and data processing functions.
- `notebooks/`: `.ipynb` notebooks for training CNN models and ensembling.
- `input/`: input images (not included due to size constraints, can be downloaded [here](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification)).
- `output/`: model configurations, weights and figures exported from notebooks.


## Working with the repo

My solution can be reproduced in the following steps:
1. Downloading competition data and adding it into the `input/` folder.
2. Running five training notebooks `model_x.ipynb` to obtain weights of base models.
3. Running the ensembling notebook `ensembling.ipynb` to obtain the final prediction.

All `model_x` notebooks have the same structure and differ in model/data parameters. Different versions are included to ensure reproducibility. It is sufficient to inspect one of the PyTorch modeling codes and go through the `functions/` folder to understand the training process. The ensembling code is also provided in this [Kaggle notebook](https://www.kaggle.com/kozodoi/71st-place-ensembling-pipeline/output).

The notebooks are designed to run on Google Colab. More details are provided in the documentation within the notebooks.
