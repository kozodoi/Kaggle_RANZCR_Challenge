# RANZCR Catheter and Line Position Challenge

The top-5% solution to the [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification) on Kaggle.

![sample](https://i.postimg.cc/tT6b3KGN/xray-sample.png)


## Summary

Hospital patients can have catheters and tubes inserted during their admission. If tubes are placed incorrectly, serious health complications can occur later. Using deep learning helps to automate detection of malpositioned tubes, which allows to reduce the workload of clinicians and prevent treatment delays.

This project works with a dataset of 30,083 high-resolution chest X-ray images. The images have 11 binary labels indicating normal, borderline or abnormal placement of endotracheal tubes, nasogastric tubes, central venous catheters and Swan-Ganz catheters.

I develop an ensemble of 5+2 CNN models implemented in `PyTorch`. The solution reaches the test AUC of 0.971 and places 71st out of 1,549 competing teams. The diagram below overviews the ensemble. The detailed summary is provided in [this writeup](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226664).

![ensemble](https://i.postimg.cc/c4cPcXng/ranzcr.png)


## Project structure

The project has the following structure:
- `codes/`: `.py` scripts with training, inference and image processing functions
- `notebooks/`: `.ipynb` Colab-friendly notebooks for training CNNs and ensembling
- `input/`: input data (not included due to size constraints, can be downloaded [here](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification))
- `output/`: model configurations, weights and figures exported from the notebooks


## Reproducing solution

The solution can be reproduced in the following steps:
1. Downloading competition data and placing it in the `input/` folder.
2. Running five training notebooks `model_x.ipynb` to obtain weights of base models.
3. Running the ensembling notebook `ensembling.ipynb` to obtain the final prediction.

All `model_x.ipynb` notebooks have the same structure and differ in model/data parameters. Different versions are included to ensure reproducibility. To understand the training process, it is sufficient to go through the `codes/` folder and inspect one of the modeling notebooks. The ensembling code is also provided in this [Kaggle notebook](https://www.kaggle.com/kozodoi/71st-place-ensembling-pipeline/output).

More details are provided in the documentation within the scripts & notebooks.
