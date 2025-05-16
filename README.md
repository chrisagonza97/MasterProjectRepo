# Urothelial Cytology Segmentation and Diagnosis Pipeline

This repository contains the codebase for my **Master's project**, focused on improving automated segmentation and diagnosis in urothelial carcinoma cytology slides. It builds on [Cedars AI Campus - Project 3](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/tree/main/Project3) by integrating stronger models and diagnostic analysis pipelines.

All code is written in **Python 3.12.2**.

---

## Project Structure and Additions

### Core Notebooks and Scripts

- **`1_load_data.ipynb`**  
  Trains and evaluates both TransUNet and U-Net. Includes experiments with different augmentation combinations and performs the first hyperparameter sweep.

- **`2_Intensity_KMeans.ipynb`**  
  Implements traditional segmentation methods using intensity-based thresholding and K-Means clustering.

- **`augment_train.py`**  
  Applies traditional data augmentations to the training set.

- **`cross_validate.ipynb`**  
  Performs cross-validation for four segmentation approaches. Also evaluates how segmentation quality affects downstream diagnosis via NC ratio and logistic regression.

- **`h_sweep_2.ipynb`**  
  Contains code for the second hyperparameter sweep for deep learning models.

- **`semseg_functions.py`**  
  Contains reusable code for training and evaluating U-Net and TransUNet, including custom training loops and hyperparameter tuning.

- **`transunet.py`**  
  Full implementation of TransUNet model, adapted for 2D cytology image segmentation.

---

### Diagnosis & Analysis

- **`1_specimen_celldata.ipynb`**  
  Analyzes cell-level specimen data and implements logistic regression and random forest classifiers for binary and multi-class diagnostic prediction tasks.

---

### Synthetic Data

- **`synthesize_data.ipynb`**  
  Preprocesses cell images for training a pix2pix GAN to generate synthetic training images.

- **`pix2pix.ipynb`**  
  Notebook for experimenting with GAN-based data synthesis.

---

### Data Folders

- `DL_result_imgs`, `cell_images`, `fake_cells_train`, `imagedata`, `img_label_jpgs`, `threshold_models`, etc.  
  Contain input images, processed datasets, and model artifacts.

---

## Reference

This repository builds off the original Cedars-Sinai project:  
**[Cedars AI Campus - Project 3](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/tree/main/Project3)**

---

## License

This project is for academic and research use only. Please cite the original Cedars AI Campus repo if used.

