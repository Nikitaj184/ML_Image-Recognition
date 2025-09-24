🏭 Industrial Defect Detection Using Image Recognition with Convolutional Neural Networks


This project implements an end-to-end defect detection system for
industrial casting products using deep learning (CNNs). It leverages the
Real-life Industrial Dataset of Casting Product to classify defective
and non-defective products.

Aim
The aim of this research is to develop an image recognition system using machine learning, Convolutional Neural Networks (CNNs) to automatically recognize casting products as defective or non-defective using real world industrial dataset of casting products from Kaggle. This research investigates and compares the performance of two different configurations of hyperparameters: —one based on the Adam optimizer and the other based on the SGD optimizer, in order to find the best configuration for this classification problem. Furthermore, the practical applicability and confidence of the best performing model is validated by manual testing on individual image predictions, replacing the subjective and error prone process of manual inspection.

Objectives
To obtain a real-world industrial dataset of casting product images from a public repository (Kaggle), and then preprocess it by standardizing image sizes, normalizing pixel values, and employing data augmentation techniques to enhance dataset diversity.
To implement and train two distinct Convolutional Neural Network (CNN) architectures with different hyperparameter settings such as optimizer, batch size and epochs.
To conduct a comprehensive evaluation of the trained models using a suite of classification metrics, including Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.
To manually test the best-performing model on individual defective and non-defective images to verify its effectiveness in a practical, real-world scenario.

Research Questions
How effective are Convolutional Neural Networks in detecting casting defects from real-world industrial images?
What model architecture and configuration yield the best accuracy and precision for this classification task?


------------------------------------------------------------------------

📂 Project Structure

    ├── 1)_preprocess_and_visualization_of_images.py   # Preprocessing & visualization of dataset images
    ├── 2)_implementation_of_industrial_defect_detection.py  # CNN-based defect detection pipeline
    ├── Dataset/  
    │   ├── def_front/   # Defective casting product images  
    │   └── ok_front/    # Non-defective casting product images  
    └── README.md

------------------------------------------------------------------------

📊 Dataset

-   Source: Kaggle Dataset – Real-life Industrial Casting Product
-   Categories:
    -   def_front → Defective casting images
    -   ok_front → Non-defective casting images
-   Image size: Resized to 224×224
-   Color mode: Grayscale (converted to quasi-RGB where needed)

------------------------------------------------------------------------

🛠 Workflow

1. Preprocessing & Visualization (1)_preprocess_and_visualization_of_images.py)

-   Loads defective and non-defective images.
-   Converts images to grayscale and quasi-RGB (stacked channels).
-   Visualizes random samples for dataset inspection.

2. Defect Detection Model (2)_implementation_of_industrial_defect_detection.py)

-   Preprocesses images (resize, grayscale, normalization).
-   Splits data into train (60%), validation (20%), test (20%) with
    stratification.
-   Applies data augmentation (rotation, flips, zoom, shear, shifts).
-   Implements CNN architectures with different optimizers:
    -   Adam (epochs=20, batch_size=16)
    -   SGD (epochs=50, batch_size=32)
-   Evaluates model using:
    -   Classification Report
    -   Confusion Matrix
    -   ROC Curve & AUC Score
    -   Training & Validation Accuracy/Loss curves
-   Includes manual testing for new casting images.

------------------------------------------------------------------------

📈 Results

-   CNN achieved high accuracy in distinguishing defective
    vs. non-defective castings.
-   ROC curves showed strong separability between classes.
-   Data augmentation significantly improved generalization.

------------------------------------------------------------------------

▶️ How to Run

1.  Clone this repository and download the dataset:

        git clone <your_repo_link>
        cd industrial-casting-defect-detection

2.  Place the dataset inside the Dataset/ folder:

        Dataset/
          ├── def_front/
          └── ok_front/

3.  Run preprocessing & visualization:

        python 1)_preprocess_and_visualization_of_images.py

4.  Train and evaluate the CNN model:

        python 2)_implementation_of_industrial_defect_detection.py

------------------------------------------------------------------------

📌 Requirements

-   Python 3.8+
-   OpenCV
-   NumPy
-   Matplotlib
-   Scikit-learn
-   TensorFlow / Keras
-   Seaborn

