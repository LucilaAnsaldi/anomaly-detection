# Anomaly Detection in Printing Areas

This project implements an anomaly detection system for detecting defects in printing areas of flexible packaging. It uses a combination of a YOLOv8 object detector and an autoencoder to identify and highlight defects in images.

---

## **Features**
- **Object Detection**: Uses YOLOv8 to detect and crop the printing areas of interest in images.
- **Anomaly Detection**: Employs an autoencoder to reconstruct the cropped images and calculate reconstruction errors to identify defects.
- **Result Visualization**: Highlights anomalies on the images and saves them in labeled folders for correct and incorrect images.

---

## **Installation Guide**

### **1. Clone the Repository**
First, clone the repository from GitHub to your local machine:

```bash
git clone https://github.com/your-username/anomaly-detection.git
cd anomaly-detection
```

### **2. Set Up Virtual Environment**
Install pipenv if you don't already have it:

```bash
pip install pipenv
```
Create a virtual environment and install the required dependencies:
```bash
pipenv install
```
Activate the environment:
```bash
pipenv shell
```

### **3. Prepare the Test Images**
Place the images you want to analyze inside the data/test_images folder. Ensure that the images are in .jpg, .jpeg, or .png format.


---

## **Usage Instructions**

From the Command Line: To process the test images, run the main pipeline:
```bash
python src/test_pipeline.py
```

View the Results:

- Correct images will be saved in data/test_results/correct.
- Images with anomalies will be saved in data/test_results/anomalies with highlighted areas.


## **Future Improvements**

- **Enhancing Data Quality:**:

Currently, the input data has limitations in terms of resolution and consistency, as the videos are recorded using a smartphone. Acquiring high-quality data directly from the production line would significantly improve the model's ability to detect anomalies with greater precision and reduce false positives caused by image artifacts.

- **Incorporating Additional Anomalies for Training:**:

The current autoencoder is trained exclusively on correct images. Expanding the dataset to include labeled images with various types of anomalies could improve the model's generalization and its ability to detect subtle defects in production.

- **Threshold Optimization:**:

The threshold for anomaly detection could be dynamically optimized based on statistical analysis or machine learning methods to minimize false positives and negatives.

- **Exploring Advanced Architectures:**:

Researching advanced architectures, such as Generative Adversarial Networks (GANs), might enhance the capability of the model to detect subtle and complex defects.

- **Interactive User Interface (UI):**:

An interactive interface could be developed for real-time video uploads and anomaly detection. This interface could include visualization tools to highlight detected anomalies and display the reconstruction error in a user-friendly way.


