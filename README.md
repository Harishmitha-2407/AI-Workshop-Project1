# 🧠 Deep Learning Projects: ANN & CNN


## 🚀 About This Repository

This repository contains two deep learning projects developed using Python, TensorFlow, and Keras. These projects demonstrate how neural networks can learn patterns from numerical data and images.

- **Project 1: ANN Using the make_moons Dataset** — A binary classification project that explores how an Artificial Neural Network learns nonlinear decision boundaries.
- **Project 2: CNN for Dogs vs Cats Classification** — An image classification project that uses a Convolutional Neural Network to distinguish between cats and dogs.

## 📑 Table of Contents

1. [ANN Using the make_moons Dataset](#1-ann-using-the-make_moons-dataset)
2. [CNN for Dogs vs Cats Classification](#2-cnn-for-dogs-vs-cats-classification)
3. [Installation and Setup](#3-installation-and-setup)
4. [How to Run the Projects](#4-how-to-run-the-projects)
5. [Technologies Used](#5-technologies-used)
6. [Expected Outputs](#6-expected-outputs)
7. [Repository Structure](#7-repository-structure)
8. [Learning Outcomes](#8-learning-outcomes)

---

# 1. 🔵 ANN Using the make_moons Dataset

## 📖 Overview

The Artificial Neural Network (ANN) project demonstrates binary classification using the synthetic `make_moons` dataset from Scikit-learn.

The dataset contains two interleaving crescent-shaped classes, making it useful for understanding how neural networks learn complex, nonlinear patterns.

### 🎯 Objectives

- Understand the fundamentals of artificial neural networks.
- Generate and visualize a synthetic dataset.
- Preprocess data using feature standardization.
- Train a neural network using TensorFlow/Keras.
- Evaluate classification performance using multiple metrics.
- Visualize the decision boundary learned by the model.

### 📊 Dataset Details

| Property | Value |
|---|---|
| Dataset | `make_moons` |
| Number of samples | 1,000 |
| Input features | 2 |
| Number of classes | 2 |
| Noise | 0.20 |
| Random state | 42 |
| Test split | 20% |

The dataset is generated automatically, so no external dataset download is required.

### 🏗️ ANN Architecture

| Layer | Configuration |
|---|---|
| Input layer | 2 features |
| Hidden layer 1 | 16 neurons, ReLU |
| Hidden layer 2 | 16 neurons, ReLU |
| Hidden layer 3 | 8 neurons, ReLU |
| Output layer | 1 neuron, Sigmoid |

### ⚙️ Training Configuration

- **Optimizer:** Adam
- **Loss function:** Binary Cross-Entropy
- **Evaluation metric:** Accuracy
- **Maximum epochs:** 150
- **Batch size:** 32
- **Early stopping:** Monitors validation loss with a patience of 15 epochs

### ✨ Key Features

- Synthetic dataset generation and visualization
- Training and testing data splitting
- Feature standardization using `StandardScaler`
- ANN model creation and training
- Training and validation accuracy graphs
- Training and validation loss graphs
- Classification report and confusion matrix
- Decision-boundary visualization
- Prediction for a new data sample
- Saving the trained model

### 💾 Saved Model

After successful execution, the trained model is saved as:

```text
ann_make_moons.keras
```

---

# 2. 🐱 CNN for Dogs vs Cats Classification

## 📖 Overview

The Convolutional Neural Network (CNN) project focuses on image classification by distinguishing between cats and dogs.

CNNs automatically learn visual features such as edges, textures, shapes, and patterns from images. These features help the model make predictions about the class of an input image.

### 🎯 Objectives

- Understand the fundamentals of convolutional neural networks.
- Load and preprocess images using TensorFlow/Keras.
- Apply image normalization and data augmentation.
- Train a CNN for binary image classification.
- Evaluate model performance using validation data.
- Visualize training metrics and classification results.

### 📂 Dataset Structure

The notebook expects a cats-and-dogs dataset organized into training and validation directories.

```text
cats_and_dogs_filtered/
├── train/
│   ├── cats/
│   └── dogs/
└── validation/
    ├── cats/
    └── dogs/
```

**Important:** The notebook searches for these folders under `/content` and `~/.keras/datasets`. The image dataset must be downloaded and extracted beforehand if it is not already available. Update the notebook's search paths if you store the folders elsewhere.

### 🖼️ Image Preprocessing

| Property | Configuration |
|---|---|
| Image dimensions | 128 × 128 pixels |
| Image channels | RGB |
| Batch size | 32 |
| Label mode | Binary |
| Normalization | Pixel values scaled by 1/255 |
| Data augmentation | Random horizontal flipping and rotation |
| Dataset optimization | TensorFlow `AUTOTUNE` prefetching |

### 🏗️ CNN Architecture

| Layer | Configuration |
|---|---|
| Input layer | 128 × 128 × 3 |
| Rescaling | 1/255 |
| Data augmentation | Random flip and rotation |
| Convolution block 1 | 32 filters with max pooling |
| Convolution block 2 | 64 filters with max pooling |
| Convolution block 3 | 128 filters with max pooling |
| Flatten layer | Converts feature maps into a vector |
| Dense layer | 128 neurons, ReLU |
| Dropout layer | 0.5 |
| Output layer | 1 neuron, Sigmoid |

### ⚙️ Training Configuration

- **Optimizer:** Adam
- **Loss function:** Binary Cross-Entropy
- **Evaluation metric:** Accuracy
- **Maximum epochs:** 20
- **Batch size:** 32
- **Early stopping:** Monitors validation loss with a patience of 5 epochs

### ✨ Key Features

- Image loading from directory structures
- Image resizing and normalization
- Automatic binary label generation
- CNN model development using TensorFlow/Keras
- Data augmentation to improve generalization
- Training and validation performance monitoring
- Accuracy and loss visualization
- Classification report and confusion matrix
- Saving the trained CNN model

### 💾 Saved Model

In Google Colab, the trained model is saved at:

```text
/content/dog_cat_cnn.keras
```

You can download it using the Colab file browser after the notebook finishes running successfully.

---

# 3. ⚙️ Installation and Setup

## Step 1: Clone the Repository

Open a terminal or command prompt and execute:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

## Step 2: Create a Virtual Environment

A virtual environment helps keep project dependencies separate from other Python projects.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

## Step 4: Install Required Libraries

Run the following command to install the main dependencies for both projects:

```bash
pip install numpy matplotlib scikit-learn tensorflow jupyter ipykernel
```

The CNN notebook also includes an installation cell for TensorFlow Datasets:

```bash
pip install tensorflow-datasets
```

Note that installing TensorFlow Datasets does not automatically download the image folders required by the current CNN notebook.

## Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the required `.ipynb` file from the Jupyter interface.

### Alternative: Google Colab

You can also upload the notebooks to Google Colab and execute the cells without configuring a local Python environment.

---

# 4. ▶️ How to Run the Projects

## Running the ANN Project

1. Open `ANN_using_make_moons_dataset.ipynb`.
2. Install the required dependencies.
3. Run the notebook cells sequentially.
4. The synthetic dataset will be generated automatically.
5. Observe the dataset visualization and training progress.
6. Review the test metrics, classification report, confusion matrix, and decision boundary.
7. Locate the saved model, `ann_make_moons.keras`, in the working directory.

## Running the CNN Project

1. Open `CNN_Dogs_vs_Cats.ipynb`.
2. Install the required dependencies.
3. Download and extract a compatible cats-and-dogs image dataset.
4. Organize the images into the required training and validation folders.
5. Place the folders in a location searched by the notebook, or update its search paths.
6. Run the notebook cells sequentially.
7. Review the training and validation metrics and visualizations.
8. Retrieve the saved model from `/content/dog_cat_cnn.keras` when running in Google Colab.

---

# 5. 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Dataset generation, preprocessing, and evaluation |
| TensorFlow | Deep learning framework |
| Keras | Neural network creation and training |
| Jupyter Notebook | Interactive development environment |
| Google Colab | Cloud-based notebook execution |

---

# 6. 📈 Expected Outputs

| Project | Expected outputs |
|---|---|
| ANN | Dataset plot, test metrics, classification report, confusion matrix, accuracy/loss graphs, decision boundary, sample prediction |
| CNN | Training history, validation metrics, classification report, confusion matrix, accuracy/loss graphs |

Actual model performance depends on the training process and the dataset used. Execute the notebooks to obtain the actual results.

---

# 7. 📁 Repository Structure

```text
Deep-Learning-Projects/
│
├── README.md
├── ANN_using_make_moons_dataset.ipynb
└── CNN_Dogs_vs_Cats.ipynb
```

---

# 8. 🎓 Learning Outcomes

By completing these projects, you will gain practical experience in:

- Artificial neural network fundamentals
- Nonlinear binary classification
- Data preprocessing and feature scaling
- Convolutional neural network fundamentals
- Image preprocessing and data augmentation
- Model training and validation
- Classification metrics and confusion matrices
- Training-curve and decision-boundary visualization
- Saving and reusing trained deep learning models

---

## 🌟 Conclusion

These projects demonstrate two important applications of deep learning.

The **ANN project** explores how neural networks learn nonlinear patterns in numerical data, while the **CNN project** demonstrates how convolutional networks learn visual features for image classification.

Together, they provide practical experience in data preparation, model development, training, evaluation, and visualization using modern deep learning tools.

**Built with Python, TensorFlow, and a passion for Deep Learning.**
