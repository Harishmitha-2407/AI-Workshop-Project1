 <div align="center">

# 🌙 Artificial Neural Network for Binary Classification
### Learning Non-Linear Patterns with the `make_moons` Dataset

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Notebook%20Project-2EA44F?style=for-the-badge)

**A complete deep-learning project demonstrating how an Artificial Neural Network learns complex, non-linear patterns to classify two moon-shaped classes.**

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Objectives](#-objectives)
3. [How It Works](#-how-it-works)
4. [Dataset Description](#-dataset-description)
5. [Model Architecture](#-model-architecture)
6. [Technology Stack](#️-technology-stack)
7. [Installation and Setup](#-installation-and-setup)
8. [Project Workflow](#-project-workflow)
9. [Model Evaluation](#-model-evaluation)
10. [Visualizations](#-visualizations)
11. [Making Predictions](#-making-predictions)
12. [Saving the Model](#-saving-the-model)
13. [Project Structure](#-project-structure)
14. [Future Improvements](#-future-improvements)
15. [Limitations](#️-limitations)
16. [Contributing](#-contributing)
17. [License](#-license)

---

## ✨ Project Overview

This project demonstrates the implementation of an **Artificial Neural Network (ANN)** for binary classification using the `make_moons` dataset provided by Scikit-learn.

The dataset contains two interleaving crescent-shaped classes. Since these classes cannot be effectively separated by a single straight line, the model must learn a non-linear decision boundary.

A feed-forward neural network is developed using TensorFlow and Keras to learn these patterns and classify the input samples into two classes.

The project covers the complete machine-learning workflow, including data generation, visualization, preprocessing, model development, training, evaluation, prediction, and model saving.

### 🎯 Project Goal

To understand how an ANN learns non-linear relationships from two-dimensional data and how its classification performance can be evaluated using standard machine-learning metrics.

## 🎯 Objectives

- Generate a synthetic dataset for binary classification.
- Understand the structure of the `make_moons` dataset.
- Visualize the distribution of the two classes.
- Split the dataset into training and testing sets.
- Standardize input features using `StandardScaler`.
- Design a multi-layer Artificial Neural Network.
- Train the model using TensorFlow/Keras.
- Apply early stopping to help reduce unnecessary training.
- Evaluate model performance using accuracy and loss.
- Generate a classification report and confusion matrix.
- Visualize training and validation performance.
- Plot the model's learned decision boundary.
- Predict the class of new input samples.
- Save the trained neural network for future use.

## 🧠 How It Works

The project follows these major steps:

1. **Data Generation:** Generate two moon-shaped classes using Scikit-learn.
2. **Data Visualization:** Display the samples to understand their distribution.
3. **Data Splitting:** Divide the dataset into training and testing sets.
4. **Feature Scaling:** Standardize the input features.
5. **Model Construction:** Create a neural network with multiple dense layers.
6. **Model Training:** Train the network using the Adam optimizer.
7. **Validation:** Monitor validation loss during training.
8. **Evaluation:** Calculate test accuracy and loss and inspect classification metrics.
9. **Visualization:** Display learning curves, the confusion matrix, and the decision boundary.
10. **Prediction:** Use the trained model to classify new samples.
11. **Model Saving:** Save the trained neural network for later use.

## 📊 Dataset Description

The project uses Scikit-learn's `make_moons` function to generate a synthetic dataset.

```python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=1000,
    noise=0.20,
    random_state=42
)
```

### Dataset Details

| Property | Description |
|---|---|
| Dataset | Make Moons |
| Source | Scikit-learn |
| Number of samples | 1,000 |
| Input features | 2 |
| Number of classes | 2 |
| Class labels | 0 and 1 |
| Noise level | 0.20 |
| Random state | 42 |
| Test set size | 20% |

### Why Use the Make Moons Dataset?

The `make_moons` dataset is useful for understanding non-linear classification because the two classes form interleaving crescent shapes.

A simple linear classifier may struggle to separate the classes accurately. A neural network with non-linear activation functions can learn more flexible decision boundaries.

The dataset is synthetic, so an external dataset download is not required.

## 🏗️ Model Architecture

The project uses a feed-forward Artificial Neural Network built with TensorFlow and Keras.

### Network Structure

```python
model = Sequential([
    Input(shape=(2,)),
    Dense(16, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

### Layer Explanation

| Layer | Configuration | Purpose |
|---|---|---|
| Input Layer | 2 features | Accepts the two input coordinates |
| Hidden Layer 1 | 16 neurons, ReLU | Learns initial feature patterns |
| Hidden Layer 2 | 16 neurons, ReLU | Learns more complex relationships |
| Hidden Layer 3 | 8 neurons, ReLU | Refines learned representations |
| Output Layer | 1 neuron, Sigmoid | Produces a probability for Class 1 |

### Activation Functions

**ReLU (Rectified Linear Unit)**

ReLU introduces non-linearity into the hidden layers, enabling the network to learn complex patterns.

**Sigmoid**

The sigmoid activation function produces an output between 0 and 1. This value is interpreted as the estimated probability of belonging to Class 1.

A threshold of 0.5 is used to convert the output probability into a predicted class.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Evaluation Metric | Accuracy |
| Maximum Epochs | 150 |
| Batch Size | 32 |
| Validation Split | 20% of training data |
| Early Stopping Monitor | `val_loss` |
| Patience | 15 epochs |
| Restore Best Weights | Enabled |

### Why Use Adam?

Adam is an optimization algorithm that updates the network's weights during training. It adapts learning rates for individual parameters and is commonly used for neural-network training.

### Why Use Binary Cross-Entropy?

Binary cross-entropy measures the difference between the predicted probabilities and the actual binary labels. It is suitable for binary classification models that use a sigmoid output.

### Why Use Early Stopping?

Early stopping monitors validation loss and stops training when improvement does not occur for the specified number of epochs. Restoring the best weights helps retain the model state with the lowest monitored validation loss.

## ⚙️ Technology Stack

The following technologies are used in this project:

- **Python:** Main programming language.
- **NumPy:** Numerical operations and array manipulation.
- **Matplotlib:** Data visualization and plotting.
- **Scikit-learn:** Dataset generation, feature scaling, data splitting, and evaluation metrics.
- **TensorFlow:** Deep-learning framework.
- **Keras:** Neural-network construction, training, evaluation, and saving.
- **Google Colab / Jupyter Notebook:** Interactive development environments.

## 🚀 Installation and Setup

### Option 1: Google Colab

1. Open Google Colab.
2. Upload `ANN_using_make_moons_dataset(1).ipynb`.
3. Run the notebook cells sequentially or select **Runtime → Run all**.
4. Review the output metrics and visualizations.

Google Colab provides a convenient environment for running the notebook without setting up Python locally.

### Option 2: Run Locally

Install Python and Jupyter Notebook, then install the required libraries.

```bash
python -m pip install numpy matplotlib scikit-learn tensorflow jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and execute the cells from beginning to end.

**Note:** If the notebook already contains a package-installation cell and the required libraries are installed, you can skip the installation step.

## 🔬 Project Workflow

### Step 1: Generate the Dataset

Generate 1,000 samples with two features and two classes using `make_moons`.

### Step 2: Visualize the Data

Plot the samples using Matplotlib. Different colors represent the two classes and help illustrate their non-linear structure.

### Step 3: Split the Dataset

Separate the dataset into training and testing portions.

- Training portion: 80%
- Testing portion: 20%

The split uses a fixed random state and stratification to help preserve class proportions.

### Step 4: Standardize Features

Use `StandardScaler` to standardize the input features.

The scaler is fitted on the training data and then applied to the test data. This prevents the test set from influencing the fitted scaling parameters.

### Step 5: Build the Neural Network

Construct the ANN using Keras's Sequential API and dense layers with ReLU and sigmoid activation functions.

### Step 6: Compile the Model

Configure the model with:

- Adam optimizer
- Binary cross-entropy loss
- Accuracy metric

### Step 7: Train the Model

Train the neural network using mini-batches and monitor validation loss. Early stopping helps avoid continuing training when validation performance stops improving.

### Step 8: Evaluate the Model

Evaluate the trained model on the held-out test set and inspect its classification performance.

### Step 9: Visualize the Results

Generate plots to understand model learning, prediction errors, and the learned decision boundary.

### Step 10: Save the Model

Save the trained neural network in Keras format for future inference.

## 📈 Model Evaluation

Model evaluation is important for understanding how well the neural network classifies unseen data.

### Metrics Used

**1. Accuracy**

Accuracy measures the proportion of correctly classified samples.

Accuracy = Correct Predictions / Total Predictions

**2. Loss**

Binary cross-entropy loss measures the error between predicted probabilities and actual labels. Lower loss generally indicates better probabilistic predictions on the evaluated dataset.

**3. Precision**

Precision measures the proportion of positive predictions that are correct.

**4. Recall**

Recall measures the proportion of actual positive samples that the model correctly identifies.

**5. F1-Score**

The F1-score is the harmonic mean of precision and recall. It provides a combined measure of these two metrics.

### Confusion Matrix

The confusion matrix compares the actual labels with the predicted labels.

- **True Negative (TN):** Class 0 correctly predicted as Class 0.
- **False Positive (FP):** Class 0 incorrectly predicted as Class 1.
- **False Negative (FN):** Class 1 incorrectly predicted as Class 0.
- **True Positive (TP):** Class 1 correctly predicted as Class 1.

The confusion matrix helps identify which class is being misclassified.

### Learning Curves

The notebook plots training and validation accuracy and loss.

These plots help to:

- Monitor model learning over epochs.
- Compare training performance with validation performance.
- Identify possible overfitting.
- Understand whether additional training may be useful.

### Decision Boundary

The decision-boundary plot shows how the trained neural network separates the two classes across the two-dimensional feature space.

It provides a visual explanation of how the network learns a non-linear classification boundary.

**Performance note:** Actual test accuracy, loss, and classification metrics should be reported from the results produced when the notebook is executed. No specific score is claimed here because a verified execution result is not included in the notebook source.

## 🧪 Making Predictions

After training, the model can classify new samples.

Example:

```python
import numpy as np

new_sample = np.array([[0.5, 0.2]])

new_sample_scaled = scaler.transform(new_sample)

prediction = model.predict(
    new_sample_scaled,
    verbose=0
)[0][0]

predicted_class = int(prediction >= 0.5)

print("Probability of Class 1:", float(prediction))
print("Predicted Class:", predicted_class)
```

The output probability indicates the model's estimated probability for Class 1.

The predicted class is:

- `1` when the probability is greater than or equal to 0.5.
- `0` when the probability is below 0.5.

**Important:** Every new sample must be transformed using the same fitted scaler used during training.

## 💾 Saving the Model

Save the trained model using:

```python
model.save("ann_make_moons.keras")
```

The model is saved in the current working directory as `ann_make_moons.keras`.

### Saving the Scaler

The fitted `StandardScaler` is a separate object and is not automatically included in the saved Keras model. Save it separately if you need consistent preprocessing for future predictions.

```python
import joblib

joblib.dump(
    scaler,
    "make_moons_scaler.joblib"
)
```

Install Joblib if necessary:

```bash
python -m pip install joblib
```

### Loading the Saved Model

```python
import joblib
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "ann_make_moons.keras"
)

scaler = joblib.load(
    "make_moons_scaler.joblib"
)

sample = np.array([[0.5, 0.2]])
sample_scaled = scaler.transform(sample)

probability = float(
    model.predict(sample_scaled, verbose=0)[0][0]
)

predicted_class = int(probability >= 0.5)

print("Probability:", probability)
print("Predicted Class:", predicted_class)
```

This allows the trained model to be reused without retraining it, provided that both the model and scaler files are available.

## 📁 Project Structure

```text
ann-make-moons-classification/
│
├── ANN_using_make_moons_dataset(1).ipynb
│
├── README.md
│
├── ann_make_moons.keras
│
└── make_moons_scaler.joblib
```

### File Description

- `ANN_using_make_moons_dataset(1).ipynb`: Contains the data-generation, preprocessing, model-training, evaluation, and visualization workflow.
- `README.md`: Provides project documentation and setup instructions.
- `ann_make_moons.keras`: Saved neural-network model, generated after running the relevant saving cell.
- `make_moons_scaler.joblib`: Optional saved feature scaler for future predictions.

The model and scaler files will not exist until they have been saved.

## 🚀 Future Improvements

The project can be extended in several ways:

- Experiment with different numbers of hidden layers and neurons.
- Compare ReLU with other activation functions.
- Test different batch sizes and learning rates.
- Investigate the effect of different noise levels.
- Increase the dataset size and compare performance.
- Compare ANN performance with logistic regression and support vector machines.
- Evaluate multiple random seeds for more reliable comparisons.
- Experiment with different classification thresholds.
- Add model-performance comparison charts.
- Create a simple web application for interactive predictions.

For fair evaluation, keep the test set separate from model tuning and compare experiments using consistent data splits.

## ⚠️ Limitations

- The project uses synthetic data rather than a real-world dataset.
- Results may vary with random seeds, library versions, and training behavior.
- A good score on this dataset does not guarantee performance on other classification problems.
- The test set should not be repeatedly used to tune the model.
- The fitted scaler must be saved separately for consistent predictions.
- The notebook source does not establish a fixed benchmark accuracy.

## 🤝 Contributing

Contributions and suggestions are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test the updated notebook.
5. Update the documentation if needed.
6. Submit a pull request describing your improvements.

## 📄 License

No license is specified in the supplied notebook. If you publish this project publicly, add a license file that reflects how you want others to use, modify, and distribute your work.

---

<div align="center">

### 🌟 Learning Deep Learning, One Neuron at a Time

**Exploring Artificial Neural Networks, non-linear classification, and model evaluation with Python.**

If you find this project useful, consider giving the repository a ⭐ on GitHub.

</div>
