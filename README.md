# Brain-Tumor-Detection-and-Data-Augmentation-with-Transfer-Learning

## Brain Tumor Detection using Deep Learning

This repository contains a project for detecting brain tumors using deep learning techniques. The primary objective of this project is to develop a convolutional neural network (CNN) model that can accurately classify brain tumor images into four categories: glioma, meningioma, notumor, and pituitary. The project employs transfer learning and data augmentation to enhance model performance and generalization.

### Project Structure

- **data/**: Contains the training and testing datasets. Each dataset is organized into subfolders representing the four categories of brain tumors.
- **notebooks/**: Jupyter notebooks for data preprocessing, model training, evaluation, and visualization.
- **models/**: Saved models and weights.
- **scripts/**: Python scripts for data loading, augmentation, and model definition.

### Key Features

- **Data Preprocessing**: Loading and preprocessing of brain tumor images, including resizing and normalization.
- **Data Augmentation**: Synthetic data generation using various augmentation techniques to improve model robustness.
- **Transfer Learning**: Utilization of the pre-trained VGG16 model for feature extraction and subsequent fine-tuning.
- **Model Training**: Training the CNN model with augmented data and monitoring performance on a validation set.
- **Model Evaluation**: Assessment of model accuracy, precision, recall, and F1-score on the testing dataset.
- **Visualization**: Displaying sample images with true and predicted labels to visualize model performance.

### How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/brain-tumor-detection.git
    cd brain-tumor-detection
    ```

2. **Set Up the Environment**:
    Create a virtual environment and install the required dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Prepare the Data**:
    Ensure the dataset is structured correctly under the `data/` directory. The data should be divided into `training` and `testing` folders, each containing subfolders for each category.

4. **Run the Notebooks**:
    Open and run the Jupyter notebooks in the `notebooks/` directory to preprocess data, train the model, and evaluate its performance.

5. **Model Inference**:
    Use the trained model to make predictions on new brain tumor images. Check the `scripts/` directory for example scripts on how to load the model and perform inference.

### Results

- The model achieves an accuracy of [insert accuracy]% on the testing dataset.
- Precision, recall, and F1-score for each class:
  - **Glioma**: Precision: [insert precision], Recall: [insert recall], F1-score: [insert F1-score]
  - **Meningioma**: Precision: [insert precision], Recall: [insert recall], F1-score: [insert F1-score]
  - **Notumor**: Precision: [insert precision], Recall: [insert recall], F1-score: [insert F1-score]
  - **Pituitary**: Precision: [insert precision], Recall: [insert recall], F1-score: [insert F1-score]

- Overall performance metrics:
  - Macro Precision: [insert macro precision]
  - Macro Recall: [insert macro recall]
  - Macro F1-score: [insert macro F1-score]

### Future Work

- Explore advanced deep learning architectures to improve accuracy and robustness.
- Expand the dataset with more diverse images to enhance model generalization.
- Implement real-time inference capabilities for deployment in clinical settings.

### Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

