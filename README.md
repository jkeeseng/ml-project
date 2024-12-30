# Koala Audio Classification

This project aims to classify audio samples to determine whether they are related to koalas or not. It utilizes machine learning techniques, data augmentation, and audio feature extraction to build an effective classification model.

## Features

- **Audio Data Preprocessing**: Handles raw audio data, including cleaning, filtering, and augmentation.
- **Machine Learning Model**: Trains a classification model to distinguish between koala-related and non-koala-related audio samples.
- **Evaluation Metrics**: Provides detailed metrics for evaluating model performance, such as accuracy, precision, and recall.

## Dataset

- **Koala Samples**: 540 audio samples (including augmented data).
- **Non-Koala Samples**: 560 audio samples.
- **Data Format**: Audio recordings.

## Requirements

The project uses the following dependencies:

- **Python** 3.6 or higher
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `librosa`
  - `matplotlib`
  - `seaborn`
  - `jupyter`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jkeeseng/ml-project.git
   cd ml-project
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

## Usage

1. Open the notebook file:

   ```
   koala classification project-1.ipynb
   ```

2. Follow the step-by-step instructions to preprocess the data, train the model, and evaluate its performance.

## Results

- The classification model achieved high accuracy in identifying koala-related audio samples.
- Detailed evaluation metrics and visualizations are available in the notebook.

## Future Work

- Explore advanced machine learning techniques, such as deep learning.
- Extend the dataset to include more diverse audio samples.
- Optimize the model for real-time audio classification.

---

# Wav2Vec2 Speech Classification

This project implements speech classification using a fine-tuned Wav2Vec2 model. It leverages the `transformers` library for model handling and PyTorch for deep learning operations.

## Features

- **Audio Preprocessing**: Handles raw audio data, including resampling and feature extraction.
- **Speech Classification Model**: Utilizes a fine-tuned Wav2Vec2 model to classify speech data.
- **Inference Pipeline**: Provides an easy-to-use pipeline for speech classification.

## Dataset

- **Audio Samples**: Includes various audio samples processed for classification.
- **Format**: WAV files compatible with the Wav2Vec2 model.

## Requirements

To run this project, install the required Python libraries:

```bash
pip install torch torchaudio transformers
```

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. Run the inference script:

   ```bash
   python wav2vec_inference.py
   ```

3. Modify the `inference_w2v_model` function as needed to use your custom model and dataset.

## Results

- The classification model outputs predictions with labels and confidence scores.

## Future Work

- Extend functionality for multi-language support.
- Optimize the model for real-time applications.

## Contributing

Contributions are welcome! If you’d like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add feature-name"
   ```

4. Push to the branch:

   ```bash
   git push origin feature-name
   ```

5. Open a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please reach out to jkeeseng on GitHub. There will be other projects updated on my repository. Please stay tuned!
