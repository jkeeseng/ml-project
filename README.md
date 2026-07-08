# End-to-End ML Deployment Project

This repository contains an end-to-end machine learning project for student performance prediction. It includes data exploration, model training, serialized model artifacts, a Flask prediction app, and deployment configuration for AWS Elastic Beanstalk.

## Project Overview

The application predicts a student's math score from demographic and academic inputs such as gender, ethnicity, parental education level, lunch type, test preparation status, reading score, and writing score.

## Repository Structure

```text
.
├── .ebextensions/          # Elastic Beanstalk configuration
├── artifacts/              # Trained model, preprocessor, and processed data artifacts
├── endtoendmlproject/      # Earlier project package/files kept with the deployment work
├── notebook/               # EDA and model training notebooks
├── src/                    # Training, transformation, prediction, logging, and utility code
├── templates/              # Flask HTML templates
├── app.py                  # Flask application entry point
├── application.py          # Elastic Beanstalk-compatible application entry point
├── requirements.txt        # Runtime dependencies
├── requirements-train.txt  # Training dependencies
└── setup.py                # Package configuration
```

## Main Features

- Exploratory data analysis and model training notebooks.
- Modular training pipeline for ingestion, transformation, and model training.
- Serialized preprocessing and model artifacts for repeatable inference.
- Flask web interface for interactive prediction.
- AWS Elastic Beanstalk deployment configuration.

## Getting Started

Create a virtual environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the Flask app locally:

```bash
python app.py
```

Open the prediction page at:

```text
http://127.0.0.1:5002/predictdata
```

## Training

Training code and notebooks are available in `src/` and `notebook/`. To install the fuller training dependency set, use:

```bash
pip install -r requirements-train.txt
```

## Deployment

The repository includes Elastic Beanstalk configuration in `.ebextensions/` and an `application.py` entry point for deployment environments that expect a WSGI application object.

## Note

The unrelated research projects have been separated into their own project folders:

- `wav2vec-pronunciation-assessment`
- `koala-audio-classification`
