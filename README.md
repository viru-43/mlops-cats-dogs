# 🐶🐱 Cats vs Dogs -- End-to-End MLOps Pipeline

## 📌 Assignment 2 -- MLOps Implementation

This project implements a complete end-to-end MLOps pipeline for a Cats
vs Dogs image classification system. The objective was not only to train
a model, but to demonstrate production-level practices including
reproducibility, experiment tracking, CI/CD automation,
containerization, and monitoring.

------------------------------------------------------------------------

## 🚀 Project Overview

This system includes:

-   Data and model versioning using **DVC**
-   Reproducible ML pipeline using `dvc.yaml`
-   Experiment tracking using **MLflow**
-   REST API deployment using **FastAPI**
-   Containerization using **Docker**
-   Automated CI/CD using **GitHub Actions**
-   Post-deployment smoke testing
-   Basic monitoring with request metrics and latency logging

------------------------------------------------------------------------

## 📁 Repository Structure

    mlops-cats-dogs/
    │
    ├── data/                 # DVC tracked raw & processed datasets
    ├── models/               # DVC tracked trained model artifact
    ├── src/                  # Core ML & API implementation
    │   ├── data_preprocessing.py
    │   ├── train.py
    │   ├── model.py
    │   ├── api.py
    │
    ├── tests/                # Unit tests
    ├── .dvc/                 # DVC configuration
    ├── .github/workflows/    # CI/CD pipeline
    ├── dvc.yaml              # Pipeline definition
    ├── dvc.lock              # Pipeline lock file
    ├── Dockerfile            # Container configuration
    ├── docker-compose.yml    # Deployment config
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🔁 Reproducible ML Pipeline (DVC)

The ML pipeline is defined in `dvc.yaml` with two stages:

1.  Preprocessing
2.  Training

To visualize the pipeline:

    dvc dag

To reproduce the pipeline:

    dvc repro

This ensures deterministic and reproducible execution.

------------------------------------------------------------------------

## 📊 Experiment Tracking (MLflow)

Training runs are tracked using MLflow.

To launch MLflow UI:

    mlflow ui

Each run logs: - Training metrics - Loss values - Model artifacts

------------------------------------------------------------------------

## 🤖 Model Artifact

Due to file size constraints, the trained model artifact is hosted on
Google Drive.

📎 **Model Download Link:**\
`https://drive.google.com/drive/folders/1h2MDOKnMIhcegU8AoMzibQztZE-bu0BN?usp=drive_link`

After downloading, place `model.pt` inside:

    models/model.pt

------------------------------------------------------------------------

## 🌐 Running the API Locally

Install dependencies:

    pip install -r requirements.txt

Start the server:

    python -m uvicorn src.api:app --reload

Open in browser:

    http://localhost:8000/docs

Available endpoints:

-   `/health` -- Service validation
-   `/predict` -- Image classification
-   `/metrics` -- Request monitoring

------------------------------------------------------------------------

## 🐳 Docker Deployment

Build image:

    docker build -t cats-dogs-api .

Run container:

    docker compose up -d

------------------------------------------------------------------------

## 🔄 CI/CD Pipeline

GitHub Actions automatically runs on every push to `master`:

Pipeline steps: - Install dependencies - Run unit tests - Build Docker
image - Push image to Docker Hub - Run container - Execute smoke test
(`/health`) - Fail pipeline if validation fails

This ensures automated deployment validation.

------------------------------------------------------------------------

## 📈 Monitoring

The API includes basic monitoring features:

-   Request counter (`/metrics`)
-   Latency tracking
-   Structured logging

------------------------------------------------------------------------

## 🎥 Demo Video

📎 **Demo Video Link:**\
`<PASTE_YOUR_GOOGLE_DRIVE_VIDEO_LINK_HERE>`{=html}

------------------------------------------------------------------------

## 🏗️ Technologies Used

-   Python
-   PyTorch
-   FastAPI
-   DVC
-   MLflow
-   Docker
-   GitHub Actions

------------------------------------------------------------------------

## 📌 Summary

This project demonstrates a complete production-oriented MLOps workflow
including:

-   Artifact versioning
-   Reproducible training pipeline
-   Experiment tracking
-   Containerized inference
-   Automated CI/CD
-   Deployment validation
-   Monitoring

------------------------------------------------------------------------

**Author:** Virendra Sahakari and Group 24 MLOps assignment group
