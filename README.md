# Adaptation of data domains in federated learning in the problem of breast mammography image classification

Machine learning models, especially deep learning models, are trained on a specific portion of data. However, data generation is an ongoing problem due to new individuals undergoing examinations (individual variability), new calibration settings for medical imaging devices, new types of diagnostic devices, etc. This raises the problem of how to effectively train a model to have generalization abilities while minimizing the impact of the type of equipment used, etc.

The aim of the project is a comparative analysis of the influence of different methods of domain adaptation (influence of equipment diversity, image formats, etc.) on the classification results of mammographic images using federated learning.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Team Members](#team-members)

## Installation

```bash
$ git clone https://github.com/Xentomm/Base-Models-and-Domain-Adaptation-for-Federated-Learning.git
$ cd project-name
$ python -m venv .env (or conda)
$ source .env/Scripts/actiavate (Windows) | source .env/bin/activate (Linux)
$ pip install -r requirements.txt
```

## Usage

Train the base models resnet or effnet using one of this commands.

```bash
$ python src/resnet rsna path/to/model --epochs=100
$ python src/effnet rsna path/to/model --epochs=100  
```

Train base model using one of the three domain adaptation techniques(Adda/Revgrad/Wdgrl).

```bash
$ python src/adda.py path/to/base_model vindir rsna --batch-size=64 --iterations=100 --epochs=20
$ python src/revgrad.py path/to/base_model vindir rsna --batch-size=64 --iterations=100 --epochs=20
$ python src/wdgrl.py path/to/base_model vindir rsna --batch-size=64 --iterations=100 --epochs=20
```

Use basic prediction if needed.

```bash
$ python src/predict.py path/to/trained_model
```

To further use this models in federated learning environment use this [link](https://github.com/Ola2808-Boro/Federated-Learning-Project) to access repository with federated learning web application.

## Features

- src/dataloader.py : data transformation to model input
- src/(model_name).py : training, validation and testing of choosen model
- src/predict.py : testing model

## Team members

- [Łukasz Erimus](https://github.com/Xentomm)
- [Aleksandra Borowska](https://github.com/Ola2808-Boro)
- [Adrian Jaromin](https://github.com/IcyArcticc)
- [Agnieszka Lewko](https://github.com/Acquilli)

Supervisor: Ph.D. Jacek Rumiński
