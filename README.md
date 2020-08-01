## Recommenders (pytorch)
Include implementations on two types of recommenders:
1. Restricted Boltzmann Machine based recommender (notebook example).
2. Autoencoder based recommender.

- common: includes utility functions shared by different recommender systems, for example, data preprocessing.
    - preprocessor.py
    - util.py
- models: includes each model's Class: 
    - encoder.py
- dataset: datasets used in this project.
    - MovieLens dataset (used in this project)
    - BookCrossing dataset