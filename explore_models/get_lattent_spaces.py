import polars as pl 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np




DATA = "/store/EQUIPES/BIM/MEMBERS/simon.herman/MicroPred/data/training_dataset/train.parquet"


def get_training_lattent_spaces(path = DATA):

    df = (
        pl.scan_parquet(path)
        .select(["protein_emb", "descriptors", "category"])
    )

    X = df.select("protein_emb").collect().to_numpy()
    y = df.select("category").collect().to_series().to_numpy()

    embeddings_scaler = StandardScaler()
    X = embeddings_scaler.fit_transform(X)

    num_components = 10
    embeddings_pca = PCA(n_components=num_components)
    X = embeddings_pca.fit_transform(X)


    PCA_embeddings_df = pl.DataFrame({ f"PCA_{i}": X[:, i] for i in range(num_components) })
    PCA_embeddings_df["category"] = y

    X = df.select("descriptors").collect().to_numpy()

    descriptors_scaler = StandardScaler()   
    X = descriptors_scaler.fit_transform(X)

    num_components = 10
    descriptors_pca = PCA(n_components=num_components)
    X = descriptors_pca.fit_transform(X)

    PCA_descriptors_df = pl.DataFrame({ f"PCA_{i}": X[:, i] for i in range(num_components) })
    PCA_descriptors_df["category"] = y
    

    return {

        "PCA_embeddings": PCA_embeddings_df,
        "embeddings_scaler": embeddings_scaler,
        "embeddings_pca": embeddings_pca,

        "PCA_descriptors": PCA_descriptors_df,
        "descriptors_scaler": descriptors_scaler,
        "descriptors_pca": descriptors_pca,

    }






