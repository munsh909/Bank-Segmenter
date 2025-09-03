import pickle
import pandas as pd

def load_models():
    scaler = pickle.load(open("Model/scaler.pkl", "rb"))
    pca = pickle.load(open("Model/pca.pkl", "rb"))
    kmeans = pickle.load(open("Model/kmeans.pkl", "rb"))
    cluster_means = pd.read_csv("Model/cluster_means.csv", index_col=0)
    
    scaler_columns = list(cluster_means.columns)  
    return scaler, pca, kmeans, cluster_means, scaler_columns

def predict_clusters(df, scaler, pca, kmeans, scaler_columns):
    df = df.copy()
    
    for col in scaler_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[scaler_columns]
    
    data_scaled = scaler.transform(df)
    data_pca = pca.transform(data_scaled)
    clusters = kmeans.predict(data_pca)
    df['Cluster'] = clusters
    return df