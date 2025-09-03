import streamlit as st
import pandas as pd
from model_utils import load_models, predict_clusters
from graph_utils import cluster_graphs, plot_cluster_frequencies, column_comparison_bar
import plotly.express as px

st.set_page_config(page_title="Cluster Explorer", layout="wide")

scaler, pca, kmeans, cluster_means, scaler_columns = load_models()

spending_cols = ['Entertainment', 'Food Dining', 'Gas Transport', 'Health Fitness',
                 'Home', 'Kids Pets', 'Personal Care', 'Travel', 'Grocery', 'Misc', 'Shopping']

job_cols = ['JobCat_Arts, Entertainment, and Media', 'JobCat_Education and Research',
            'JobCat_Finance, Insurance, and Business Services',
            'JobCat_Healthcare and Social Assistance',
            'JobCat_Manufacturing, Construction, and Logistics',
            'JobCat_Retail, Hospitality, and Customer Service',
            'JobCat_Technology and Engineering']

key_numeric_cols = cluster_means.columns.drop(spending_cols + job_cols, errors='ignore')

tabs = st.tabs(
    ["Uploaded Data"] + [f"Cluster {i}" for i in cluster_means.index] + ["Comparisons"]
)

with tabs[0]:
    st.header("Uploaded Data Cluster Predictions")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data_with_clusters = predict_clusters(data, scaler, pca, kmeans, scaler_columns)
        
        st.subheader("Predicted Clusters")
        st.dataframe(data_with_clusters[['Cluster']])
        
        st.subheader("Cluster Summaries (All Clusters)")
        st.dataframe(cluster_means)
        
        st.subheader("Cluster Frequencies in Uploaded Data")
        plot_cluster_frequencies(data_with_clusters)

    else:
        st.write("Upload a CSV file to see cluster predictions and summaries.")
        
for i, tab in enumerate(tabs[1:-1]):
    with tab:
        st.header(f"Graphs for Cluster {i}")
        cluster_graphs(cluster_means, i, spending_cols, job_cols)


with tabs[-1]:
    st.header("Comparisons Between Clusters")
    column_comparison_bar(cluster_means)