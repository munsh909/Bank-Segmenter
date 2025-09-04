import streamlit as st
import pandas as pd
import plotly.express as px

def plot_pie(values, labels, title):
    values = values.clip(lower=0)
    if values.sum() == 0:
        st.write(f"No positive values to display for {title}")
        return
    
    fig = px.pie(
        names=labels,
        values=values,
        title=title,
        hover_data=[labels, values]
    )
    st.plotly_chart(fig, use_container_width=True)

def cluster_graphs(cluster_means, cluster_id, spending_cols, job_cols):
    st.subheader(f"Cluster {cluster_id} graphs")

    plot_pie(cluster_means.loc[cluster_id, spending_cols], spending_cols, "Spending Distribution")

    plot_pie(cluster_means.loc[cluster_id, job_cols], job_cols, "Job Categories")

def plot_cluster_frequencies(data_with_clusters):
    freq = data_with_clusters['Cluster'].value_counts().sort_index()
    df_freq = pd.DataFrame({
        'Cluster': freq.index.astype(str),
        'Count': freq.values
    })
    fig = px.bar(df_freq, x='Cluster', y='Count', title='Cluster Frequencies', text='Count')
    st.plotly_chart(fig, use_container_width=True)
    
def column_comparison_bar(cluster_means):
    st.subheader("Compare Feature Across Clusters")
    
    numeric_cols = cluster_means.select_dtypes(include='number').columns
    col_to_plot = st.selectbox("Select a column to compare", numeric_cols)
    
    df_plot = cluster_means.reset_index().melt(id_vars='Cluster', value_vars=[col_to_plot],
                                               var_name='Feature', value_name='Value')
    
    fig = px.bar(
        df_plot,
        x='Cluster',
        y='Value',
        color='Cluster',
        title=f'Comparison of {col_to_plot} Across Clusters',
        text='Value'
    )
    st.plotly_chart(fig, use_container_width=True)
