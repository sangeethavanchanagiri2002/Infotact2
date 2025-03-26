import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os

# Title of the app
st.title("Clustering with KMeans")

# File upload section
st.header("Upload a CSV File")
uploaded_file = st.file_uploader("cinema_merged_data_modifiedcolumns.csv", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Check if the dataset is empty
        if data.empty:
            st.error("The uploaded file is empty!")
        else:
            st.success("File uploaded successfully!")
            st.write("### Dataset Preview")
            st.dataframe(data.head())

            # Feature selection
            st.write("### Select Features for Clustering")
            selected_features = st.multiselect(
                "Choose columns to include:",
                options=data.columns,
                default=data.columns.tolist(),
            )

            if selected_features:
                df_selected = data[selected_features]

                # Preprocessing: Scaling data
                scaler = StandardScaler()
                df_scaled = scaler.fit_transform(df_selected)

                # Silhouette scores for k values from 2 to 10
                st.write("### Silhouette Score Calculation")
                silhouette_coefficients = []
                for k in range(2, 11):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(df_scaled)
                    score = silhouette_score(df_scaled, kmeans.labels_)
                    silhouette_coefficients.append([k, score])

                # Plot silhouette scores
                silhouette_df = pd.DataFrame(silhouette_coefficients, columns=["K", "Silhouette Score"])
                st.line_chart(silhouette_df.set_index("K"), use_container_width=True)

                # Determine the best k
                best_k = max(silhouette_coefficients, key=lambda x: x[1])[0]

                # Train final KMeans model
                st.write("### Training Final KMeans Model")
                best_model = KMeans(n_clusters=best_k, random_state=42)
                data["Cluster"] = best_model.fit_predict(df_scaled)

                st.write(f"Optimal Number of Clusters (K): {best_k}")

                # Visualize clusters using the first two features
                if len(selected_features) >= 2:
                    st.write("### Cluster Visualization")
                    fig = px.scatter(
                        data,
                        x=selected_features[0],
                        y=selected_features[1],
                        color=data["Cluster"].astype(str),
                        title="Cluster Visualization",
                        labels={"color": "Cluster"},
                    )
                    st.plotly_chart(fig)

                # Show clustered data
                st.write("### Clustered Data")
                st.dataframe(data)

                # Option to download clustered data
                st.download_button(
                    label="Download Clustered Data",
                    data=data.to_csv(index=False).encode("utf-8"),
                    file_name="clustered_data.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Please select at least one feature for clustering.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Upload a file to get started.")
