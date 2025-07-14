# correlation_app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Correlation Analysis", page_icon="📊", layout="wide")
st.title("TATA Commercial CSI - Correlation Analysis")

# Load dataset
csv_path = "TATA COM.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df_cleaned = df.dropna()

# Step 1: Select Branch
branches = df_cleaned["PLANT"].unique()
selected_branch = st.selectbox("Select Branch", sorted(branches))

# Filter data for selected branch
branch_df = df_cleaned[df_cleaned["PLANT"] == selected_branch]

# Step 2: Select numeric columns
numeric_columns = branch_df.select_dtypes(include='number').columns.tolist()
selected_columns = st.multiselect(
    "Select numeric columns for correlation",
    numeric_columns,
    default=numeric_columns
)

# Threshold for minimum records per advisor
min_required = 5

if selected_columns and not branch_df.empty:

    # ✅ Overall Branch Correlation
    st.markdown(f"## Overall Correlation - `{selected_branch}` Branch")
    overall_corr = branch_df[selected_columns].corr(method='spearman')
    overall_fig = px.imshow(
        overall_corr,
        text_auto=True,
        color_continuous_scale='RdBu',
        title=f"Spearman Correlation Matrix - {selected_branch} (Overall)",
        aspect='auto'
    )
    st.plotly_chart(overall_fig, use_container_width=True)

    # ✅ Advisor-wise Correlation Heatmaps
    st.markdown("## Service Advisor-wise Correlation Heatmaps")

    advisors = branch_df["Service Advisor"].unique()

    for advisor in sorted(advisors):
        advisor_df = branch_df[branch_df["Service Advisor"] == advisor]
        record_count = len(advisor_df)

        st.markdown(f"---\n### Advisor: `{advisor}` (Records: {record_count})")

        if record_count >= min_required:
            corr_matrix = advisor_df[selected_columns].corr(method='spearman')
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu',
                title=f"Spearman Correlation - {advisor}",
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Not enough data for Advisor: `{advisor}` (only {record_count} records)")

elif not selected_columns:
    st.warning("Please select at least one numeric column.")

elif branch_df.empty:
    st.warning("No data available for the selected branch.")

# Add copyright at the bottom
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #071429;
        color: #b4b8bf;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        © 2025 DIMO Customer Experience. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)