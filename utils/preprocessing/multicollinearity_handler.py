# STANDARD MODULES
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

# USER MODULES
from utils.preprocessing.data_preprocessor_abc import DataPreprocessorHandler


class MultiCollinearityHandler(DataPreprocessorHandler):
    def __init__(self):
        self.multicollinearity_detection_options = None
        self.multicollinearity_handler_options = None
        st.session_state.setdefault("existing_method_multicollinearity_handler", {})

    def display_info(self, df):
        st.markdown("### 🧠 Handling Multi-Collinearity")
        st.markdown(
            """
        **Why care about multicollinearity?**  
        Multicollinearity occurs when two or more independent variables in the dataset are highly correlated.  
        This can:
        - Make model coefficients unstable
        - Reduce model interpretability
        - Affect model performance for linear models (like Linear/Logistic Regression)
        """
        )

        st.markdown("#### 🔍 Correlation Matrix (Pearson)")
        corr = df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        threshold = 0.85
        upper_triangle = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
        high_corr_pairs = [
            (col1, col2, round(upper_triangle.loc[col1, col2], 2))
            for col1 in upper_triangle.columns
            for col2 in upper_triangle.columns
            if not pd.isna(upper_triangle.loc[col1, col2])
            and abs(upper_triangle.loc[col1, col2]) > threshold
        ]

        if high_corr_pairs:
            st.markdown("#### ⚠️ Highly Correlated Pairs (>|0.85|):")
            for col1, col2, val in high_corr_pairs:
                st.write(f"• `{col1}` and `{col2}` → Correlation: {val}")
        else:
            st.success("No feature pairs exceed the correlation threshold.")

    def run(self, df, settings, saved_configuration_file):
        self.display_info(df=df)
        return df, settings

    def init_parameters_for_col(
        self, df, col="", categorical_cols=..., missing_values=...
    ):
        pass

    def apply_method(self, df, col, method_name, outlier_indices=...):
        pass
