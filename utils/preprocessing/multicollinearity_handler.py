# STANDARD MODULES
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt

# USER MODULES
from utils.settings_manager import save_configuration_if_updated
from utils.preprocessing.data_preprocessor_abc import DataPreprocessorHandler
from utils.preprocessing.data_preprocessing_helper_text import (
    DataPreprocessingOptionsHelperText,
)


class MultiCollinearityHandler(DataPreprocessorHandler):
    def __init__(self):
        self.multicollinearity_detection_options = (
            DataPreprocessingOptionsHelperText.list_multicollinearity_detection_options()
        )
        self.multicollinearity_handler_options = None
    def init_parameters_for_col(self, df, col = "", categorical_cols = ..., missing_values = ..., settings = ...):
        pass
    def display_info(self, df):
        st.markdown("### 🧠 Handling Multi-Collinearity")
        st.markdown(
            """
            **Why care about multicollinearity?**  
            Multicollinearity occurs when two or more independent variables in the dataset are highly correlated.  
            This can:
            - Make model coefficients unstable  
            - Reduce model interpretability  
            - Affect performance of linear models (e.g. Linear/Logistic Regression)
            """
        )

    def run(self, df, settings, saved_configuration_file):
        original_shape = df.shape
        self.display_info(df)

        multicollinear_settings = settings.get("Handle_multicollinearity_methods", {})

        default_method = multicollinear_settings.get(
            "Multicollinear_detection_methods",
            [next(iter(self.multicollinearity_detection_options.keys()))],
        )[0]

        default_threshold = multicollinear_settings.get("Pearson_correlation_th", [0.5])[0]

        with st.expander("📊 Multicollinearity Detection & Handling", expanded=False):
            st.markdown("#### Multicollinearity Detection & Handling in the Dataset")

            with st.expander("🔍 Show Available Detection Techniques"):
                explanations = "\n".join(
                    [
                        f"{i + 1}. **{k}**: {v}"
                        for i, (k, v) in enumerate(
                            self.multicollinearity_detection_options.items()
                        )
                    ]
                )
                st.markdown(f"**Available Techniques:**\n\n{explanations}")

            try:
                default_index = list(self.multicollinearity_detection_options).index(default_method)
            except (ValueError, IndexError):
                default_index = 0

            selected_method = st.selectbox(
                "📌 Select a Multicollinearity Detection Method",
                options=list(self.multicollinearity_detection_options.keys()),
                help="Choose how to detect multicollinearity",
                index=default_index,
            )

            pearson_threshold = st.slider(
                "🎚️ Set Pearson Correlation Threshold",
                min_value=0.0,
                max_value=1.0,
                value=default_threshold,
                step=0.01,
            )

            # Save updated settings
            settings = save_configuration_if_updated(
                config_file_name=saved_configuration_file,
                new_config_data=[selected_method],
                config_data_key="Multicollinear_detection_methods",
            )
            settings = save_configuration_if_updated(
                config_file_name=saved_configuration_file,
                new_config_data=[pearson_threshold],
                config_data_key="Pearson_correlation_th",
            )

            try:
                self.apply_method(
                    df=df,
                    method=selected_method,
                    pearson_threshold=pearson_threshold,
                )
            except Exception as error:
                st.error(
                    f"🚨 Error applying method `{selected_method}`. Details: {repr(error)}"
                )
                st.exception(error)

            st.warning(
                f"📏 Dataset size changed from **{original_shape}** to **{df.shape}** after multicollinearity handling."
            )

        st.write(f"**📐 Shape (Before → After):** {original_shape} → {df.shape}")
        return df, settings

    def apply_method(self, df, method, pearson_threshold):
        if method == "Correlation Matrix (Pearson)":
            self.apply_multicollinear_detection(df, pearson_threshold)
        else:
            st.info("ℹ️ This detection method is not implemented yet.")

    def display_plot(self, fig):
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, caption="Multicollinearity Heatmap", width=480)

    def apply_multicollinear_detection(self, df, threshold):
        try:
            with st.container():
                if st.checkbox("❓ Visualize Correlation Matrix"):
                    st.markdown("### 🔍 Correlation Matrix (Pearson)")
                    corr = df.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    self.display_plot(fig)

            corr = df.corr(numeric_only=True)
            upper_triangle = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))

            high_corr_pairs = [
                (c1, c2, round(upper_triangle.loc[c1, c2], 2))
                for c1 in upper_triangle.columns
                for c2 in upper_triangle.columns
                if c1 != c2
                and not pd.isna(upper_triangle.loc[c1, c2])
                and abs(upper_triangle.loc[c1, c2]) > threshold
            ]

            if high_corr_pairs:
                st.markdown("### ⚠️ Highly Correlated Feature Pairs")
                for c1, c2, val in high_corr_pairs:
                    st.markdown(f"- `{c1}` and `{c2}` → **Correlation**: `{val}`")
            else:
                st.success("✅ No feature pairs exceed the selected correlation threshold.")
        except Exception as e:
            st.error("🚨 Multicollinearity detection failed.")
            st.exception(e)
