# STANDARD MODULES
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
        self.multicollinearity_handler_options = (
            DataPreprocessingOptionsHelperText.get_multicollinearity_handler_options()
        )

    def init_parameters_for_col(
        self, df, col="", categorical_cols=..., missing_values=..., settings=...
    ):
        multicollinear_settings = settings.get("Handle_multicollinearity_methods", {})
        default_detection_method = multicollinear_settings.get(
            "Multicollinear_detection_methods",
            [next(iter(self.multicollinearity_detection_options))],
        )[0]
        pearson_threshold = multicollinear_settings.get(
            "Pearson_correlation_th", [0.5]
        )[0]
        VIF_threshold = multicollinear_settings.get("VIF_th", [10.0])[0]
        default_handler_method = multicollinear_settings.get(
            "Multicollinear_handler_methods",
            [next(iter(self.multicollinearity_handler_options))],
        )[0]
        dropped_columns = multicollinear_settings.get(
            "Multicollinear_dropped_columns", []
        )
        return (
            default_detection_method,
            default_handler_method,
            pearson_threshold,
            VIF_threshold,
            dropped_columns,
        )

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

        (
            default_detection_method,
            default_handler_method,
            pearson_threshold,
            VIF_threshold,
            dropped_columns,
        ) = self.init_parameters_for_col(df=df, settings=settings)

        with st.expander("📊 Multicollinearity Detection & Handling", expanded=False):
            st.markdown("#### Multicollinearity Detection & Handling in the Dataset")

            with st.expander("🔍 Show Available Detection Techniques"):
                explanations_detection = "\n".join(
                    [
                        f"{i + 1}. **{k}**: {v}"
                        for i, (k, v) in enumerate(
                            self.multicollinearity_detection_options.items()
                        )
                    ]
                )
                st.markdown(
                    f"**Available Detection Techniques:**\n\n{explanations_detection}"
                )

            with st.expander("🔍 Show Available Handler Techniques"):
                explanations_handler = "\n".join(
                    [
                        f"{i + 1}. **{k}**: {v}"
                        for i, (k, v) in enumerate(
                            self.multicollinearity_handler_options.items()
                        )
                    ]
                )
                st.markdown(
                    f"**Available Handler Techniques:**\n\n{explanations_handler}"
                )

            try:
                default_detection_index = list(
                    self.multicollinearity_detection_options
                ).index(default_detection_method)
            except (ValueError, IndexError):
                default_detection_index = 0

            try:
                default_handler_index = list(
                    self.multicollinearity_handler_options
                ).index(default_handler_method)
            except (ValueError, IndexError):
                default_handler_index = 0

            try:
                selected_detection_method = st.selectbox(
                    "📌 Select a Multicollinearity Detection Method",
                    options=list(self.multicollinearity_detection_options.keys()),
                    help="Choose how to detect multicollinearity",
                    index=default_detection_index,
                )
                pearson_threshold, VIF_threshold, high_corr_pairs = self.apply_method(
                    df=df,
                    method="multicollinearity_detection",
                    selected_detection_method=selected_detection_method,
                    pearson_threshold=pearson_threshold,
                    VIF_threshold=VIF_threshold,
                )
                selected_handler_method = st.selectbox(
                    "📌 Select a Multicollinearity Handler Method",
                    options=list(self.multicollinearity_handler_options.keys()),
                    help="Choose how to handle multicollinearity",
                    index=default_handler_index,
                )

                df, dropped_columns = self.apply_method(
                    df=df,
                    dropped_columns=dropped_columns,
                    method="multicollinearity_handler",
                    selected_handler_method=selected_handler_method,
                    multicollinear_columns=high_corr_pairs,
                )

                # Save updated settings
                settings = save_configuration_if_updated(
                    config_file_name=saved_configuration_file,
                    new_config_data=[selected_detection_method],
                    config_data_key="Multicollinear_detection_methods",
                )

                settings = save_configuration_if_updated(
                    config_file_name=saved_configuration_file,
                    new_config_data=[selected_handler_method],
                    config_data_key="Multicollinear_handler_methods",
                )

                settings = save_configuration_if_updated(
                    config_file_name=saved_configuration_file,
                    new_config_data=[pearson_threshold],
                    config_data_key="Pearson_correlation_th",
                )

                settings = save_configuration_if_updated(
                    config_file_name=saved_configuration_file,
                    new_config_data=[VIF_threshold],
                    config_data_key="VIF_th",
                )

                settings = save_configuration_if_updated(
                    config_file_name=saved_configuration_file,
                    new_config_data=dropped_columns,
                    config_data_key="Multicollinear_dropped_columns",
                )

            except Exception as error:
                st.error(
                    f"🚨 Error occured at applying detection `{selected_detection_method}` and handler {selected_handler_method}. Details: {repr(error)}"
                )
                st.exception(error)

            st.warning(
                f"📏 Dataset size changed from **{original_shape}** to **{df.shape}** after multicollinearity handling."
            )
        st.write(f"**📐 Shape (Before → After):** {original_shape} → {df.shape}")
        return df, settings

    def apply_method(
        self,
        df,
        method,
        dropped_columns=None,
        multicollinear_columns=None,
        selected_handler_method=None,
        selected_detection_method=None,
        pearson_threshold=None,
        VIF_threshold=None,
    ):
        if method == "multicollinearity_detection":
            return self.apply_multicollinear_detection(
                df, pearson_threshold, VIF_threshold, selected_detection_method
            )
        elif method == "multicollinearity_handler":
            return self.apply_multicollinear_handler(
                df, multicollinear_columns, selected_handler_method, dropped_columns
            )
        else:
            st.info("ℹ️ This detection method is not implemented yet.")
            return pearson_threshold, VIF_threshold, []

    def display_plot(self, fig):
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, caption="Multicollinearity Heatmap", width=480)

    def apply_multicollinear_handler(
        self, df, multicollinear_columns, selected_handler_method, dropped_columns=None
    ):
        dropped_columns = None
        try:
            if selected_handler_method == "Drop One of the Highly Correlated Features":
                dropped_columns = st.multiselect(
                    "Select columns to drop",
                    options=multicollinear_columns,
                    key="Drop_Multicollinear_Columns",
                )
                if dropped_columns:
                    st.info(
                        f"👇 You selected the following columns to drop: {', '.join(dropped_columns)}."
                    )
                    df = df.drop(columns=dropped_columns)
                else:
                    st.write("No columns selected for dropping.")

            elif selected_handler_method == "Apply Dimensionality Reduction":
                pass
            elif selected_handler_method == "Combine Correlated Columns":
                pass
        except Exception as e:
            st.error(
                f"🚨 Multicollinearity handler failed using {selected_handler_method}."
            )
            st.exception(e)
        return df, dropped_columns

    def apply_multicollinear_detection(
        self, df, pearson_threshold, VIF_threshold, selected_detection_method
    ):
        high_corr_pairs = []

        try:
            if selected_detection_method == "Correlation Matrix (Pearson)":
                with st.container():
                    if st.checkbox("❓ Visualize Correlation Matrix"):
                        st.markdown("### 🔍 Correlation Matrix (Pearson)")
                        corr = df.corr(numeric_only=True)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                        self.display_plot(fig)

                    pearson_threshold = st.slider(
                        "🎚️ Set Pearson Correlation Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=pearson_threshold,
                        step=0.01,
                    )

                    corr = df.corr(numeric_only=True)
                    upper_triangle = corr.where(
                        ~np.tril(np.ones(corr.shape)).astype(bool)
                    )

                    high_corr_pairs = [
                        (c1, c2, round(upper_triangle.loc[c1, c2], 2))
                        for c1 in upper_triangle.columns
                        for c2 in upper_triangle.columns
                        if c1 != c2
                        and not pd.isna(upper_triangle.loc[c1, c2])
                        and abs(upper_triangle.loc[c1, c2]) > pearson_threshold
                    ]

                    if high_corr_pairs:
                        st.markdown("### ⚠️ Highly Correlated Feature Pairs")
                        for c1, c2, val in high_corr_pairs:
                            st.markdown(
                                f"- `{c1}` and `{c2}` → **Correlation**: `{val}`"
                            )
                        high_corr_pairs = [
                            column[i] for column in high_corr_pairs for i in (0, 1)
                        ]

                    else:
                        st.success(
                            "✅ No feature pairs exceed the selected correlation threshold."
                        )

            elif selected_detection_method == "Variance Inflation Factor (VIF)":
                with st.container():
                    VIF_threshold = st.slider(
                        "🎚️ Set VIF Threshold",
                        min_value=0.0,
                        max_value=30.0,
                        value=VIF_threshold,
                        step=0.01,
                    )

                    vif_data = pd.DataFrame()
                    vif_data["Feature"] = df.columns
                    vif_data["VIF"] = [
                        variance_inflation_factor(df.values, i)
                        for i in range(df.shape[1])
                    ]

                    st.subheader("📊 VIF Values for Features")
                    st.dataframe(vif_data)

                    high_corr_pairs = vif_data[vif_data["VIF"] > VIF_threshold]
                    high_corr_pairs = list(high_corr_pairs["Feature"])

                    if high_corr_pairs:
                        st.warning("⚠️ Features exceeding the VIF threshold:")
                        st.dataframe(high_corr_pairs)
                    else:
                        st.success("✅ No features exceed the selected VIF threshold.")

        except Exception as e:
            st.error(
                f"🚨 Multicollinearity detection failed using {selected_detection_method}."
            )
            st.exception(e)

        return pearson_threshold, VIF_threshold, high_corr_pairs
