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

    def init_parameters_for_col(
        self, df, col="", categorical_cols=..., missing_values=...
    ):
        for col in df.columns:
            defaults = {
                f"pearson_correlation_th_{col}": 0.5,
                f"variance_inflation_th_{col}": 5,
            }

        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = st.session_state[
                    "existing_method_multicollinearity_handler"
                ].get(key, default)

    def run(self, df, settings, saved_configuration_file):
        original_shape = df.shape

        self.display_info(df=df)

        config_list = self.sync_column_config_list(
            settings, "Handle_multicollinearity_methods", df.columns
        )

        for element in config_list:
            col = next(iter(element))
            if col not in st.session_state["existing_method_multicollinearity_handler"]:
                st.session_state["existing_method_multicollinearity_handler"][col] = (
                    element[col]
                )

        self.init_parameters_for_col(df)

        with st.expander("📊 Multicollinearity Detection & Handling", expanded=False):
            st.markdown(
                "<h3>Multicollinearity Detection & Handling in the Dataset</h3>",
                unsafe_allow_html=True,
            )
            help_multicollinearity_detection = "\n".join(
                [
                    f"\n{index+1}- {key}: {value}"
                    for index, (key, value) in enumerate(
                        self.multicollinearity_detection_options.items()
                    )
                ]
            )
            with st.expander(
                "🔍 Show Available Multicollinearity Detection Techniques Options Explanations"
            ):
                st.markdown(
                    f"**👽 Available multicollinearity detection techniques**:\n{help_multicollinearity_detection}"
                )
            for col in df.columns:
                default_method = st.session_state[
                    "existing_method_multicollinearity_handler"
                ].get(col, next(iter(self.multicollinearity_detection_options.keys())))
                default_method = (
                    default_method
                    if default_method
                    else [
                        next(iter(self.multicollinearity_detection_options.keys())),
                        0.5,
                        5,
                    ]
                )
                st.session_state[f"pearson_correlation_th_{col}"] = default_method[1]
                st.session_state[f"variance_inflation_th_{col}"] = default_method[2]

                multicollinearity_detection_method = st.selectbox(
                    f"How to multicollinearity detect in {col}.",
                    options=self.multicollinearity_detection_options.keys(),
                    help=f"Select a method for multicollinearity in {col}.",
                    index=list(self.multicollinearity_detection_options.keys()).index(
                        default_method[0]
                    ),
                    key=f"multicollinearity_detection_selectbox_{col}",
                )

                try:
                    self.apply_method(
                        df=df,
                        col=col,
                        method=multicollinearity_detection_method,
                        type_of_method="multicollinearity_detection",
                    )
                    for element in config_list:
                        if col in element:
                            element[col]=[
                                multicollinearity_detection_method,
                                st.session_state.get(f"pearson_correlation_th_{col}"),
                                st.session_state.get(f"variance_inflation_th_{col}"),

                            ]

                except Exception as error:
                    st.error(
                        f"🚨 Error occurred when detect multicollinearity with {multicollinearity_detection_method} and handle multicollinearity with None at {col}. Error: {repr(error)}"
                    )

                st.warning(
                    f"📏 Dataset size changed from **{original_shape}** to **{df.shape}** after applying multicollinearity handler."
                )

            settings = save_configuration_if_updated(
                config_file_name=saved_configuration_file,
                new_config_data=config_list,
                config_data_key="Handle_multicollinearity_methods",
            )

        st.write(f"**Shape (Before → After):** {original_shape} → {df.shape}")
        return df, settings

    def apply_method(self, df, col, method, type_of_method):
        if type_of_method == "multicollinearity_detection":
            self.apply_multicollinear_detection(fill_method_name=method, df=df, col=col)

    def display_plot(self, col, fig):
        def render_figure(fig, caption):
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.image(buf, caption=caption, width=480)

        render_figure(fig=fig, caption=f"Multicollinearity Figure {col}")

    def apply_multicollinear_detection(self, fill_method_name, df, col):
        try:
            if fill_method_name == "Correlation Matrix (Pearson)":
                    with st.container():
                        visualize = st.checkbox(
                            f"❓ Visualize multicollinearity for '{col}'",
                            key=f"visualize_{col}_by_target_multicollinearity",
                        )

                        if visualize:
                            st.markdown("### 🔍 Correlation Matrix (Pearson)")
                            corr = df.corr(numeric_only=True)

                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                            self.display_plot(col=col, fig=fig)

                    default_th_key = f"pearson_correlation_th_{col}"
                    default_threshold = st.session_state.get(default_th_key, 1.0)

                    threshold = st.slider(
                        "🎚️ Set Pearson Correlation Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_threshold,
                        step=0.01,
                        key=f"set_pearson_th_{col}",
                    )

                    st.session_state[default_th_key] = threshold

                    corr = df.corr(numeric_only=True)
                    upper_triangle = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))

                    high_corr_pairs = [
                        (col1, col2, round(upper_triangle.loc[col1, col2], 2))
                        for col1 in upper_triangle.columns
                        for col2 in upper_triangle.columns
                        if col1 != col2 and not pd.isna(upper_triangle.loc[col1, col2])
                        and abs(upper_triangle.loc[col1, col2]) > threshold
                    ]
                    if high_corr_pairs:
                        st.markdown("### ⚠️ Highly Correlated Feature Pairs")
                        for col1, col2, val in high_corr_pairs:
                            st.markdown(f"- `{col1}` and `{col2}` → **Correlation**: `{val}`")
                    else:
                        st.success("✅ No feature pairs exceed the selected correlation threshold.")
                    st.error(high_corr_pairs)
        except Exception as e:
            return f"🚨 Error during multicollinearity detection for {col}: {repr(e)}. Traceback error: {traceback.format_exc()}"
