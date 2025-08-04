# STANDARD MODULES
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# USER MODULES
from utils.settings_manager import save_configuration_if_updated
from utils.preprocessing.data_preprocessor_abc import DataPreprocessorHandler
from utils.preprocessing.data_preprocessing_helper_text import DataPreprocessingOptionsHelperText


class MultiCollinearityHandler(DataPreprocessorHandler):
    def __init__(self):
        self.detection_options = DataPreprocessingOptionsHelperText.list_multicollinearity_detection_options()
        self.handler_options = DataPreprocessingOptionsHelperText.get_multicollinearity_handler_options()
        self.combine_methods = ["mean", "diff", "prod"]
        self.target_col = None
   

    def init_parameters_for_col(self, df, col="", categorical_cols=..., missing_values=..., settings=...):
        return (
            settings.get("Multicollinear_detection_methods", [next(iter(self.detection_options))])[0],
            settings.get("Multicollinear_handler_methods", [next(iter(self.handler_options))])[0],
            settings.get("Pearson_correlation_th", [0.5])[0],
            settings.get("VIF_th", [10.0])[0],
            settings.get("Multicollinear_dropped_columns", []),
            settings.get("Multicollinear_combine_correlated_method", [self.combine_methods[0]])[0],
            settings.get("Multicollinear_target_column", [next(iter(df.columns))][0])
        )

    def display_info(self):
        st.markdown("### 🧠 Handling Multi-Collinearity")
        st.markdown("""
            **Why care about multicollinearity?**  
            It occurs when independent variables are highly correlated, causing:
            - Unstable model coefficients
            - Reduced interpretability
            - Poor performance in linear models
        """)

    def run(self, df, settings, saved_configuration_file):
        original_shape = df.shape
        self.display_info()

        (
            detection_method,
            handler_method,
            pearson_th,
            vif_th,
            dropped_cols,
            combine_method,
            self.target_col
        ) = self.init_parameters_for_col(df=df, settings=settings)

        self.target_col = st.selectbox(
            label="🎯 Select the Target Column (Will Not Be Modified)",
            options=df.columns.tolist(),
            index=list(df.columns).index(self.target_col),
            key="target_col_multicollinear",
            help=(
                "This column will be treated as the target variable and will remain unchanged. "
            ),
        )
        with st.expander("📊 Multicollinearity Detection & Handling", expanded=False):
            self._show_option_descriptions()

            try:
                
                selected_detection = st.selectbox(
                    "📌 Select Detection Method",
                    options=list(self.detection_options),
                    index=list(self.detection_options).index(detection_method)
                )

                pearson_th, vif_th, multicollinear_pairs = self.apply_method(
                    df=df,
                    method="multicollinearity_detection",
                    selected_detection_method=selected_detection,
                    pearson_threshold=pearson_th,
                    VIF_threshold=vif_th
                )

                selected_handler = st.selectbox(
                    "📌 Select Handler Method",
                    options=list(self.handler_options),
                    index=list(self.handler_options).index(handler_method)
                )

                df, dropped_cols, combine_method = self.apply_method(
                    df=df,
                    method="multicollinearity_handler",
                    selected_handler_method=selected_handler,
                    multicollinear_columns=multicollinear_pairs,
                    default_combine_correlated_column_method=combine_method,
                    dropped_columns=dropped_cols,
                    selected_detection_method=selected_detection
                )

                for key, val in {
                    "Multicollinear_detection_methods": [selected_detection],
                    "Multicollinear_handler_methods": [selected_handler],
                    "Pearson_correlation_th": [pearson_th],
                    "VIF_th": [vif_th],
                    "Multicollinear_dropped_columns": dropped_cols,
                    "Multicollinear_combine_correlated_method": combine_method,
                    "Multicollinear_target_column":self.target_col,
                }.items():
                    settings = save_configuration_if_updated(saved_configuration_file, val, key)

            except Exception as e:
                st.error(f"🚨 Error applying detection `{detection_method}` or handler `{handler_method}`.")
                st.exception(e)

            st.warning(f"📏 Dataset size changed: **{original_shape} → {df.shape}**")
            st.write(f"**📐 Final Shape:** {df.shape}")

        return df, settings

    def _show_option_descriptions(self):
        with st.expander("🔍 Show Available Detection Techniques"):
            for i, (name, desc) in enumerate(self.detection_options.items(), 1):
                st.markdown(f"{i}. **{name}**: {desc}")

        with st.expander("🔧 Show Available Handler Techniques"):
            for i, (name, desc) in enumerate(self.handler_options.items(), 1):
                st.markdown(f"{i}. **{name}**: {desc}")

    def apply_method(self, df, method, **kwargs):
        if method == "multicollinearity_detection":
            return self.detect_multicollinearity(df, **kwargs)
        elif method == "multicollinearity_handler":
            return self.handle_multicollinearity(df, **kwargs)
        else:
            st.info("ℹ️ Method not implemented.")
            return kwargs.get("pearson_threshold"), kwargs.get("VIF_threshold"), []

    def detect_multicollinearity(self, df, pearson_threshold, VIF_threshold, selected_detection_method):
        correlated_pairs = []

        try:
            if selected_detection_method == "Correlation Matrix (Pearson)":
                if st.checkbox("📊 Visualize Correlation Matrix"):
                    st.markdown("### 🔍 Correlation Matrix")
                    corr = df.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    self.display_plot(fig)

                pearson_threshold = st.slider("🎚️ Pearson Threshold", 0.0, 1.0, pearson_threshold, 0.01)
                corr = df.corr(numeric_only=True)
                upper_triangle = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))

                correlated_pairs = [
                    (c1, c2, round(upper_triangle.loc[c1, c2], 2))
                    for c1 in upper_triangle.columns
                    for c2 in upper_triangle.columns
                    if c1 != c2 and not pd.isna(upper_triangle.loc[c1, c2])
                    and abs(upper_triangle.loc[c1, c2]) > pearson_threshold
                ]

                if correlated_pairs:
                    st.markdown("### ⚠️ Highly Correlated Feature Pairs")
                    for c1, c2, val in correlated_pairs:
                        st.markdown(f"- `{c1}` and `{c2}` → **Correlation**: `{val}`")
                else:
                    st.success("✅ No pairs exceed the threshold.")

            elif selected_detection_method == "Variance Inflation Factor (VIF)":
                vif_threshold = st.slider("🎚️ VIF Threshold", 0.0, 30.0, VIF_threshold, 0.01)
                vif_data = pd.DataFrame({
                    "Feature": df.columns,
                    "VIF": [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
                })
                st.dataframe(vif_data)
                correlated_pairs = list(vif_data[vif_data["VIF"] > vif_threshold]["Feature"])

                if correlated_pairs:
                    st.warning("⚠️ Features exceeding VIF threshold:")
                    st.dataframe(pd.DataFrame(correlated_pairs, columns=["Feature"]))
                else:
                    st.success("✅ All VIF values are within acceptable range.")

        except Exception as e:
            st.error(f"🚨 Detection failed using method: {selected_detection_method}")
            st.exception(e)

        return pearson_threshold, VIF_threshold, correlated_pairs

    def handle_multicollinearity(
        self, df, multicollinear_columns, selected_handler_method,
        selected_detection_method, dropped_columns=None,
        default_combine_correlated_column_method=None
    ):
        if not multicollinear_columns:
            st.success("✅ No multicollinear columns detected — skipping handling step.")
            return df, dropped_columns, default_combine_correlated_column_method
        try:
            if selected_handler_method == "Drop One of the Highly Correlated Features":
                if isinstance(multicollinear_columns[0], tuple):
                    multicollinear_columns = list({
                        col for pair in multicollinear_columns for col in pair if isinstance(col, str)
                    })
                multicollinear_columns.remove(self.target_col)
                dropped_columns = st.multiselect(
                    "Select columns to drop", options=multicollinear_columns, key="Drop_Multicollinear_Columns",default=dropped_columns
                )

                if dropped_columns:
                    df.drop(columns=dropped_columns, inplace=True)
                    st.info(f"✅ Dropped columns: {', '.join(dropped_columns)}")
                else:
                    st.write("⚠️ No columns selected for dropping.")

            elif selected_handler_method == "Combine Correlated Columns":
                if selected_detection_method == "Variance Inflation Factor (VIF)":
                    st.warning("❗ VIF does not return column pairs. Use correlation-based detection for combination.")
                    return df, dropped_columns, default_combine_correlated_column_method

                default_combine_correlated_column_method = st.selectbox(
                    "📌 Select combination method", options=self.combine_methods,
                    index=self.combine_methods.index(default_combine_correlated_column_method)
                )

                for idx, group in enumerate(multicollinear_columns):
                    group = [col for col in group if isinstance(col, str)]
                    df_subset = df[group]
                    col_prefix = f"group_{group[0]}_{group[1]}_{idx+1}"
                    if self.target_col in group:
                        st.warning(
                            f"⚠️ Skipping group `{group}` because it includes the target column `{self.target_col}`. "
                            "This group will remain unchanged to avoid modifying the target."
                        )
                        continue
                    if default_combine_correlated_column_method == "mean":
                        df[f"{col_prefix}_mean"] = df_subset.mean(axis=1)
                    elif default_combine_correlated_column_method == "diff":
                        df[f"{col_prefix}_diff"] = df_subset.max(axis=1) - df_subset.min(axis=1)
                    elif default_combine_correlated_column_method == "prod":
                        df[f"{col_prefix}_prod"] = df_subset.prod(axis=1)

                    df.drop(columns=group, inplace=True)

                st.success("✅ Correlated columns combined successfully.")
                st.dataframe(df)

        except Exception as e:
            st.error(f"🚨 Handler method failed: {selected_handler_method}")
            st.exception(e)

        return df, dropped_columns, default_combine_correlated_column_method

    def display_plot(self, fig):
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, caption="Multicollinearity Heatmap", width=480)