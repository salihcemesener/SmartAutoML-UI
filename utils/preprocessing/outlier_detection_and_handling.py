# STANDARD MODULES
import traceback
from io import BytesIO

import numpy as np
import streamlit as st
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer, RobustScaler, QuantileTransformer

# USER MODULES
from utils.settings_manager import save_configuration_if_updated
from utils.plot_utils.plotting import feature_outlier_analysis_plot
from utils.preprocessing.data_preprocessor_abc import DataPreprocessorHandler
from utils.preprocessing.data_preprocessing_helper_text import (
    DataPreprocessingOptionsHelperText,
)


class OutlierDetectionAndHandler(DataPreprocessorHandler):
    def __init__(self):
        self.detection_options = (
            DataPreprocessingOptionsHelperText.list_outlier_detection_strategies()
        )
        self.handler_options = (
            DataPreprocessingOptionsHelperText.get_outlier_handler_options()
        )
        st.session_state.setdefault("existing_method_remove_outliers", {})

    def display_info(self, df):
        st.header("Outlier Detection and Handlers")
        st.write(
            """
            Outliers are extreme values that lie far outside the typical data range and
            can distort model training. By spotting these anomalies and choosing to
            exclude them or reduce their impact, you help your model focus on the core
            patterns in your data.
        """
        )
        st.subheader("Statistical Overview")
        st.dataframe(df.describe(include="all").transpose())

    def display_plot(self, df, col):
        def render(fig, caption, container):
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", bbox_inches="tight")
            container.image(buf, caption=caption, width=480)

        if st.checkbox(
            f"❓ Visualize '{col}' by target", key=f"visualize_{col}_by_target"
        ):
            figs = feature_outlier_analysis_plot(
                df=df, col=col, target=st.session_state["remove_outliers_target_col"]
            )
            for i in range(0, len(figs), 2):
                cols = st.columns(2)
                for j in range(2):
                    idx = i + j
                    if idx < len(figs):
                        render(figs[idx], f"Figure {idx + 1}", cols[j])

    def set_target_column(self, label, columns):
        st.session_state["remove_outliers_target_col"] = st.selectbox(
            label=label,
            options=columns,
            index=list(columns).index(
                st.session_state.get("remove_outliers_target_col", columns[0])
            ),
            key="remove_outliers_target_col",
        )

    def init_parameters_for_col(self, df, col):
        if "remove_outliers_target_col" not in st.session_state:
            st.session_state["remove_outliers_target_col"] = st.session_state[
                "existing_method_remove_outliers"
            ].get("remove_outliers_target_col", df.columns[0])

        defaults = {
            f"z_score_th_{col}": 3.0,
            f"IQR_multiplier_{col}": 1.5,
            f"MAD_scale_factor_{col}": 1.4826,
            f"MAD_th_{col}": 3.0,
            f"Winsorization_upper_percentile_{col}": 95.0,
            f"Winsorization_lower_percentile_{col}": 5.0,
            f"Isolation_Forest_n_estimators_{col}": 100.0,
            f"Isolation_Forest_max_samples_{col}": "auto",
            f"Isolation_Forest_contamination_{col}": 0.05,
            f"Isolation_Forest_max_features_{col}": 5.0,
            f"LOF_n_neighbors_{col}": 20.0,
            f"LOF_contamination_{col}": 0.05,
            f"LOF_metric_{col}": "euclidean",
            f"Handling_method_to_remove_outliers_{col}": "remove",
            f"Winsorization_Handler_upper_percentile_{col}": 95.0,
            f"Winsorization_Handler_lower_percentile_{col}": 5.0,
        }

        for key, val in defaults.items():
            st.session_state.setdefault(
                key, st.session_state["existing_method_remove_outliers"].get(key, val)
            )

    def apply_method(
        self, df, col, method=None, method_type="", detected_outlier_indices=[]
    ):
        if method_type == "outlier_detection":
            return self.apply_outlier_detection(method, df, col)
        elif method_type == "outlier_handling":
            return self.apply_outlier_handler(
                df,
                col,
                detected_outlier_indices,
                st.session_state[f"Handling_method_to_remove_outliers_{col}"],
            )

    def run(self, df, settings, saved_configuration_file):
        original_shape = df.shape
        self.display_info(df)
        config_list = self.sync_column_config_list(
            settings, "Remove_outliers_handle_methods", df.columns
        )

        # for element in config_list:
        #     col = next(iter(element))
        #     st.session_state["existing_method_remove_outliers"].setdefault(
        #         col, element[col]
        #     )

        for col in df.columns:
            self.init_parameters_for_col(df, col)

        st.session_state["selected_target_col"] = st.selectbox(
            "Visualize outliers by using target as:",
            options=df.columns.tolist(),
            index=df.columns.get_loc(
                st.session_state.get("remove_outliers_target_col", df.columns[0])
            ),
            key="remove_outliers_target_col",
        )

        with st.expander("📈 Outlier Detection & Handling", expanded=False):
            self.show_explanations()

            for col in df.columns:
                self.display_plot(df, col)
                detect_method = st.selectbox(
                    f"Select outlier detection method for {col}",
                    options=self.detection_options.keys(),
                    key=f"outlier_detection_selectbox_{col}",
                )
                remove_method = st.selectbox(
                    f"Select outlier handling method for {col}",
                    options=self.handler_options.keys(),
                    key=f"outlier_removal_selectbox_{col}",
                )

                try:
                    detect_msg, outlier_idx = self.apply_method(
                        df, col, detect_method, "outlier_detection"
                    )
                    df, remove_msg = self.apply_method(
                        df,
                        col,
                        method_type="outlier_handling",
                        detected_outlier_indices=outlier_idx,
                    )
                    st.info(
                        f"**Detection:** {detect_msg}\n\n**Handling:** {remove_msg}"
                    )
                    st.write(
                        f"📊 Range of `{col}` after handling: min = {df[col].min()}, max = {df[col].max()}"
                    )
                except Exception as e:
                    st.error(f"❌ Failed on `{col}` using `{detect_method}`: {repr(e)}")

            settings = save_configuration_if_updated(
                saved_configuration_file, config_list, "Remove_outliers_handle_methods"
            )
            if df.shape != original_shape:
                st.warning(f"⚠️ Data shape changed from {original_shape} to {df.shape}")

        return df, settings

    def show_explanations(self):
        with st.expander("🔍 Outlier Detection Techniques"):
            for i, (key, desc) in enumerate(self.detection_options.items(), 1):
                st.markdown(f"{i}. **{key}**: {desc}")
        with st.expander("🔍 Outlier Handling Techniques"):
            for i, (key, desc) in enumerate(self.handler_options.items(), 1):
                st.markdown(f"{i}. **{key}**: {desc}")
        with st.expander("📘 Boxplot Interpretation"):
            st.image("data/ui_images/boxplot_explanation.png", width=512)
            st.markdown(
                """
                - **Median**: middle value
                - **Box edges**: Q1 & Q3 (25th & 75th percentile)
                - **Whiskers**: data range within 1.5 * IQR
                - **Dots**: potential outliers
            """
            )

    def apply_outlier_detection(self, fill_method_name, df, col):
        try:
            series = df[col]
            outlier_indices = []

            if fill_method_name == "Do Nothing":
                return (
                    f"ℹ️ No detection performed on `{col}` — method is set to 'Do Nothing'.",
                    outlier_indices,
                )

            elif fill_method_name == "Z-Score Method (Standard Score)":
                threshold = st.session_state.get(f"z_score_th_{col}", 3.0)
                threshold = st.slider(
                    "Set Z-score threshold:",
                    min_value=0.0,
                    max_value=100.0,
                    value=threshold,
                    step=1.0,
                    key=f"set_z_score_{col}",
                )
                st.session_state[f"z_score_th_{col}"] = threshold
                z_scores = zscore(series)
                outlier_indices = df.index[np.abs(z_scores) > threshold].tolist()
                return (
                    f"📏 Z-Score: Detected {len(outlier_indices)} outliers (threshold = {threshold}).",
                    outlier_indices,
                )

            elif fill_method_name == "Interquartile Range (IQR) Method":
                multiplier = st.session_state.get(f"IQR_multiplier_{col}", 1.5)
                multiplier = st.slider(
                    "Set IQR multiplier:",
                    min_value=0.0,
                    max_value=100.0,
                    value=multiplier,
                    step=1.0,
                    key=f"set_IQR_multiplier_{col}",
                )
                st.session_state[f"IQR_multiplier_{col}"] = multiplier
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR
                outlier_indices = df.index[(series < lower) | (series > upper)].tolist()
                return (
                    f"📦 IQR: Detected {len(outlier_indices)} outliers.",
                    outlier_indices,
                )

            elif fill_method_name == "Median Absolute Deviation (MAD)":
                scale_factor = st.session_state.get(f"MAD_scale_factor_{col}", 1.4826)
                threshold = st.session_state.get(f"MAD_th_{col}", 3.0)
                scale_factor = st.slider(
                    "Set MAD scale factor:",
                    min_value=0.1,
                    max_value=5.0,
                    value=scale_factor,
                    step=0.1,
                    key=f"set_mad_scale_{col}",
                )
                threshold = st.slider(
                    "Set MAD threshold:",
                    min_value=0.0,
                    max_value=10.0,
                    value=threshold,
                    step=0.5,
                    key=f"set_mad_th_{col}",
                )
                st.session_state[f"MAD_scale_factor_{col}"] = scale_factor
                st.session_state[f"MAD_th_{col}"] = threshold
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z = np.abs(series - median) / (mad * scale_factor + 1e-8)
                outlier_indices = df.index[modified_z > threshold].tolist()
                return (
                    f"📐 MAD: Detected {len(outlier_indices)} outliers.",
                    outlier_indices,
                )

            elif fill_method_name == "Winsorization (Percentile Capping)":
                lower_pct = st.session_state.get(
                    f"Winsorization_lower_percentile_{col}", 5.0
                )
                upper_pct = st.session_state.get(
                    f"Winsorization_upper_percentile_{col}", 95.0
                )

                lower_pct = st.slider(
                    "Set lower percentile:",
                    min_value=0.0,
                    max_value=49.0,
                    value=lower_pct,
                    step=1.0,
                    key=f"set_winsor_lower_{col}",
                )
                upper_pct = st.slider(
                    "Set upper percentile:",
                    min_value=51.0,
                    max_value=100.0,
                    value=upper_pct,
                    step=1.0,
                    key=f"set_winsor_upper_{col}",
                )

                st.session_state[f"Winsorization_lower_percentile_{col}"] = lower_pct
                st.session_state[f"Winsorization_upper_percentile_{col}"] = upper_pct

                lower = np.percentile(series, lower_pct)
                upper = np.percentile(series, upper_pct)
                outlier_indices = df.index[(series < lower) | (series > upper)].tolist()
                return (
                    f"🔧 Winsorization: Detected {len(outlier_indices)} values outside {lower_pct:.0f}–{upper_pct:.0f} percentile caps.",
                    outlier_indices,
                )

            elif fill_method_name == "Isolation Forest (Unsupervised ML)":
                contamination = st.session_state.get(
                    f"Isolation_Forest_contamination_{col}", 0.05
                )
                max_features = st.session_state.get(
                    f"Isolation_Forest_max_features_{col}", 1.0
                )

                contamination = st.slider(
                    "Set contamination rate:",
                    min_value=0.0,
                    max_value=0.5,
                    value=float(contamination),
                    step=0.01,
                    key=f"set_isof_cont_{col}",
                )
                max_features = st.slider(
                    "Set max features:",
                    min_value=1.0,
                    max_value=float(series.shape[0]),
                    value=max_features,
                    step=0.01,
                    key=f"set_isof_features_{col}",
                )
                st.session_state[f"Isolation_Forest_contamination_{col}"] = (
                    contamination
                )
                st.session_state[f"Isolation_Forest_max_features_{col}"] = max_features

                model = IsolationForest(
                    contamination=contamination,
                    max_features=max_features / series.shape[0],
                    random_state=42,
                )
                preds = model.fit_predict(series.values.reshape(-1, 1))
                outlier_indices = df.index[preds == -1].tolist()
                return (
                    f"🌲 Isolation Forest: Detected {len(outlier_indices)} outliers (contamination = {contamination}).",
                    outlier_indices,
                )

            elif fill_method_name == "Local Outlier Factor (LOF)":
                contamination = st.session_state.get(f"LOF_contamination_{col}", 0.05)
                n_neighbors = st.session_state.get(f"LOF_n_neighbors_{col}", 20.0)
                metric = st.session_state.get(f"LOF_metric_{col}", "euclidean")

                contamination = st.slider(
                    "Set contamination rate:",
                    min_value=0.0,
                    max_value=0.5,
                    value=float(contamination),
                    step=0.01,
                    key=f"set_lof_cont_{col}",
                )
                n_neighbors = st.slider(
                    "Set number of neighbors:",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(n_neighbors),
                    step=0.01,
                    key=f"set_lof_neighbors_{col}",
                )

                metric = st.selectbox(
                    f"📐 Select distance metric for LOF to detect outliers in `{col}`:",
                    options=[
                        "euclidean",
                        "manhattan",
                        "chebyshev",
                        "minkowski",
                        "cosine",
                    ],
                    help="Choose the distance metric used in LOF (Local Outlier Factor). Affects how outlier distances are calculated.",
                    index=[
                        "euclidean",
                        "manhattan",
                        "chebyshev",
                        "minkowski",
                        "cosine",
                    ].index(st.session_state.get(f"LOF_metric_{col}", "euclidean")),
                    key=f"lof_metric_selectbox_{col}",
                )

                st.session_state[f"LOF_contamination_{col}"] = contamination
                st.session_state[f"LOF_n_neighbors_{col}"] = n_neighbors
                st.session_state[f"LOF_metric_{col}"] = metric
                lof = LocalOutlierFactor(
                    contamination=contamination,
                    n_neighbors=int(n_neighbors),
                    metric=metric,
                )
                preds = lof.fit_predict(series.values.reshape(-1, 1))
                outlier_indices = df.index[preds == -1].tolist()
                return (
                    f"📍 LOF: Detected {len(outlier_indices)} outliers (neighbors = {n_neighbors}).",
                    outlier_indices,
                )

            else:
                return (
                    f"⚠️ No valid outlier detection method selected for `{col}`.",
                    outlier_indices,
                )

        except Exception as e:
            return (
                f"🚨 Error during outlier detection for `{col}`: {repr(e)}. Traceback error: {traceback.format_exc()}",
                outlier_indices,
            )

    def apply_outlier_handler(self, df, col, outlier_indices, method="remove"):
        if not outlier_indices:
            return df, f"No outliers to handle in `{col}`."

        if method == "skip":
            return df, f"Outliers not handled in `{col}`."

        if method == "remove":
            df = df.drop(index=outlier_indices)
            return df, f"✅ Removed {len(outlier_indices)} outliers from `{col}`."

        elif method == "impute_median":
            median = df[col].median()
            df.loc[outlier_indices, col] = median
            return (
                df,
                f"🩺 Imputed {len(outlier_indices)} outliers in `{col}` with median: {median:.3f}",
            )

        elif method == "impute_mean":
            mean = df[col].mean()
            df.loc[outlier_indices, col] = mean
            return (
                df,
                f"🧪 Imputed {len(outlier_indices)} outliers in `{col}` with mean: {mean:.3f}",
            )

        elif method == "mark":
            if f"{col}_is_outlier" not in df.columns:
                df[f"{col}_is_outlier"] = False
            df.loc[outlier_indices, f"{col}_is_outlier"] = True
            return (
                df,
                f"🏷️ Flagged {len(outlier_indices)} outliers in new column `{col}_is_outlier`.",
            )

        elif method == "log":
            if (df[col] < 0).any():
                return (
                    df,
                    f"❌ Log transform cannot be applied to negative values in `{col}`.",
                )
            df[col] = np.log1p(df[col])
            return (
                df,
                f"📉 Applied log(x+1) transform on `{col}` to compress large values.",
            )

        elif method == "sqrt":
            if (df[col] < 0).any():
                return (
                    df,
                    f"❌ Square root transform cannot be applied to negative values in `{col}`.",
                )
            df[col] = np.sqrt(df[col])
            return (
                df,
                f"🟪 Applied square root transform on `{col}` for mild compression.",
            )

        elif method == "box-cox":
            try:
                pt = PowerTransformer(method="box-cox", standardize=True)
                df[col] = pt.fit_transform(df[[col]])
                return df, f"📦 Applied Box-Cox transformation on `{col}`."
            except Exception as e:
                return (
                    df,
                    f"❌ Box-Cox failed for `{col}`: {repr(e)} (Requires all values > 0).",
                )

        elif method == "yeo-johnson":
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                df[col] = pt.fit_transform(df[[col]])
                return (
                    df,
                    f"🔁 Applied Yeo-Johnson transformation on `{col}` (handles negatives).",
                )
            except Exception as e:
                return df, f"❌ Yeo-Johnson failed for `{col}`: {repr(e)}."

        elif method == "robust-scaler":
            try:
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(df[[col]])
                return df, f"⚖️ Applied RobustScaler on `{col}` (uses median and IQR)."
            except Exception as e:
                return df, f"❌ RobustScaler failed for `{col}`: {repr(e)}."

        elif method == "quantile":
            try:
                qt = QuantileTransformer(output_distribution="normal")
                df[col] = qt.fit_transform(df[[col]])
                return (
                    df,
                    f"📊 Applied QuantileTransformer on `{col}` to normalize distribution.",
                )
            except Exception as e:
                return df, f"❌ Quantile transform failed for `{col}`: {repr(e)}."

        elif method == "winsorize":
            try:
                lower_percentile = st.session_state.get(
                    f"Winsorization_Handler_lower_percentile_{col}", 5.0
                )
                upper_percentile = st.session_state.get(
                    f"Winsorization_Handler_upper_percentile_{col}", 95.0
                )

                lower_percentile = st.slider(
                    "Set lower percentile cap (e.g., 5 means values below 5th percentile will be capped):",
                    min_value=0.0,
                    max_value=49.0,
                    value=float(lower_percentile),
                    step=1.0,
                    key=f"set_winsorize_handler_lower_{col}",
                )
                upper_percentile = st.slider(
                    "Set upper percentile cap (e.g., 95 means values above 95th percentile will be capped):",
                    min_value=51.0,
                    max_value=100.0,
                    value=float(upper_percentile),
                    step=1.0,
                    key=f"set_winsorize_handler_upper_{col}",
                )

                st.session_state[f"Winsorization_Handler_lower_percentile_{col}"] = (
                    lower_percentile
                )
                st.session_state[f"Winsorization_Handler_upper_percentile_{col}"] = (
                    upper_percentile
                )

                lower_prop = lower_percentile / 100.0
                upper_prop = 1 - (upper_percentile / 100.0)

                df[col] = winsorize(df[col], limits=(lower_prop, upper_prop))

                return (
                    df,
                    f"🔧 Winsorized `{col}` using {lower_percentile:.0f}–{upper_percentile:.0f} percentile caps.",
                )

            except Exception as e:
                return df, f"❌ Winsorization failed for `{col}`: {repr(e)}."
        return df, f"⚠️ Unknown outlier handling method `{method}`."
