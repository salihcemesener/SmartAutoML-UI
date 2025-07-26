# STANDARD MODULES
import pandas as pd
import streamlit as st
import category_encoders as ce
from pandas.api.types import is_object_dtype
from sklearn.preprocessing import LabelEncoder

# USER MODULES
from utils.settings_manager import save_configuration_if_updated
from utils.preprocessing.data_preprocessor_abc import DataPreprocessorHandler
from utils.preprocessing.data_preprocessing_helper_text import (
    DataPreprocessingOptionsHelperText,
)
from utils.preprocessing.encoding_methods import (
    target_encoding,
    weighted_target_encoding,
    k_fold_target_encoding,
)


class CategoricalValueHandler(DataPreprocessorHandler):
    def __init__(self):
        self.encoding_options = (
            DataPreprocessingOptionsHelperText.get_categorical_encoding_options()
        )
        st.session_state.setdefault("existing_method_categorical_values", {})

    def display_info(self, df):
        st.markdown("<h3>Handling Categorical Data</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            Categorical columns need to be encoded for machine learning models. Below are columns detected:
        """
        )
        categorical_cols = [col for col in df.columns if is_object_dtype(df[col])]
        summary = pd.DataFrame(
            {
                "Column Name": categorical_cols,
                "# Unique Values": df[categorical_cols].nunique(),
            }
        ).astype(str)
        st.dataframe(summary, use_container_width=True)
        return categorical_cols

    def init_parameters_for_col(self, df, categorical_cols):
        for col in categorical_cols:
            st.session_state.setdefault(f"num_folds_{col}", 5)
            st.session_state.setdefault(f"target_col_{col}", df.columns[0])
            st.session_state.setdefault(f"smoothing_factor_{col}", 10.0)
            st.session_state.setdefault(f"variable_order_{col}", None)

    def run(self, df, settings, saved_configuration_file):
        categorical_cols = self.display_info(df)
        if not categorical_cols:
            st.success("No categorical columns detected.")
            return df, settings

        config_list = self.sync_column_config_list(
            settings, "Categorical_value_handle_methods", categorical_cols
        )

        self.init_parameters_for_col(df, categorical_cols)

        # for element in config_list:
        #     col = next(iter(element))

        #     if col not in st.session_state["existing_method_categorical_values"]:
        #         st.session_state["existing_method_categorical_values"][col] = element[
        #             col
        #         ]
        with st.expander("Convert Categorical to Numeric Data", expanded=False):
            st.markdown("<h3>Convert Categorical Columns</h3>", unsafe_allow_html=True)
            st.markdown(
                """Select a method to convert each categorical column into numeric format."""
            )

            for col in categorical_cols:
                st.info(f"Column **{col}** has {df[col].nunique()} unique values.")
                default = st.session_state["existing_method_categorical_values"].get(
                    col,
                    [
                        next(iter(self.encoding_options.keys())),
                        5,
                        df.columns[0],
                        10.0,
                        None,
                    ],
                )
                method = st.selectbox(
                    f"Encoding method for {col}",
                    options=list(self.encoding_options.keys()),
                    index=list(self.encoding_options.keys()).index(default[0]),
                    key=f"select_encoding_method_{col}",
                )

                st.session_state[f"num_folds_{col}"] = default[1]
                st.session_state[f"target_col_{col}"] = default[2]
                st.session_state[f"smoothing_factor_{col}"] = default[3]
                st.session_state[f"variable_order_{col}"] = default[4]

                df = self.apply_method(df, col, method)
                for element in config_list:
                    if col in element:
                        element[col] = [
                            method,
                            st.session_state[f"num_folds_{col}"],
                            st.session_state[f"target_col_{col}"],
                            st.session_state[f"smoothing_factor_{col}"],
                            st.session_state[f"variable_order_{col}"],
                        ]

            settings = save_configuration_if_updated(
                config_file_name=saved_configuration_file,
                new_config_data=config_list,
                config_data_key="Categorical_value_handle_methods",
            )

        return df, settings

    def apply_method(self, df, col, method):
        columns = df.columns
        values = df[col].dropna().unique().tolist()

        if method == "Label Encoding":
            df[col] = LabelEncoder().fit_transform(df[col])

        elif method == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=[col], drop_first=True)

        elif method == "Frequency Encoding":
            df[col] = df[col].map(df[col].value_counts())

        elif method == "Ordinal Encoding":
            if len(values) > 10:
                confirm = st.checkbox(
                    f"🚨 High cardinality for `{col}`. Proceed with Ordinal Encoding?",
                    key=f"ordinal_confirm_{col}",
                )
                if not confirm:
                    return df
            selected_order = st.multiselect(
                f"Set custom order for `{col}`:",
                values,
                default=st.session_state[f"variable_order_{col}"] or values,
                key=f"order_select_{col}",
            )
            st.session_state[f"variable_order_{col}"] = selected_order
            mapping = {val: i for i, val in enumerate(selected_order)}
            df[col] = df[col].map(mapping)

        elif method == "Binary Encoding":
            df = ce.BinaryEncoder(cols=[col]).fit_transform(df)

        elif method == "Target Encoding":
            st.session_state[f"target_col_{col}"] = st.selectbox(
                f"Target for encoding `{col}`:",
                columns,
                key=f"target_column_select_{col}",
            )
            df = target_encoding(df, col, st.session_state[f"target_col_{col}"])

        elif method == "Weighted Mean Target Encoding (Bayesian Mean Encoding)":
            st.session_state[f"target_col_{col}"] = st.selectbox(
                f"Target for weighted target encoding `{col}`:",
                columns,
                key=f"target_column_weighted_{col}",
            )
            smoothing = st.slider(
                "Smoothing factor:",
                0.0,
                100.0,
                st.session_state[f"smoothing_factor_{col}"],
                key=f"smoothing_slider_{col}",
            )
            st.session_state[f"smoothing_factor_{col}"] = smoothing
            df = weighted_target_encoding(
                df, col, st.session_state[f"target_col_{col}"], smoothing
            )

        elif method == "K-Fold Target Encoding":
            st.session_state[f"target_col_{col}"] = st.selectbox(
                f"Target for K-Fold encoding `{col}`:",
                columns,
                key=f"target_column_kfold_{col}",
            )
            num_folds = st.slider(
                "Number of folds:",
                2,
                10,
                st.session_state[f"num_folds_{col}"],
                key=f"num_folds_slider_{col}",
            )
            smoothing = st.slider(
                "Smoothing factor:",
                0.0,
                100.0,
                st.session_state[f"smoothing_factor_{col}"],
                key=f"smoothing_slider_kfold_{col}",
            )
            st.session_state[f"num_folds_{col}"] = num_folds
            st.session_state[f"smoothing_factor_{col}"] = smoothing
            df = k_fold_target_encoding(
                df, col, st.session_state[f"target_col_{col}"], num_folds, smoothing
            )

        else:
            st.warning(f"Unsupported method `{method}` for `{col}`.")

        st.success(f"✅ Applied `{method}` to `{col}`.")
        return df
