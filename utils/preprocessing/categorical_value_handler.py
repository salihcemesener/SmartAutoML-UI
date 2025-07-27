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
from utils.preprocessing.categorical_encoding_methods import (
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

        # Upload saved settings
        for element in config_list:
            col = next(iter(element))
            if col not in st.session_state["existing_method_categorical_values"]:
                st.session_state["existing_method_categorical_values"][col] = element[
                    col
                ]

        with st.expander("Convert Categorical to Numeric Data", expanded=False):
            st.markdown("<h3>Convert Categorical Columns</h3>", unsafe_allow_html=True)
            st.markdown(
                """Select a method to convert each categorical column into numeric format."""
            )

            for col in categorical_cols:
                st.info(f"Column **{col}** has {df[col].nunique()} unique values.")

                fallback = [
                    next(iter(self.encoding_options.keys())),  # encoding method
                    5,                                         # top-N categories
                    df.columns[0],                             # target column for impact-based methods
                    10.0,                                      # frequency threshold, or smoothing param
                    None,                                      # custom encoder or fallback
                ]
                if not st.session_state["existing_method_categorical_values"].get(col):
                    st.session_state["existing_method_categorical_values"][col] = fallback
                default = st.session_state["existing_method_categorical_values"][col]

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
        try:
            columns = df.columns
            st.session_state[f"num_folds_{col}"] = st.session_state.get(
                f"num_folds_{col}", 5
            )
            st.session_state[f"target_col_{col}"] = st.session_state.get(
                f"target_col_{col}", columns[0]
            )
            st.session_state[f"smoothing_factor_{col}"] = st.session_state.get(
                f"smoothing_factor_{col}", 10.0
            )

            unique_vals = df[col].dropna().unique().tolist()
            default_order = st.session_state.get(f"variable_order_{col}")

            if len(unique_vals) > 10:
                if method == "Ordinal Encoding":
                    confirm = st.checkbox(
                        "🚨 That column has many unique values. Are you sure you want to apply ordinal encoding? This setting may significantly increase the size of your JSON configuration file.",
                        key=f"ordinal_encoding_variable_order_{col}",
                    )
                    if confirm:
                        st.session_state[f"variable_order_{col}"] = (
                            default_order or unique_vals
                        )
            else:
                st.session_state[f"variable_order_{col}"] = default_order or unique_vals

            def set_target_column(label):
                st.session_state[f"target_col_{col}"] = st.selectbox(
                    label=label,
                    options=columns,
                    index=list(columns).index(st.session_state[f"target_col_{col}"]),
                    key=f"set_target_col_{col}",
                )

            def set_smoothing_factor():
                st.session_state[f"smoothing_factor_{col}"] = st.slider(
                    "Set smoothing factor:",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state[f"smoothing_factor_{col}"],
                    step=1.0,
                    key=f"set_smoothing_factor_{col}",
                )

            def set_num_folds():
                st.session_state[f"num_folds_{col}"] = st.slider(
                    "Select number of folds (K):",
                    min_value=2,
                    max_value=10,
                    value=st.session_state[f"num_folds_{col}"],
                    step=1,
                    key=f"set_num_folds_{col}",
                )

            if method == "Label Encoding":
                df[col] = LabelEncoder().fit_transform(df[col])
                st.success(f"**{col}** converted using Label Encoding.")

            elif method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=[col], drop_first=True)
                st.success(f"**{col}** converted using One-Hot Encoding.")

            elif method == "Frequency Encoding":
                df[col] = df[col].map(df[col].value_counts())
                st.success(f"**{col}** converted using Frequency Encoding.")

            elif method == "Ordinal Encoding":
                st.session_state[f"variable_order_{col}"] = st.multiselect(
                    f"Define the custom order for **{col}**:",
                    df[col].dropna().unique().tolist(),
                    default=st.session_state[f"variable_order_{col}"],
                    help="Manually define the meaningful order (e.g., Low < Medium < High).",
                )
                mapping = {
                    val: idx
                    for idx, val in enumerate(st.session_state[f"variable_order_{col}"])
                }
                df[col] = df[col].map(mapping)
                st.success(f"**{col}** converted using Ordinal Encoding.")

            elif method == "Binary Encoding":
                df = ce.BinaryEncoder(cols=[col]).fit_transform(df)
                st.success(f"**{col}** converted using Binary Encoding.")

            elif method == "Target Encoding":
                set_target_column(
                    f"Select Target Column for Target Encoding for **{col}**:"
                )
                df = target_encoding(df, col, st.session_state[f"target_col_{col}"])
                st.success(
                    f"**{col}** converted using Target Encoding based on {st.session_state[f'target_col_{col}']}."
                )

            elif (
                method == "Weighted Mean Target Encoding (Bayesian Mean Encoding)"
            ):
                set_target_column(
                    f"Select Target Column for Weighted Mean Target Encoding for **{col}**:"
                )
                set_smoothing_factor()
                df = weighted_target_encoding(
                    df,
                    col,
                    st.session_state[f"target_col_{col}"],
                    st.session_state[f"smoothing_factor_{col}"],
                )
                st.success(
                    f"**{col}** converted using Weighted Target Encoding with smoothing {st.session_state[f'smoothing_factor_{col}']} based on {st.session_state[f'target_col_{col}']}."
                )

            elif method == "K-Fold Target Encoding":
                set_target_column(
                    f"Select Target Column for K-Fold Target Encoding for **{col}**:"
                )
                set_num_folds()
                set_smoothing_factor()
                df = k_fold_target_encoding(
                    df,
                    col,
                    st.session_state[f"target_col_{col}"],
                    num_folds=st.session_state[f"num_folds_{col}"],
                    smoothing=st.session_state[f"smoothing_factor_{col}"],
                )
                st.success(
                    f"**{col}** converted using K-Fold Target Encoding with {st.session_state[f'num_folds_{col}']} folds and smoothing {st.session_state[f'smoothing_factor_{col}']} based on {st.session_state[f'target_col_{col}']}."
                )

            else:
                st.warning(f"No valid encoding method selected for {col}.")

            st.markdown("""---""")
            return df

        except Exception as e:
            st.error(f"🚨 Error occurred while encoding **{col}**: {repr(e)}")
            return df

