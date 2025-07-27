# STANDARD MODULES
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

# USER MODULES
from utils.settings_manager import save_configuration_if_updated
from utils.preprocessing.data_preprocessor_abc import DataPreprocessorHandler
from utils.preprocessing.data_preprocessing_helper_text import (
    DataPreprocessingOptionsHelperText,
)


class MissingValueHandler(DataPreprocessorHandler):
    def __init__(self):
        self.numerical_missing_options, self.categorical_missing_options = (
            DataPreprocessingOptionsHelperText.get_missing_value_options()
        )
        st.session_state.setdefault("existing_method_missing_values", {})

    def init_parameters_for_col(self, df, missing_values):

        for col in missing_values:
            key = f"custom_value_{col}"
            st.session_state.setdefault(
                key,
                st.session_state["existing_method_missing_values"].get(
                    f"Custom_Value_{col}", None
                ),
            )

    def display_info(self, df):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.markdown(
                "<h3>No Columns with Missing Values</h3>", unsafe_allow_html=True
            )
            return missing

        dtypes = df.dtypes[missing.index]
        example_values = [
            df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            for col in missing.index
        ]
        st.markdown("<h3>Columns with Missing Values</h3>", unsafe_allow_html=True)
        st.markdown("Explore imputation and removal techniques below.")
        st.dataframe(
            pd.DataFrame(
                {
                    "Missing Count": missing,
                    "Example Value": example_values,
                    "Data Type": dtypes,
                }
            ).astype(str),
            use_container_width=True,
        )
        return missing

    def print_no_missing_summary(self, original_shape, df):
        st.success("👌 No missing values detected in the uploaded dataset.")
        st.write(f"**Missing Count:** 0")
        st.write(f"**Shape (Before → After):** {original_shape} → {df.shape}")
        st.write("---")

    def run(self, df, settings, saved_configuration_file):
        original_shape = df.shape
        if not df.isnull().sum().any():
            self.print_no_missing_summary(original_shape, df)
            return df, settings
        missing_values = self.display_info(df)

        config_list = self.sync_column_config_list(
            settings, "Missing_value_handle_methods", missing_values.index
        )

        self.init_parameters_for_col(df, missing_values=missing_values.index)

        # Upload saved settings
        for element in config_list:
            col = next(iter(element))
            if col not in st.session_state["existing_method_missing_values"]:
                st.session_state["existing_method_missing_values"][col] = element[col]

        with st.expander("Impute or Remove Missing Values", expanded=False):
            self._display_missing_value_options_help()

            for col in missing_values.index:
                is_numeric = is_numeric_dtype(df[col])
                options = (
                    self.numerical_missing_options
                    if is_numeric
                    else self.categorical_missing_options
                )

                if not st.session_state["existing_method_missing_values"].get(col):
                    st.session_state["existing_method_missing_values"][col] = [
                        next(iter(options.keys())),
                        None,
                    ]

                default_method = st.session_state["existing_method_missing_values"][col]

                st.session_state[f"custom_value_{col}"] = default_method[1]

                method = st.selectbox(
                    f"Handle missing values in {'numeric' if is_numeric else 'categorical'} column `{col}` (missing: {df[col].isnull().sum()})",
                    options=list(options.keys()),
                    index=list(options).index(default_method[0]),
                )
                try:
                    df = self.apply_method(df, col, method)
                    for entry in config_list:
                        if col in entry:
                            entry[col] = [
                                method,
                                st.session_state[f"custom_value_{col}"],
                            ]
                            break
                except Exception as e:
                    st.error(f"🚨 Error handling `{col}` with `{method}`: {repr(e)}")
        settings = save_configuration_if_updated(
            config_file_name=saved_configuration_file,
            new_config_data=config_list,
            config_data_key="Missing_value_handle_methods",
        )

        st.write(f"**Remaining Missing Count:** {df.isnull().sum().sum()}")
        st.write(f"**Shape (Before → After):** {original_shape} → {df.shape}")
        st.write("---")
        return df, settings

    def apply_method(self, df, col, method_name):
        return (
            self._apply_numeric(df, col, method_name)
            if is_numeric_dtype(df[col])
            else self._apply_categorical(df, col, method_name)
        )

    def _apply_numeric(self, df, col, method):
        try:
            if method == "Fill with Mean":
                df[col] = df[col].fillna(df[col].mean())
                st.success(
                    f"{col} column missing values are filled using mean. Mean value: {df[col].mean()}"
                )
            elif method == "Fill with Median":
                df[col] = df[col].fillna(df[col].median())
                st.success(
                    f"{col} column missing values are filled using median. Median of the columns: {df[col].median()}"
                )
            elif method == "Fill with Custom Value":
                st.session_state[f"custom_value_{col}"] = st.number_input(
                    f"Custom value for {col}",
                    value=st.session_state[f"custom_value_{col}"] or 0,
                )
                df[col] = df[col].fillna(st.session_state[f"custom_value_{col}"])
                st.success(
                    f'{col} column missing values filled with custom value: {st.session_state[f"custom_value_{col}"]}.'
                )
            elif method == "Remove Rows":
                df.dropna(subset=[col], inplace=True)
                st.success(f"Rows with missing values in {col} column removed.")

            elif method == "Remove Columns":
                df.drop(columns=[col], inplace=True)
                st.success(f"{col} column removed from the DataFrame.")

        except Exception as e:
            st.error(f"Error filling numeric `{col}` using `{method}`: {repr(e)}")
        return df

    def _apply_categorical(self, df, col, method):
        try:
            if method == "Fill with Mode":
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                st.success(f"{col} column missing values filled with mode: {mode_val}.")

            elif method == "Fill with Custom Value":
                df = self._custom_categorical_fill(df, col)
                st.success(f"{col} column missing values filled with custom value.")

            elif method == "Remove Rows":
                df.dropna(subset=[col], inplace=True)
                st.success(f"Rows with missing values in {col} column removed.")

            elif method == "Remove Columns":
                df.drop(columns=[col], inplace=True)
                st.success(f"{col} column removed from the DataFrame.")

        except Exception as e:
            st.error(f"Error filling categorical `{col}` using `{method}`: {repr(e)}")
        return df

    def _custom_categorical_fill(self, df, col):
        try:
            values = df[col].dropna().value_counts().index.tolist()
            default = st.session_state.get(f"custom_value_{col}", values[0])
            index = values.index(default) if default in values else 0
            st.session_state[f"custom_value_{col}"] = st.selectbox(
                f"Select replacement value for `{col}`:", values, index=index
            )
            df[col] = df[col].fillna(st.session_state[f"custom_value_{col}"])
        except Exception as e:
            st.error(f"Error in custom fill for `{col}`: {repr(e)}")
        return df

    def _display_missing_value_options_help(self):
        with st.expander("🔍 Help: Missing Value Handling Options"):
            num = "\n".join(
                [
                    f"{i+1}. `{k}`: {v}"
                    for i, (k, v) in enumerate(self.numerical_missing_options.items())
                ]
            )
            cat = "\n".join(
                [
                    f"{i+1}. `{k}`: {v}"
                    for i, (k, v) in enumerate(self.categorical_missing_options.items())
                ]
            )
            st.markdown(f"**Numeric:**\n{num}")
            st.markdown(f"**Categorical:**\n{cat}")
