import pandas as pd
from io import BytesIO
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_object_dtype

from utils.settings_manager import save_configuration_if_updated
from utils.preprocessing.data_preprocessing_utils import (
    get_missing_value_options,
    get_categorical_encoding_options,
    list_outlier_detection_strategies,
    get_outlier_removal_options,
    apply_numeric_fill_method,
    apply_categorical_fill_method,
    apply_categorical_to_numerical,
    apply_outlier_detection,
    apply_outlier_removal,
)
from utils.plot_utils.plotting import feature_outlier_analysis


def display_missing_values_info(df):
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        st.markdown("<h3>No Columns with Missing Values</h3>", unsafe_allow_html=True)
        return missing_values

    cols_with_missing = missing_values.index
    dtypes = df.dtypes[cols_with_missing]
    example_values = [
        df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        for col in cols_with_missing
    ]
    missing_info = pd.DataFrame(
        {
            "Number of Rows Missing Values": missing_values,
            "Example Value": example_values,
            "Data Type": dtypes,
        }
    )

    st.markdown("<h3>Columns with Missing Values</h3>", unsafe_allow_html=True)
    st.markdown(
        "Missing values can affect machine learning model performance. "
        "Explore imputation and removal techniques below."
    )
    st.dataframe(missing_info.astype(str), use_container_width=True)
    return missing_values


def ensure_value_handling_config(settings, check_setting_name, values_to_check):
    handling_method_for_list = settings.get(check_setting_name, [])
    existing_columns = {next(iter(element)) for element in handling_method_for_list}
    if isinstance(values_to_check, pd.Series):
        columns_to_check = set(values_to_check.index)
    else:
        columns_to_check = set(values_to_check)

    missing_columns = columns_to_check - existing_columns
    handling_method_for_list.extend({col: []} for col in missing_columns)
    return handling_method_for_list


def handle_missing_values(df, settings, saved_configuration_file):
    original_df_shape = df.shape
    if "existing_method_missing_values" not in st.session_state:
        st.session_state["existing_method_missing_values"] = {}

    has_missing = df.isnull().sum().any()
    if not has_missing:
        st.success("👌 No missing values detected in the uploaded .csv file.")
        st.write(f"**Total number of missing values:** 0")
        st.write(
            f"**Shape of the data before and after handle missing values**: {original_df_shape} to {df.shape}"
        )
        st.write("---")
        return df, settings

    numerical_missing_options, categorical_missing_options = get_missing_value_options()
    missing_values = display_missing_values_info(df=df)

    handling_method_for_missing_value = ensure_value_handling_config(
        settings=settings,
        check_setting_name="Missing_value_handle_methods",
        values_to_check=missing_values,
    )

    existing_methods = st.session_state["existing_method_missing_values"]
    for element in handling_method_for_missing_value:
        col = next(iter(element))
        if col not in existing_methods:
            existing_methods[col] = element[col]

    for col in missing_values.index:
        if f"custom_value_{col}" not in st.session_state:
            st.session_state[f"custom_value_{col}"] = existing_methods.get(
                f"Custom_Value_{col}", None
            )

    with st.expander("Impute or Remove Missing Values", expanded=False):
        st.markdown("<h3>Fill Missing Values per Column</h3>", unsafe_allow_html=True)

        help_numerical_fill = "\n".join(
            [
                f"\n{index+1}- `{key}`: {value}"
                for index, (key, value) in enumerate(numerical_missing_options.items())
            ]
        )

        help_categorical_fill = "\n".join(
            [
                f"\n{index+1}- `{key}`: {value}"
                for index, (key, value) in enumerate(
                    categorical_missing_options.items()
                )
            ]
        )
        with st.expander("🔍 Show Available Handle Missing Value Options Explanations"):
            st.write(
                f"👽 Available filling techniques for **numerical columns**:\n{help_numerical_fill}"
            )
            st.write(
                f"👽 Available filling techniques for **categorical columns**:\n{help_categorical_fill}"
            )
        for col in missing_values.index:
            with st.container():
                is_numeric = is_numeric_dtype(df[col])
                options = (
                    numerical_missing_options
                    if is_numeric
                    else categorical_missing_options
                )
                default_method = existing_methods.get(col, next(iter(options.keys())))
                default_method = (
                    default_method
                    if default_method
                    else [next(iter(options.keys())), None]
                )
                st.session_state[f"custom_value_{col}"] = default_method[1]
                method = st.selectbox(
                    f"How to handle missing values in {'numeric' if is_numeric else 'categorical'} column **{col}**. "
                    f"Number of missing values: `{df[col].isnull().sum()}`",
                    options.keys(),
                    help=f"Select a method for {'numeric' if is_numeric else 'categorical'} column {col}.",
                    index=list(options.keys()).index(default_method[0]),
                )
                try:
                    df = (
                        apply_numeric_fill_method(
                            fill_method_name=method, df=df, col=col
                        )
                        if is_numeric
                        else apply_categorical_fill_method(
                            fill_method_name=method, df=df, col=col
                        )
                    )

                    for element in handling_method_for_missing_value:
                        if col in element:
                            element[col] = [
                                method,
                                st.session_state[f"custom_value_{col}"],
                            ]
                            break

                except KeyError:
                    st.info(f"🚨 Column {col} removed from the data frame.")
                except ValueError:
                    st.error(f"🚨 Enter a valid number for {col}")
                except Exception as error:
                    st.error(
                        f"🚨 Error occurred when filling missing value with {method}. Error: {repr(error)}"
                    )

    settings = save_configuration_if_updated(
        config_file_name=saved_configuration_file,
        new_config_data=handling_method_for_missing_value,
        config_data_key="Missing_value_handle_methods",
    )

    st.write(f"**Total number of missing values:** {df.isnull().sum().sum()}")
    st.write(
        f"**Shape of the data before and after handle missing values**: {original_df_shape} to {df.shape}"
    )
    st.write("---")
    return df, settings


def display_categorica_values_info(df):

    st.markdown(f"<h3>Handling Categorical Data</h3>", unsafe_allow_html=True)
    st.markdown(
        "Categorical columns often need to be converted to numerical format to be used in machine learning models. In this section, we will explore various encoding techniques to handle categorical values efficiently."
    )

    categorical_cols = [col for col in df.columns if is_object_dtype(df[col])]

    visualized_categorical = pd.DataFrame(
        {
            "Column Name": categorical_cols,
            "Number of Unique Values": df[categorical_cols].nunique(),
        }
    ).astype(str)
    st.dataframe(visualized_categorical)

    return categorical_cols


def handle_categorical_values(df, settings, saved_configuration_file):

    if "existing_method_categorical_values" not in st.session_state:
        st.session_state["existing_method_categorical_values"] = {}

    has_categorical_cols = [col for col in df.columns if is_object_dtype(df[col])]

    if not has_categorical_cols:
        st.success("👌 No categorical values detected in the uploaded .csv file.")
        st.write(f"**Total number of categorical columns:** 0")
        st.write("---")
        return df, settings
    encoding_options = get_categorical_encoding_options()
    categorical_cols = display_categorica_values_info(df)

    handling_method_for_categorical_value = ensure_value_handling_config(
        settings=settings,
        check_setting_name="Categorical_value_handle_methods",
        values_to_check=has_categorical_cols,
    )
    existing_methods = st.session_state["existing_method_categorical_values"]
    for element in handling_method_for_categorical_value:
        col = next(iter(element))
        if col not in existing_methods:
            existing_methods[col] = element[col]

    for col in categorical_cols:
        if f"num_folds_{col}" not in st.session_state:
            st.session_state[f"num_folds_{col}"] = existing_methods.get(
                f"num_folds_{col}", 5
            )
        if f"target_col_{col}" not in st.session_state:
            st.session_state[f"target_col_{col}"] = existing_methods.get(
                f"target_col_{col}", df.columns[0]
            )
        if f"smoothing_factor_{col}" not in st.session_state:
            st.session_state[f"smoothing_factor_{col}"] = existing_methods.get(
                f"smoothing_factor_{col}", 10.0
            )
        if f"variable_order_{col}" not in st.session_state:
            st.session_state[f"variable_order_{col}"] = existing_methods.get(
                f"variable_order_{col}", None
            )

    with st.expander("Convert Categorical to Numeric Data", expanded=False):
        st.markdown(
            "<h3>Convert Categorical Columns to Numeric Values</h3>",
            unsafe_allow_html=True,
        )
        help_categorical_conv = "\n".join(
            [
                f"\n{index+1}- `{key}`: {value}"
                for index, (key, value) in enumerate(encoding_options.items())
            ]
        )
        with st.expander("🔍 Show Available Categorical Encoding Options Explanations"):
            st.write(
                f"👽 Available encoding techniques for **categorical columns**:\n{help_categorical_conv}"
            )

        for col in categorical_cols:
            with st.container():
                st.info(
                    f"🔎 Processing **{col}** column has {df[col].nunique()} unique values detected."
                )
                default_method = existing_methods.get(
                    col, next(iter(encoding_options.keys()))
                )
                default_method = (
                    default_method
                    if default_method
                    else [
                        next(iter(encoding_options.keys())),
                        5,
                        df.columns[0],
                        10.0,
                        None,
                    ]
                )
                st.session_state[f"num_folds_{col}"] = default_method[1]
                st.session_state[f"target_col_{col}"] = default_method[2]
                st.session_state[f"smoothing_factor_{col}"] = default_method[3]
                st.session_state[f"variable_order_{col}"] = default_method[4]

                convert_to_numeric_method = st.selectbox(
                    f"How to convert values in '{col}' to numeric?",
                    options=encoding_options.keys(),
                    help=f"Select a method for categorical column {col} encoding.",
                    index=list(encoding_options.keys()).index(default_method[0]),
                )
                try:
                    df = apply_categorical_to_numerical(
                        fill_method_name=convert_to_numeric_method,
                        df=df,
                        col=col,
                    )
                    for element in handling_method_for_categorical_value:
                        if col in element:
                            element[col] = [
                                convert_to_numeric_method,
                                st.session_state[f"num_folds_{col}"],
                                st.session_state[f"target_col_{col}"],
                                st.session_state[f"smoothing_factor_{col}"],
                                st.session_state[f"variable_order_{col}"],
                            ]

                except Exception as error:
                    st.error(
                        f"🚨 Error occurred when convert categorical value to numerical value with {convert_to_numeric_method}. Error: {repr(error)}"
                    )

                settings = save_configuration_if_updated(
                    config_file_name=saved_configuration_file,
                    new_config_data=handling_method_for_categorical_value,
                    config_data_key="Categorical_value_handle_methods",
                )

        st.dataframe(df.astype(str))
        st.markdown("""---""")
    return df, settings


def display_remove_outliers_info(df):
    st.header("Remove Outliers")
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


def display_feature_analysis_section(df, col, target_col):
    def render_figure(fig, caption, container):
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        container.image(buf, caption=caption, width=480)

    with st.container():
        if st.checkbox(
            f"❓ Visualize '{col}' by target", key=f"visualize_{col}_by_target"
        ):
            figs = feature_outlier_analysis(df=df, col=col, target=target_col)

            num_figs = len(figs)
            for i in range(0, num_figs, 2):
                cols = st.columns(2)
                for j in range(2):
                    idx = i + j
                    if idx < num_figs:
                        render_figure(
                            figs[idx], caption=f"Figure {idx + 1}", container=cols[j]
                        )


def set_target_column(label, columns):
    st.session_state["remove_outliers_target_col"] = st.selectbox(
        label=label,
        options=columns,
        index=list(columns).index(st.session_state["remove_outliers_target_col"]),
        key=f"remove_outliers_set_target_col",
    )


def init_outlier_params_for_col(df, col, existing_methods):
    if "remove_outliers_target_col" not in st.session_state:
        st.session_state["remove_outliers_target_col"] = existing_methods.get(
            "remove_outliers_target_col", df.columns[0]
        )
    defaults = {
        "selected_target_col": col,
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
        f"Mahalanobis_th_{col}": 20.0,
        f"Handling_method_to_remove_outliers_{col}": [
            "remove",
            "impute_median",
            "impute_mean",
            "mark",
            "skip",
        ],
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = existing_methods.get(key, default)


def handle_remove_outliers(df, settings, saved_configuration_file):
    original_dimension = df.shape
    if "existing_method_remove_outliers" not in st.session_state:
        st.session_state["existing_method_remove_outliers"] = {}

    display_remove_outliers_info(df=df)

    handling_method_for_remove_outliers = ensure_value_handling_config(
        settings=settings,
        check_setting_name="Remove_outliers_handle_methods",
        values_to_check=df.columns,
    )
    existing_methods = st.session_state["existing_method_remove_outliers"]
    for element in handling_method_for_remove_outliers:
        col = next(iter(element))
        if col not in existing_methods:
            existing_methods[col] = element[col]

    for col in df.columns:
        init_outlier_params_for_col(df, col, existing_methods)

    outlier_detection_options = list_outlier_detection_strategies()
    outlier_removal_options = get_outlier_removal_options()

    st.session_state["selected_target_col"] = st.selectbox(
        "Visualize outliers by using target as:",
        options=df.columns.tolist(),
        index=df.columns.get_loc(
            st.session_state.get("remove_outliers_target_col", df.columns[0])
        ),
        key="remove_outliers_target_col",
        help="This column will be used as the grouping variable in the box‐plots and as the y-axis in the joint‐plot.",
    )

    with st.expander("Outlier Detection & Handling", expanded=False):
        st.markdown("<h3>Remove Outliers from the Dataset</h3>", unsafe_allow_html=True)
        help_outlier_detection = "\n".join(
            [
                f"\n{index+1}- `{key}`: {value}"
                for index, (key, value) in enumerate(outlier_detection_options.items())
            ]
        )

        help_outlier_removal = "\n".join(
            [
                f"\n{index+1}- `{key}`: {value}"
                for index, (key, value) in enumerate(outlier_removal_options.items())
            ]
        )
        with st.expander(
            "🔍 Show Available Outlier Detection Techniques Options Explanations"
        ):
            st.markdown(
                f"👽 Available outlier detection techniques**:\n{help_outlier_detection}"
            )

        with st.expander(
            "🔍 Show Available Outlier Removal Techniques Options Explanations"
        ):
            st.markdown(
                f"👽 Available outlier removal techniques**:\n{help_outlier_removal}"
            )

        with st.expander("🔍 Interpretation of the Box Plots Explanation"):
            st.markdown("### 📈 How to interpret these box-plots")
            st.image(
                "data/ui_images/boxplot_explanation.png",
                width=512,
            )
            st.markdown(
                """
                - **Median line** (inside the box): the 50th percentile of the data.  
                - **Box edges**: represent the 1st (Q1) and 3rd (Q3) quartiles — the middle 50% of values.  
                - **Whiskers**: typically extend to the most extreme data points within 1.5 × IQR.  
                - **Dots beyond whiskers**: potential outliers.  
                - **Box height**: indicates variability.  
                - **Comparing boxes**: helps in identifying separation power by target classes.
                """
            )
        for col in df.columns:
            display_feature_analysis_section(
                df=df, col=col, target_col=st.session_state["selected_target_col"]
            )

            default_method = existing_methods.get(
                col, next(iter(outlier_detection_options.keys()))
            )

            default_method = (
                default_method
                if default_method
                else [
                    next(iter(outlier_detection_options.keys())),
                    3.0,
                    1.5,
                    1.4826,
                    3.0,
                    95.0,
                    5.0,
                    100.0,
                    "auto",
                    0.05,
                    5.0,
                    20.0,
                    0.05,
                    "euclidean",
                    20.0,
                    next(iter(outlier_removal_options.keys())),
                ]
            )
            st.session_state[f"z_score_th_{col}"] = default_method[1]
            st.session_state[f"IQR_multiplier_{col}"] = default_method[2]
            st.session_state[f"MAD_scale_factor_{col}"] = default_method[3]
            st.session_state[f"MAD_th_{col}"] = default_method[4]
            st.session_state[f"Winsorization_upper_percentile_{col}"] = default_method[
                5
            ]
            st.session_state[f"Winsorization_lower_percentile_{col}"] = default_method[
                6
            ]
            st.session_state[f"Isolation_Forest_n_estimators_{col}"] = default_method[7]
            st.session_state[f"Isolation_Forest_max_samples_{col}"] = default_method[8]
            st.session_state[f"Isolation_Forest_contamination_{col}"] = default_method[
                9
            ]
            st.session_state[f"Isolation_Forest_max_features_{col}"] = default_method[
                10
            ]
            st.session_state[f"LOF_n_neighbors_{col}"] = default_method[11]
            st.session_state[f"LOF_contamination_{col}"] = default_method[12]
            st.session_state[f"LOF_metric_{col}"] = default_method[13]
            st.session_state[f"Mahalanobis_th_{col}"] = default_method[14]
            st.session_state[f"Handling_method_to_remove_outliers_{col}"] = (
                default_method[15]
            )

            outlier_detection_method = st.selectbox(
                f"How to outlier detect in {col}?",
                options=outlier_detection_options.keys(),
                help=f"Select a method for outlier detect in {col}.",
                index=list(outlier_detection_options.keys()).index(default_method[0]),
                key=f"outlier_detection_selectbox_{col}",
            )
            if outlier_detection_method == "Local Outlier Factor (LOF)":
                st.session_state[f"LOF_metric_{col}"] = st.selectbox(
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
            outlier_removal_method = st.selectbox(
                f"How would you like to handle detected outliers in `{col}`?",
                options=outlier_removal_options.keys(),
                help=f"Select a method for outlier removal in {col}.",
                index=list(outlier_removal_options.keys()).index(default_method[15]),
                key=f"outlier_removal_selectbox_{col}",
            )
            try:
                outlier_detection_output, detected_outlier_indices = (
                    apply_outlier_detection(
                        fill_method_name=outlier_detection_method, df=df, col=col
                    )
                )

                df, outlier_removal_output = apply_outlier_removal(
                    df=df,
                    col=col,
                    outlier_indices=detected_outlier_indices,
                    method=outlier_removal_method,
                )
                for element in handling_method_for_remove_outliers:
                    if col in element:
                        element[col] = [
                            outlier_detection_method,
                            st.session_state.get(f"z_score_th_{col}"),
                            st.session_state.get(f"IQR_multiplier_{col}"),
                            st.session_state.get(f"MAD_scale_factor_{col}"),
                            st.session_state.get(f"MAD_th_{col}"),
                            st.session_state.get(
                                f"Winsorization_upper_percentile_{col}"
                            ),
                            st.session_state.get(
                                f"Winsorization_lower_percentile_{col}"
                            ),
                            st.session_state.get(
                                f"Isolation_Forest_n_estimators_{col}"
                            ),
                            st.session_state.get(f"Isolation_Forest_max_samples_{col}"),
                            st.session_state.get(
                                f"Isolation_Forest_contamination_{col}"
                            ),
                            st.session_state.get(
                                f"Isolation_Forest_max_features_{col}"
                            ),
                            st.session_state.get(f"LOF_n_neighbors_{col}"),
                            st.session_state.get(f"LOF_contamination_{col}"),
                            st.session_state.get(f"LOF_metric_{col}"),
                            st.session_state.get(f"Mahalanobis_th_{col}"),
                            st.session_state.get(
                                f"Handling_method_to_remove_outliers_{col}"
                            ),
                        ]

                st.info(
                    f"🧪 **Outlier Detection Result:**\n\n{outlier_detection_output}\n\n**Outlier Removal Result:**\n\n{outlier_removal_output}"
                )
            except Exception as error:
                st.error(
                    f"🚨 Error occurred when remove outliers with {outlier_detection_method} at {col}. Error: {repr(error)}"
                )

            settings = save_configuration_if_updated(
                config_file_name=saved_configuration_file,
                new_config_data=handling_method_for_remove_outliers,
                config_data_key="Remove_outliers_handle_methods",
            )
            st.warning(
                f"📏 Dataset size changed from **{original_dimension}** to **{df.shape}** after applying outlier removal."
            )

    return df, settings
