# data_preparation.py

import streamlit as st
from sklearn.model_selection import train_test_split


def select_features_and_split_data(df, split_size, seed_number):

    st.markdown(
        f"<h3>Select Columns for Model Input and Target</h3>", unsafe_allow_html=True
    )
    st.markdown(
        "Selecting the input columns and the target column is crucial for building effective machine learning models. In this section, we will choose the columns that will be used as input features for the model, as well as the column that will serve as the target variable for predictions."
    )

    target_column = None
    input_feature_columns = []

    with st.expander("Select Input and Target Features"):
        if not df.empty:
            input_feature_columns = st.multiselect(
                "Select the columns for input features (one or more columns):",
                options=df.columns.to_list(),
                default=df.columns.tolist(),
                help="Select the columns to be used as input features for model training and testing.",
            )

            if not input_feature_columns:
                st.warning(
                    "No input features selected. Please choose one or more columns."
                )
            available_target_columns = [
                col for col in df.columns if col not in input_feature_columns
            ]

            if available_target_columns:
                target_column = st.selectbox(
                    "Select the column for the target feature:",
                    options=available_target_columns,
                    help="Choose the target column for model training and testing.",
                )
            else:
                st.warning(
                    "No columns left to choose from for the target feature. Please select input features first."
                )
        else:
            st.warning(
                "The DataFrame is empty or not available. Please load the data first."
            )

    if input_feature_columns and target_column:
        st.success(f"Input features selected: {', '.join(input_feature_columns)}")
        st.success(f"Target feature selected: {target_column}")
        st.info(f"Shape of input features: {df[input_feature_columns].shape}")
        st.info(f"Shape of target feature: {df[[target_column]].shape}")
    elif not input_feature_columns:
        st.error("Please select input features.")
    elif not target_column:
        st.error("Please select a target feature.")

    st.markdown(f"<h4>Training and Test Data</h4>", unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = perform_train_test_split(
        df, split_size, input_feature_columns, target_column, seed_number
    )

    if X_train is not None:
        st.info(
            f"Shape of training data to predict target {X_train.shape} and shape of the target {y_train.shape}"
        )
        st.info(
            f"Shape of test data to predict target {X_test.shape} and shape of the target {y_test.shape}"
        )

    st.markdown("---")
    return X_train, X_test, y_train, y_test


def perform_train_test_split(
    df, split_size, input_feature_columns, target_column, seed_number
):

    missing_params = {}

    if split_size is None:
        missing_params["split_size"] = split_size
    if not input_feature_columns:
        missing_params["input_feature_columns"] = input_feature_columns
    if not target_column:
        missing_params["target_column"] = target_column

    if missing_params:
        missing_params_str = ", ".join(
            [f"{key} ---> {value}" for key, value in missing_params.items()]
        )
        st.error(f"Missing parameters: {missing_params_str}")
        return None, None, None, None
    try:
        X = df[input_feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_size, random_state=seed_number
        )
    except Exception as error:
        st.error(f"Unexpected error in perform_train_test_split. Error: {str(error)}")
        return None, None, None, None
    return X_train, X_test, y_train, y_test
