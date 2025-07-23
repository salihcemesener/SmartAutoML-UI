from PIL import Image
import streamlit as st

from utils.settings_manager import configuration_ui
from utils.settings_manager import save_configuration_if_updated
from utils.preprocessing.feature_selection_and_data_split import (
    select_features_and_split_data,
)
from utils.preprocessing.data_preprocessing import (
    handle_missing_values,
    handle_categorical_values,
    handle_remove_outliers,
)
from utils.exploratory_data_analysis.data_exploration import (
    display_dataset_summary,
    exploratory_data_analysis,
    plot_correlation_map,
)


def adjust_ui_view():
    st.set_page_config(
        page_title="AutoML App",
        layout="wide",
        page_icon=Image.open("./data/ui_images/header_icon.png"),
    )

    image = Image.open("./data/ui_images/ui_header_image.png")
    image = image.resize((1920, 1080))

    st.markdown(
        "<h1 style='text-align: center;'>Welcome to the AutoML App</h1>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            image,
            use_container_width=True,
        )


def explore_and_preprocess_dataset(
    df, split_size, seed_number, choosed_task, settings, uploaded_json_file_name
):
    X_train, X_test, y_train, y_test = None, None, None, None

    if df is None:
        st.warning(
            "Please upload a '.csv' file to perform exploratory data analysis and preprocessing!"
        )
        st.markdown("""---""")
        return None, X_train, X_test, y_train, y_test

    saved_configuration_file = configuration_ui(
        settings=settings, uploaded_json_file_name=uploaded_json_file_name
    )
    settings = save_configuration_if_updated(
        config_file_name=saved_configuration_file,
        new_config_data=[
            choosed_task,
            split_size,
            seed_number,
        ],
        config_data_key="General settings",
    )
    # Step 1: Dataset Summary
    df, settings = display_dataset_summary(
        df=df, settings=settings, saved_configuration_file=saved_configuration_file
    )
    if st.checkbox(
        "❓ I have reviewed the dataset summary and am ready to proceed",
        key="summary_done",
    ):
        st.success("✅ Proceeding to handle missing values...")
    else:
        return df, X_train, X_test, y_train, y_test

    # Step 2: Handle Missing Values
    df, settings = handle_missing_values(
        df=df, settings=settings, saved_configuration_file=saved_configuration_file
    )
    if st.checkbox(
        "❓ I have handled missing values and am ready to proceed",
        key="missing_values_done",
    ):
        st.success("✅ Proceeding to handle categorical values...")
    else:
        return df, X_train, X_test, y_train, y_test

    # Step 3: Handle Categorical Values
    df, settings = handle_categorical_values(
        df=df, settings=settings, saved_configuration_file=saved_configuration_file
    )
    if st.checkbox(
        "❓ I have encoded categorical values and am ready to proceed",
        key="categorical_done",
    ):
        st.success("✅ Proceeding to remove outliers...")
    else:
        return df, X_train, X_test, y_train, y_test
    # Step 4: Remove Outliers
    df, settings = handle_remove_outliers(
        df=df, settings=settings, saved_configuration_file=saved_configuration_file
    )
    if st.checkbox(
        "❓ I have removed outliers and am ready to proceed", key="remove_outliers_done"
    ):
        st.success(
            "✅ Dataset preprocessing completed! You can proceed to the next stage."
        )
    else:
        return df, X_train, X_test, y_train, y_test

    # Optional future steps
    # df = exploratory_data_analysis(df)
    # df, settings = plot_correlation_map(df=df, settings=settings, saved_configuration_file=saved_configuration_file)
    # X_train, X_test, y_train, y_test = select_features_and_split_data(df=df, split_size=split_size, seed_number=seed_number)

    return df, X_train, X_test, y_train, y_test
