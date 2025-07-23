import json
import pandas as pd
from PIL import Image
import streamlit as st

from utils.settings_manager import save_configuration_if_updated


def load_sidebar_inputs():
    df = None
    settings = {}
    split_size = None
    seed_number = None
    choosed_task = None
    uploaded_json_file_name = None

    with st.sidebar:
        upload_icon = Image.open("./data/ui_images/file_upload_icon.png")
        col1, col2 = st.columns([1, 7])
        with col1:
            st.image(upload_icon, use_container_width=True)
        with col2:
            st.header("Upload Your CSV Data")

        uploaded_dataset_file = st.file_uploader(
            "Upload your input CSV file here",
            type=["csv"],
            help="Upload a CSV file.",
        )
        if uploaded_dataset_file is not None:
            try:
                df = pd.read_csv(uploaded_dataset_file)
                st.success("✔️ File uploaded successfully.")
            except Exception as error:
                st.error(f"🚨 Error occured reading a .csv file: {repr(error)}")

        st.write(
            "Upload a previously saved JSON file to restore your settings and continue from where you left off."
        )

        uploaded_json_file = st.file_uploader(
            "Upload your settings file (.json format):",
            type=["json"],
            help="Choose a JSON file that contains your saved settings. This will automatically load your previous configurations.",
        )

        if uploaded_json_file is not None:
            uploaded_json_file_name = uploaded_json_file.name.split(".")[0]
            try:
                content = uploaded_json_file.read().decode("utf-8").strip()

                if not content:
                    raise ValueError("JSON file is empty.")

                settings = json.loads(content)

                choosed_task = settings["General settings"][0]
                split_size = settings["General settings"][1]
                seed_number = settings["General settings"][2]

                st.success(
                    f"✔️ Settings file loaded successfully! Uploaded file: {uploaded_json_file.name}"
                )

            except ValueError as e:
                st.warning(f"⚠️ {e}")

            except json.JSONDecodeError as e:
                st.error(f"🚨 Invalid JSON format: {e}")

            except KeyError as e:
                st.error(f"🚨 Missing expected key in JSON: {e}")

            except Exception as e:
                st.error(f"🚨 Unexpected error while loading JSON file: {e}")

    with st.sidebar.header("Select a Task to Perform on the Dataset"):
        choosable_tasks = [
            "Regression",
            "Time Series Forecasting",
            "Classification",
            "Clustering",
        ]
        default_index = (
            choosable_tasks.index(choosed_task)
            if choosed_task in choosable_tasks
            else 0
        )

        choosed_task = st.sidebar.selectbox(
            label="Select Analysis Type",
            options=choosable_tasks,
            index=default_index,
            help=(
                "Choose the type of analysis to perform:\n\n"
                "- **Regression**: Predict continuous values (e.g., house prices).\n"
                "- **Time Series Forecasting**: Predict future values based on time trends.\n"
                "- **Classification**: Assign data to categories (e.g., spam vs. not spam).\n"
                "- **Clustering**: Group similar data points without predefined labels."
            ),
        )
    with st.sidebar.header("Set Parameters"):
        split_size = st.sidebar.slider(
            "Data Split Ratio (%for the training set remaining used for test set)",
            10,
            100,
            split_size if split_size else 80,
            1,
            help="Percentage of data used for training (remaining for testing).",
        )
        seed_number = st.sidebar.number_input(
            label="Random Seed",
            min_value=1,
            max_value=1000,
            value=seed_number if seed_number else 42,
            step=1,
            help="Set a random seed for reproducibility of data splits and model initialization.",
        )
    return df, split_size, seed_number, choosed_task, settings, uploaded_json_file_name
