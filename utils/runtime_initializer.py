# STANDARD MODULES
import streamlit as st

# USER MODULES
from utils.settings_manager import configuration_ui
from utils.settings_manager import save_configuration_if_updated
from utils.preprocessing.missing_value_handler import MissingValueHandler
from utils.preprocessing.categorical_value_handler import CategoricalValueHandler
from utils.exploratory_data_analysis.data_exploration import display_dataset_summary
from utils.preprocessing.multicollinearity_handler import MultiCollinearityHandler
from utils.preprocessing.outlier_detection_and_handling import (
    OutlierDetectionAndHandler,
)


class RuntimeInitializer:
    def __init__(self):
        self.missing_value_handler = MissingValueHandler()
        self.categorical_value_handler = CategoricalValueHandler()
        self.outlier_detection_and_handler = OutlierDetectionAndHandler()
        self.multicollinearity_handler = MultiCollinearityHandler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def run(
        self,
        df,
        split_size,
        seed_number,
        choosed_task,
        settings,
        uploaded_json_file_name,
    ):
        if df is None:
            st.warning(
                "Please upload a '.csv' file to perform exploratory data analysis and preprocessing!"
            )
            st.markdown("""---""")
            return None

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
            return df

        # Step 2: Handle Missing Values
        df, settings = self.missing_value_handler.run(
            df=df, settings=settings, saved_configuration_file=saved_configuration_file
        )
        if st.checkbox(
            "❓ I have handled missing values and am ready to proceed",
            key="missing_values_done",
        ):
            st.success("✅ Proceeding to handle categorical values...")
        else:
            return df

        # Step 3: Handle Categorical Values
        df, settings = self.categorical_value_handler.run(
            df=df, settings=settings, saved_configuration_file=saved_configuration_file
        )
        if st.checkbox(
            "❓ I have encoded categorical values and am ready to proceed",
            key="categorical_done",
        ):
            st.success("✅ Proceeding to remove outliers...")
        else:
            return df

        # Step 4: Remove Outliers
        df, settings = self.outlier_detection_and_handler.run(
            df=df, settings=settings, saved_configuration_file=saved_configuration_file
        )
        if st.checkbox(
            "❓ I have removed outliers and am ready to proceed",
            key="remove_outliers_done",
        ):
            st.success("✅ Procedding to handle multicollinearity...")
        else:
            return df

        # Step 5: Handle Multicollinearity
        df, settings = self.multicollinearity_handler.run(
            df=df, settings=settings, saved_configuration_file=saved_configuration_file
        )

        if st.checkbox(
            "❓ I have handle multicollinearity and am ready to proceed",
            key="handle_multicollinearity_done",
        ):
            st.success(
                "✅ Dataset preprocessing completed! You can proceed to the next stage."
            )
        else:
            return df

        return df
