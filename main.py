from utils.user_interface.sidebar_ui_utils import load_sidebar_inputs
from utils.user_interface.ui_layout import (
    adjust_ui_view,
    explore_and_preprocess_dataset,
)

adjust_ui_view()
df, split_size, seed_number, choosed_task, settings, uploaded_json_file_name = (
    load_sidebar_inputs()
)
df, X_train, X_test, y_train, y_test = explore_and_preprocess_dataset(
    df=df,
    split_size=split_size,
    seed_number=seed_number,
    choosed_task=choosed_task,
    settings=settings,
    uploaded_json_file_name=uploaded_json_file_name,
)
