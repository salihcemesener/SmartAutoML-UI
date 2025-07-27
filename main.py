from utils.runtime_initializer import RuntimeInitializer
from utils.user_interface.ui_initializer import load_sidebar_inputs, adjust_ui_view


def main():
    adjust_ui_view()

    df, split_size, seed, task, settings, config_name = load_sidebar_inputs()

    if df is not None:
        runtime_initializer = RuntimeInitializer()
        df = runtime_initializer.run(
            df=df,
            split_size=split_size,
            seed_number=seed,
            choosed_task=task,
            settings=settings,
            uploaded_json_file_name=config_name,
        )


if __name__ == "__main__":
    main()
