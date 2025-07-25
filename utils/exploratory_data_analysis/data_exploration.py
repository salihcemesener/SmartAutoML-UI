import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from utils.settings_manager import save_configuration_if_updated


def duplicate_rows(df):
    # Identify duplicated rows (excluding the first occurrence)
    duplicates = df[df.duplicated()]
    remove_duplicate = [False]

    with st.expander("🧩 Duplicate Rows Check", expanded=False):
        if not duplicates.empty:
            st.warning("⚠️ The following rows are duplicated:")
            st.dataframe(duplicates)

            if st.checkbox("🗑️ Remove duplicated rows", key="remove_duplicates"):
                df = df.drop_duplicates()
                remove_duplicate = [True]
                st.success("✅ Duplicates removed. Showing updated DataFrame:")
                st.dataframe(df)
        else:
            st.success("✅ No duplicated rows found.")

    return df, remove_duplicate


def duplicate_rows_drop_columns_in_dataset(df, dropped_columns=None, key="drop_cols"):
    dropped_columns = dropped_columns or []
    df, remove_duplicate = duplicate_rows(df=df)
    seed_flag = f"{key}_seeded"
    if not st.session_state.get(seed_flag, False):
        st.session_state[key] = dropped_columns
        st.session_state[seed_flag] = True

    to_drop = st.multiselect(
        "Select columns to drop", options=df.columns.tolist(), key=key
    )

    if to_drop:
        st.info(f"👇 You selected the following columns to drop: {', '.join(to_drop)}.")
        df = df.drop(columns=to_drop)
    else:
        st.write("No columns selected for dropping.")

    return df, to_drop, remove_duplicate


def display_dataset_summary(df, settings, saved_configuration_file):

    dropped_columns = settings.get("Dropped_columns_name", {})
    st.markdown("<h3>Original Dataset Summary</h3>", unsafe_allow_html=True)
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {', '.join(df.columns)}")
    st.write(f"**Missing values:** {df.isnull().sum().sum()}")
    st.markdown("---")

    st.markdown("<h3>Drop Unnecessary Columns</h3>", unsafe_allow_html=True)
    with st.expander("📊 Dataset Exploration & Column Management", expanded=False):
        # Process dropped columns and duplicates
        df, dropped_columns, remove_duplicates = duplicate_rows_drop_columns_in_dataset(
            df, dropped_columns, key="dataset_summary"
        )

        # Dictionary of settings to save
        config_updates = {
            "Dropped_columns_name": dropped_columns,
            "Duplicate_removel": remove_duplicates,
        }

        # Save each config item
        for config_key, config_value in config_updates.items():
            settings = save_configuration_if_updated(
                config_file_name=saved_configuration_file,
                new_config_data=config_value,
                config_data_key=config_key,
            )

    with st.expander("📄 Show Updated Dataset Preview", expanded=False):
        st.subheader("Updated Dataset Preview")
        st.dataframe(df.astype(str))

    with st.expander("📊 Show Updated Dataset Statistical Overview", expanded=False):
        st.subheader("Updated Dataset Statistical Overview")
        st.dataframe(df.describe(include="all").astype(str).transpose())

    st.write(f"**Current Shape:** {df.shape}")
    st.write(f"**Current Columns:** {', '.join(df.columns)}")
    st.write(f"**Current Missing values:** {df.isnull().sum().sum()}")
    st.markdown("---")

    return df, settings


def plot_correlation_map(df, settings, saved_configuration_file):
    """Plots the correlation heatmap of the dataset, efficiently handling column drops."""
    dropped_columns = []
    already_dropped_columns = []
    st.markdown("""---""")
    st.markdown(f"<h3>Correlation Plot</h3>", unsafe_allow_html=True)

    if settings:
        already_dropped_columns = settings["Dropped_columns_name"]
        st.info(
            f"✔️ Settings include dropped columns. Already dropped columns names: {', '.join(dropped_columns)}"
        )

    with st.expander("📈 Plot Correlation of the Variables", expanded=True):
        st.write(
            "This plot shows the correlation between the variables in the dataset. "
            "You can drop columns based on the correlation."
        )

        # Apply column drop and duplicate removal
        df, dropped_columns, remove_duplicates = duplicate_rows_drop_columns_in_dataset(
            df, dropped_columns, key="drop_corr"
        )

        # Combine new and already dropped columns
        updated_dropped_columns = dropped_columns + already_dropped_columns

        # Save updated settings
        config_updates = {
            "Dropped_columns_name": updated_dropped_columns,
            "Duplicate_removal": remove_duplicates,
        }

        for config_key, config_value in config_updates.items():
            settings = save_configuration_if_updated(
                config_file_name=saved_configuration_file,
                new_config_data=config_value,
                config_data_key=config_key,
            )

        if not df.empty:
            corr_matrix = df.corr()
            fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                vmax=1,
                vmin=-1,
                center=0,
                linewidths=0.1,
                annot_kws={"size": 10},
                square=True,
                ax=ax,
                fmt=".2f",
                cbar_kws={"shrink": 0.8},
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
            ax.set_title("Correlation Heatmap", fontsize=14)

            st.write(fig)
        else:
            st.warning(
                "🚨 No correlation plot generated. Make sure the dataset has numerical values."
            )

    st.markdown("""---""")
    return settings, df


def plot_bar(df, selected_feature, target_feature):
    # Checking if the selected feature is numerical and the target feature is categorical
    st.write(
        "When analyzing a numerical feature against a categorical feature, a bar plot can give you an idea of the distribution of the numerical variable across categories."
    )

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=df[target_feature].astype("category"),
        y=df[selected_feature],
        hue=df[selected_feature],
        palette="coolwarm",
        ax=ax,
    )

    # Adding labels and title
    ax.set_title(f"Bar Plot of {selected_feature} by {target_feature}", fontsize=16)
    ax.set_xlabel(target_feature, fontsize=14)
    ax.set_ylabel(selected_feature, fontsize=14)
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels for better visibility

    # Display the plot
    st.write(fig)


def exploratory_data_analysis(df):
    st.markdown("""---""")
    st.markdown(f"<h3>Exploratory Data Analysis</h3>", unsafe_allow_html=True)

    with st.expander("Plot Variables Against Target"):
        st.write(
            "Select an input feature to explore its relationship with the target feature."
        )
        selected_feature = st.selectbox(
            label="Select input feature for plot", options=df.columns
        )
        target_feature = st.selectbox(
            label="Select target feature for plot", options=df.columns
        )
        st.write(
            f"Exploring the relationship between `{selected_feature}` and `{target_feature}`."
        )
        plot_bar(
            df=df, selected_feature=selected_feature, target_feature=target_feature
        )
    st.markdown("""---""")
    return df
