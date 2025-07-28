import streamlit as st
import json
import os


def configuration_ui(settings, uploaded_json_file_name):
    st.markdown("---")
    st.markdown("<h2>Save Your Settings</h2>", unsafe_allow_html=True)

    st.write(
        "In this section, you can provide a name for the JSON file where your configuration settings will be saved. "
        "This allows you to store your adjustments and reuse them in the future."
    )
    if settings:
        json_file_name = st.text_input(
            "Enter a name for the settings file (.json format):",
            value=uploaded_json_file_name,
            help="This file will store all your settings adjustments. Choose a meaningful name.",
        )
    else:
        json_file_name = st.text_input(
            "Enter a name for the settings file (.json format):",
            value=uploaded_json_file_name if uploaded_json_file_name else "",
            help="This file will store all your settings adjustments. Choose a meaningful name.",
        )
        if json_file_name == "":
            st.error(
                "🚨 Please enter a valid name for the settings file or if you have already existing config file upload it."
            )
            st.stop()

    st.write(f"📁 Your settings will be saved as: `/model/cfg/{json_file_name}.json`")

    if settings:
        with st.expander(f"⚙️ Uploaded settings are:", expanded=False):
            st.json(settings)
    else:
        st.warning(
            f"🚨 No JSON file uploaded. Starting from scratch and saved to `{json_file_name}` json file."
        )

    st.markdown("---")
    return json_file_name


def save_configuration_if_updated(config_file_name, new_config_data, config_data_key):
    file_path = f"./model/cfg/{config_file_name}.json"

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = {}
    else:
        existing_data = {}
    existing_raw = existing_data.get(config_data_key, [])
    existing_values = set(json.dumps(d, sort_keys=True) for d in existing_raw)
    new_values = set(json.dumps(d, sort_keys=True) for d in new_config_data)

    if existing_values == new_values:
        return existing_data

    existing_data[config_data_key] = new_config_data

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    return existing_data
