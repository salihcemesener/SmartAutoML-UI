import pandas as pd
import streamlit as st
import category_encoders as ce  # For Target and Binary Encoding
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing.encoding_methods import (
    target_encoding,
    weighted_target_encoding,
    k_fold_target_encoding,
)


def get_categorical_encoding_options():
    encoding_options = {
        "Label Encoding": (
            "Assigns a unique integer to each category (e.g., red=0, blue=1). "
            "Best for ordinal data (where order matters) or tree-based models. "
            "May introduce false numerical relationships in linear models."
        ),
        "One-Hot Encoding": (
            "Creates a binary column for each category (e.g., red → [1,0,0], blue → [0,1,0]). "
            "Ideal for nominal data (categories without order) and linear models. "
            "Can cause high memory usage and computation time if there are too many unique categories."
        ),
        "Frequency Encoding": (
            "Replaces categories with their occurrence frequency in the dataset. "
            "Useful when category frequency is meaningful and captures distribution. "
            "Does not create additional columns, making it memory efficient."
        ),
        "Ordinal Encoding": (
            "Assigns integers based on a specified order (e.g., Low=0, Medium=1, High=2). "
            "Best for ordered categories. If order is not well-defined, it may reduce model performance."
        ),
        "Binary Encoding": (
            "Converts categories to binary format by first mapping to integers, then splitting into binary digits. "
            "Efficient for high-cardinality categorical data as it reduces dimensionality compared to one-hot encoding."
        ),
        "Target Encoding": (
            "Replaces each category with the **mean target value** for that category. "
            "Works well for categorical features in classification and regression problems. "
            "Can lead to **data leakage** if not handled correctly."
        ),
        "Weighted Mean Target Encoding (Bayesian Mean Encoding)": (
            "A variation of target encoding that prevents overfitting by smoothing category means with the global mean. "
            "The **hyperparameter** controls the balance between category mean and global mean:\n"
            "- **Small values (0-2):** More weight on category mean; works well with enough data per category.\n"
            "- **Large values (10-100):** More weight on global mean; helps regularize rare categories but may lose distinctions."
        ),
        "K-Fold Target Encoding": (
            "A variation of weighted mean target encoding that reduces data leakage using cross-validation. "
            "Splits the dataset into `K` folds and applies target encoding using other folds, preventing direct leakage. "
            "This method helps generalize encoding and reduces overfitting compared to simple target encoding.\n\n"
            "**Effects of Hyperparameter (Smoothing Factor):**\n"
            "- **Small values (0-2):** Dominated by category mean; can overfit if a category has few samples.\n"
            "- **Large values (10-100):** Pulls rare categories toward global mean, reducing overfitting but losing category-specific details.\n\n"
            "**Effects of K (Number of Folds):**\n"
            "- **Small K (e.g., 2-3 folds):** Higher variance, less stable encoding due to fewer training splits.\n"
            "- **Large K (e.g., 10+ folds):** More stable encoding, but computation cost increases.\n"
            "- **Typical Choice:** `K=5` or `K=10` balances stability and computational efficiency."
        ),
    }
    return encoding_options


def get_remove_outliers_options():
    remove_outliers_options = {
        "Do Nothing": ("Leave the column unchanged-no outlier removal applied."),
        "Z-Score Method (Standard Score)": (
            "Calculate the standardized score for each observation:\n\n"
            "Z = (X - μ) / σ\n\n"
            "**Hyperparameters**:\n\n"
            "- threshold (float): cutoff for |Z| beyond which points are flagged (commonly 2.5 or 3.0).\n\n"
            "**Pros**:\n\n"
            "- Simple and intuitive when data are approximately Gaussian.\n"
            "- Fast to compute on large datasets.\n\n"
            "**Cons**:\n\n"
            "- Sensitive to extreme values, which can skew μ and σ.\n"
            "- Assumes data follow a roughly normal distribution.\n\n"
        ),
        "Interquartile Range (IQR) Method": (
            "Use the middle 50% spread of the data:\n\n"
            "IQR = Q3 – Q1\n\n"
            "Where Q1 and Q3 are the 25th and 75th percentiles, respectively.\n\n"
            "**Hyperparameters**:\n\n"
            "- multiplier (float): factor multiplied by IQR to define outlier fences (default = 1.5).\n\n"
            "**Pros**:\n\n"
            "- Robust to extreme values and skewed distributions.\n"
            "- Easy to understand and implement.\n\n"
            "**Cons**:\n\n"
            "- May be less effective on very small samples.\n"
            "- Uses a fixed multiplier that may need adjustment for different datasets.\n\n"
        ),
        "Median Absolute Deviation (MAD)": (
            "Measure deviation from the median:\n\n"
            "MAD = median(|X - median(X)|)\n\n"
            "**Hyperparameters**:\n\n"
            "- scale_factor (float): scaling applied to MAD (1.4826 for consistency with σ under normality).\n"
            "- threshold (float): cutoff on scaled deviations for outlier detection (often T = 3).\n\n"
            "**Pros**:\n\n"
            "- Extremely robust to outliers and heavy tails.\n"
            "- Does not assume any specific distribution.\n\n"
            "**Cons**:\n\n"
            "- Requires two median computations, so slightly more costly.\n"
            "- The scaling factor can be non‐intuitive.\n\n"
        ),
        "Winsorization (Percentile Capping)": (
            "Limit extreme values to chosen percentiles.\n\n"
            "**Hyperparameters**:\n\n"
            "- lower_percentile (float): lower bound percentile (e.g., 0.05 for 5th percentile).\n"
            "- upper_percentile (float): upper bound percentile (e.g., 0.95 for 95th percentile).\n\n"
            "**Pros**:\n\n"
            "- Preserves dataset size and relative ranking.\n"
            "- Simple and flexible percentile choices.\n\n"
            "**Cons**:\n\n"
            "- Can distort distribution tails if caps are too aggressive.\n"
            "- Loses information about true extreme values.\n\n"
        ),
        "Isolation Forest (Unsupervised ML)": (
            "Build an ensemble of isolation trees by randomly splitting features.\n\n"
            "**Hyperparameters**:\n\n"
            "- n_estimators (int): number of trees in the forest (default = 100).\n"
            "- max_samples (int or float): samples per tree (default = 'auto').\n"
            "- contamination (float): expected proportion of outliers (e.g., 0.05).\n"
            "- max_features (int or float): features per split.\n\n"
            "**Pros**:\n\n"
            "- Effective in high-dimensional spaces.\n"
            "- Makes no assumptions about data distribution.\n\n"
            "**Cons**:\n\n"
            "- Requires tuning of multiple parameters.\n"
            "- Training can be slower on very large datasets.\n\n"
        ),
        "Local Outlier Factor (LOF)": (
            "Estimate each point’s local density relative to its neighbors.\n\n"
            "**Hyperparameters**:\n\n"
            "- n_neighbors (int): neighborhood size (default = 20).\n"
            "- contamination (float): expected fraction of outliers (default = 0.05).\n"
            "- metric (str): e.g., 'euclidean', 'manhattan', 'cosine'.\n\n"
            "**Pros**:\n\n"
            "- Detects contextual anomalies within clusters.\n"
            "- Non-parametric.\n\n"
            "**Cons**:\n\n"
            "- Sensitive to n_neighbors.\n"
            "- Distance computations can be expensive.\n\n"
        ),
        "Mahalanobis Distance": (
            "Compute the multivariate distance from the mean:\n\n"
            "D² = (x - μ)^T Σ⁻¹ (x - μ)\n\n"
            "- `μ`: mean vector of the features\n"
            "- `Σ`: covariance matrix of the features\n\n"
            "**Hyperparameters**:\n\n"
            "- `threshold` (float): maximum squared distance allowed.\n"
            "  This value is chosen from the chi-square distribution with p degrees of freedom,\n"
            "  so that approximately (1 – α)% of data falls within the normal ellipse.\n"
            "  For example, with p=2 and α=0.05, threshold ≈ 5.99 covers ~95% of points.\n\n"
            "**Pros**:\n\n"
            "- Accounts for feature correlations and scale differences.\n"
            "- Defines an elliptical decision boundary matching the data shape.\n\n"
            "**Cons**:\n\n"
            "- Requires a well-conditioned covariance matrix.\n"
            "- Unstable if features are collinear or sample size is small.\n\n"
            "**How it works**:\n\n"
            "- Center data by subtracting μ so the cloud is at the origin.\n"
            "- ‘Whiten’ data by applying Σ⁻¹/², rescaling each direction by its variance.\n"
            "- Compute Euclidean length in this whitened space; points beyond the threshold ellipse are outliers.\n\n"
        ),
    }
    return remove_outliers_options


def get_missing_value_options():

    numerical_missing_options = {
        "Fill with Mean": "Use when data follows a normal distribution. Not suitable for skewed data. Replaces NaN with the column mean.",
        "Fill with Median": "Best for skewed data or when outliers are present. Replaces NaN with the column median.",
        "Fill with Custom Value": "Manually set a value to fill missing data. Be cautious, as an inappropriate choice may degrade model performance.",
        "Remove Rows": "Use when the percentage of missing data in a column is small. Instead of filling, simply drop the rows.",
        "Remove Columns": "Recommended when more than 30%-40% of a column's data is missing and cannot be reliably predicted. Dropping prevents misleading model learning.",
        "Do Nothing": "Use if the model can naturally handle missing values without preprocessing.",
    }

    categorical_missing_options = {
        "Fill with Mode": "Replaces missing values with the most frequently occurring category in the column. Suitable when missing values follow a common pattern.",
        "Fill with Custom Value": "Manually set a value to fill missing data. Be careful, as an incorrect choice may introduce bias or mislead the model.",
        "Remove Rows": "Use when the percentage of missing data is small. Instead of filling, simply drop the rows to maintain data integrity.",
        "Remove Columns": "Recommended when more than 30%-40% of a column's data is missing and cannot be reliably predicted. Dropping prevents misleading model learning.",
        "Do Nothing": "Use if the model can naturally handle missing values without preprocessing.",
    }
    return numerical_missing_options, categorical_missing_options


def apply_numeric_fill_method(fill_method_name, df, col):
    try:

        if fill_method_name == "Fill with Mean":
            df[col] = df[col].fillna(df[col].mean())
            st.success(
                f"{col} column missing values are filled using mean. Mean value: {df[col].mean()}"
            )

        elif fill_method_name == "Fill with Median":
            df[col] = df[col].fillna(df[col].median())
            st.success(
                f"{col} column missing values are filled using median. Median of the columns: {df[col].median()}"
            )

        elif fill_method_name == "Fill with Custom Value":

            st.session_state[f"custom_value_{col}"] = st.number_input(
                f"Custom value for {col}",
                value=(
                    st.session_state[f"custom_value_{col}"]
                    if st.session_state[f"custom_value_{col}"]
                    else 0
                ),
            )
            df[col] = df[col].fillna(st.session_state[f"custom_value_{col}"])
            st.success(
                f'{col} column missing values filled with custom value: {st.session_state[f"custom_value_{col}"]}.'
            )

        elif fill_method_name == "Remove Rows":
            df.update(df.dropna(subset=[col]))
            st.success(f"Rows with missing values in {col} column removed.")

        elif fill_method_name == "Remove Columns":
            df = df.drop(columns=[col])
            st.success(f"{col} column removed from the DataFrame.")

        return df

    except Exception as error:
        st.error(
            f"🚨 Error occurred while handling missing values in numeric column {col} using {fill_method_name}: {repr(error)}"
        )
        return df


def apply_categorical_fill_method(fill_method_name, df, col):
    try:
        if fill_method_name == "Fill with Mode":
            mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_value)
            st.success(f"{col} column missing values filled with mode: {mode_value}.")

        elif fill_method_name == "Fill with Custom Value":
            df = fill_with_custom_value_categorical(df, col)
            st.success(f"{col} column missing values filled with custom value.")

        elif fill_method_name == "Remove Rows":
            df = df.dropna(subset=[col])
            st.success(f"Rows with missing values in {col} column removed.")

        elif fill_method_name == "Remove Columns":
            df = df.drop(columns=[col])
            st.success(f"{col} column removed from the DataFrame.")

        return df

    except Exception as error:
        st.error(
            f"🚨 Error occurred while handling missing values in column {col} using {fill_method_name}: {repr(error)}"
        )
        return df


def fill_with_custom_value_categorical(df, col):
    try:
        sorted_values = sorted(
            df[col].dropna().value_counts().index,
            key=lambda x: df[col].value_counts()[x],
            reverse=True,
        )
        default_value = st.session_state.get(f"custom_value_{col}", sorted_values[0])
        default_index = (
            sorted_values.index(default_value) if default_value in sorted_values else 0
        )
        st.session_state[f"custom_value_{col}"] = st.selectbox(
            f"Select from existing categorical variable for {col}",
            sorted_values,
            help="Select one of the existing values for filling missing. The values are sorted by frequency in descending order.",
            index=default_index,
        )

        df[col] = df[col].fillna(st.session_state[f"custom_value_{col}"])
        return df

    except Exception as error:
        st.error(
            f"🚨 Error occurred while filling missing values for {col} with custom value: {repr(error)}"
        )
        return df


def apply_categorical_to_numerical(fill_method_name, df, col):
    try:
        columns = df.columns
        st.session_state[f"num_folds_{col}"] = st.session_state.get(
            f"num_folds_{col}", 5
        )
        st.session_state[f"target_col_{col}"] = st.session_state.get(
            f"target_col_{col}", columns[0]
        )
        st.session_state[f"smoothing_factor_{col}"] = st.session_state.get(
            f"smoothing_factor_{col}", 10.0
        )

        unique_vals = df[col].dropna().unique().tolist()
        default_order = st.session_state.get(f"variable_order_{col}")

        if len(unique_vals) > 10:
            if fill_method_name == "Ordinal Encoding":
                confirm = st.checkbox(
                    "🚨 That column has many unique values. Are you sure you want to apply ordinal encoding? This setting may significantly increase the size of your JSON configuration file.",
                    key=f"ordinal_encoding_variable_order_{col}",
                )
                if confirm:
                    st.session_state[f"variable_order_{col}"] = (
                        default_order or unique_vals
                    )
        else:
            st.session_state[f"variable_order_{col}"] = default_order or unique_vals

        def set_target_column(label):
            st.session_state[f"target_col_{col}"] = st.selectbox(
                label=label,
                options=columns,
                index=list(columns).index(st.session_state[f"target_col_{col}"]),
                key=f"set_target_col_{col}",
            )

        def set_smoothing_factor():
            st.session_state[f"smoothing_factor_{col}"] = st.slider(
                "Set smoothing factor:",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state[f"smoothing_factor_{col}"],
                step=1.0,
                key=f"set_smoothing_factor_{col}",
            )

        def set_num_folds():
            st.session_state[f"num_folds_{col}"] = st.slider(
                "Select number of folds (K):",
                min_value=2,
                max_value=10,
                value=st.session_state[f"num_folds_{col}"],
                step=1,
                key=f"set_num_folds_{col}",
            )

        if fill_method_name == "Label Encoding":
            df[col] = LabelEncoder().fit_transform(df[col])
            st.success(f"**{col}** converted using Label Encoding.")

        elif fill_method_name == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            st.success(f"**{col}** converted using One-Hot Encoding.")

        elif fill_method_name == "Frequency Encoding":
            df[col] = df[col].map(df[col].value_counts())
            st.success(f"**{col}** converted using Frequency Encoding.")

        elif fill_method_name == "Ordinal Encoding":
            st.session_state[f"variable_order_{col}"] = st.multiselect(
                f"Define the custom order for **{col}**:",
                df[col].dropna().unique().tolist(),
                default=st.session_state[f"variable_order_{col}"],
                help="Manually define the meaningful order (e.g., Low < Medium < High).",
            )
            mapping = {
                val: idx
                for idx, val in enumerate(st.session_state[f"variable_order_{col}"])
            }
            df[col] = df[col].map(mapping)
            st.success(f"**{col}** converted using Ordinal Encoding.")

        elif fill_method_name == "Binary Encoding":
            df = ce.BinaryEncoder(cols=[col]).fit_transform(df)
            st.success(f"**{col}** converted using Binary Encoding.")

        elif fill_method_name == "Target Encoding":
            set_target_column(
                f"Select Target Column for Target Encoding for **{col}**:"
            )
            df = target_encoding(df, col, st.session_state[f"target_col_{col}"])
            st.success(
                f"**{col}** converted using Target Encoding based on {st.session_state[f'target_col_{col}']}."
            )

        elif (
            fill_method_name == "Weighted Mean Target Encoding (Bayesian Mean Encoding)"
        ):
            set_target_column(
                f"Select Target Column for Weighted Mean Target Encoding for **{col}**:"
            )
            set_smoothing_factor()
            df = weighted_target_encoding(
                df,
                col,
                st.session_state[f"target_col_{col}"],
                st.session_state[f"smoothing_factor_{col}"],
            )
            st.success(
                f"**{col}** converted using Weighted Target Encoding with smoothing {st.session_state[f'smoothing_factor_{col}']} based on {st.session_state[f'target_col_{col}']}."
            )

        elif fill_method_name == "K-Fold Target Encoding":
            set_target_column(
                f"Select Target Column for K-Fold Target Encoding for **{col}**:"
            )
            set_num_folds()
            set_smoothing_factor()
            df = k_fold_target_encoding(
                df,
                col,
                st.session_state[f"target_col_{col}"],
                num_folds=st.session_state[f"num_folds_{col}"],
                smoothing=st.session_state[f"smoothing_factor_{col}"],
            )
            st.success(
                f"**{col}** converted using K-Fold Target Encoding with {st.session_state[f'num_folds_{col}']} folds and smoothing {st.session_state[f'smoothing_factor_{col}']} based on {st.session_state[f'target_col_{col}']}."
            )

        else:
            st.warning(f"No valid encoding method selected for {col}.")

        st.markdown("""---""")
        return df

    except Exception as e:
        st.error(f"🚨 Error occurred while encoding **{col}**: {repr(e)}")
        return df


def apply_outlier_removal(fill_method_name, df, col):
    try:
        st.session_state[f"z_score_th_{col}"] = st.session_state.get(
            f"z_score_th_{col}", 3
        )

        st.session_state[f"IQR_multiplier_{col}"] = st.session_state.get(
            f"IQR_th_{col}", 1.5
        )

        st.session_state[f"MAD_scale_factor_{col}"] = st.session_state.get(
            f"MAD_scale_factor_{col}", 1.4826
        )
        st.session_state[f"MAD_th_{col}"] = st.session_state.get(f"MAD_th_{col}", 3)

        st.session_state[f"Winsorization_upper_percentile_{col}"] = (
            st.session_state.get(f"Winsorization_upper_percentile_{col}", 95)
        )
        st.session_state[f"Winsorization_lower_percentile_{col}"] = (
            st.session_state.get(f"Winsorization_lower_percentile_{col}", 5)
        )

        st.session_state[f"Isolation_Forest_n_estimators_{col}"] = st.session_state.get(
            f"Isolation_Forest_n_estimators_{col}", 100
        )
        st.session_state[f"Isolation_Forest_max_samples_{col}"] = st.session_state.get(
            f"Isolation_Forest_max_samples_{col}", "auto"
        )
        st.session_state[f"Isolation_Forest_contamination_{col}"] = (
            st.session_state.get(f"Isolation_Forest_contamination_{col}", 0.05)
        )
        st.session_state[f"Isolation_Forest_max_features_{col}"] = st.session_state.get(
            f"Isolation_Forest_max_features_{col}", 5
        )

        st.session_state[f"LOF_n_neighbors_{col}"] = st.session_state.get(
            f"LOF_n_neighbors_{col}", 20
        )
        st.session_state[f"LOF_contamination_{col}"] = st.session_state.get(
            f"LOF_contamination_{col}", 0.05
        )
        st.session_state[f"LOF_metric_{col}"] = st.session_state.get(
            f"LOF_metric_{col}", "euclidian"
        )

        st.session_state[f"Mahalanobis_th_{col}"] = st.session_state.get(
            f"LOF_n_neighbors_{col}", 20
        )
        if fill_method_name == "Do Nothing":
            st.info(
                f"Skipping outlier removal for column `{col}`; values remain unchanged."
            )

        elif fill_method_name == "Z-Score Method (Standard Score)":
            pass
        elif fill_method_name == "Interquartile Range (IQR) Method":
            pass
        elif fill_method_name == "Median Absolute Deviation (MAD)":
            pass
        elif fill_method_name == "Winsorization (Percentile Capping)":
            pass
        elif fill_method_name == "Isolation Forest (Unsupervised ML)":
            pass
        elif fill_method_name == "Local Outlier Factor (LOF)":
            pass
        elif fill_method_name == "Mahalanobis Distance":
            pass
        else:
            st.warning(f"No valid remove outliers method selected for {col}.")
        st.markdown("""---""")
        return df
    except Exception as e:
        st.error(f"🚨 Error occurred while remove outliers **{col}**: {repr(e)}")
        return df
