class DataPreprocessingOptionsHelperText:

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

    def list_outlier_detection_strategies():
        outlier_detection_options = {
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
        }
        return outlier_detection_options

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

    def get_outlier_handler_options():
        outlier_handler_options = {
            "remove": (
                "Completely removes rows containing outliers from the dataset. "
                "Reduces dataset size, which may affect class balance. Recommended when outliers are extreme and clearly invalid."
            ),
            "impute_median": (
                "Replaces each outlier with the column’s median value. "
                "Preserves row count and is robust to skewed data. Useful when the median reflects the typical value."
            ),
            "impute_mean": (
                "Substitutes outlier values with the column mean. "
                "Maintains dataset size but may distort distributions if data is heavily skewed."
            ),
            "mark": (
                "Flags outliers in a new boolean column (e.g., `feature_is_outlier`) without modifying the original data. "
                "Enables flexible downstream filtering, modeling, or analysis while keeping raw data intact."
            ),
            "skip": (
                "Takes no action on detected outliers. "
                "Outliers remain unchanged. Ideal for exploratory analysis or if another process will handle them."
            ),
            "log": (
                "Applies a log(x + 1) transformation to compress large values. "
                "Only applicable to strictly positive data."
            ),
            "sqrt": (
                "Applies a square root transformation to reduce the effect of large values. "
                "Only applicable to non-negative data."
            ),
            "box-cox": (
                "Applies the Box-Cox power transformation to make data more normally distributed. "
                "Requires all values to be positive."
            ),
            "yeo-johnson": (
                "A power transformation similar to Box-Cox, but supports zero and negative values. "
                "Useful for stabilizing variance and normalizing skewed data."
            ),
            "robust-scaler": (
                "Scales data using the median and IQR instead of mean and standard deviation. "
                "Robust to outliers and preserves relative relationships."
            ),
            "quantile": (
                "Maps feature values to a normal or uniform distribution based on quantile ranks. "
                "It transforms the feature so that its distribution matches a target distribution (e.g., Gaussian or uniform). "
                "Effectively handles outliers by spreading out dense regions and compressing sparse ones. "
                "Useful for making skewed data more Gaussian-like, especially before applying linear models."
            ),
            "winsorize": (
                "Limits (caps) extreme values to specified percentiles — for example, values below the 5th percentile "
                "are set to the 5th percentile value, and those above the 95th percentile are set to the 95th. "
                "This reduces the influence of outliers while preserving dataset size. "
                "Unlike detection-based methods, it does not identify outliers but modifies them to reduce their effect. "
                "Ideal for models sensitive to extreme values, such as linear regression."
            ),
        }
        return outlier_handler_options

    def list_multicollinearity_detection_options():
        multicollinearity_detection_options = {
            "Correlation Matrix (Pearson)": (
                "Visualizes pairwise linear relationships between numerical features using a heatmap. "
                "Helps identify highly correlated feature pairs that may introduce redundancy. "
                "The Pearson correlation coefficient ranges from -1 to 1 and is calculated as:\n"
                "`corr(X, Y) = cov(X, Y) / (std(X) * std(Y))`"
            ),
            "Variance Inflation Factor (VIF)": (
                "Quantifies how much the variance of a regression coefficient is inflated due to multicollinearity. "
                "Higher VIF values (typically > 5 or 10) suggest a feature is highly correlated with others "
                "and may be redundant. The formula is:\n"
                "`VIF = 1 / (1 - R²)` where R² is the coefficient of determination from regressing the feature "
                "on all other features."
            ),
        }
        return multicollinearity_detection_options
