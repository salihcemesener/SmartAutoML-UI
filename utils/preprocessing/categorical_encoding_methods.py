import numpy as np
import pandas as pd


def target_encoding(df, col, target_column):
    target_mean = df.groupby(col)[target_column].mean()
    df[col] = df[col].map(target_mean)
    return df


def weighted_target_encoding(df, col, target_column, smoothing=2):
    target_mean = df[target_column].mean()  # Global mean of the target column
    column_mean = df.groupby(col)[
        target_column
    ].mean()  # Mean of the target column for each category
    number_of_unique = df[col].value_counts()  # Number of occurrences for each category
    # Apply the weighted mean formula for each category
    encoded_values = df[col].map(
        lambda x: (number_of_unique[x] * column_mean[x] + smoothing * target_mean)
        / (number_of_unique[x] + smoothing)
    )
    df[col] = encoded_values
    return df


def k_fold_target_encoding(df, col, target_column, num_folds=2, smoothing=2):
    df_subsets = np.array_split(df, num_folds)
    for index, subset in enumerate(df_subsets):
        remaining_df = pd.concat(
            df_subsets[:index] + df_subsets[index + 1 :], ignore_index=True
        )
        # Calculate the global mean of the target column
        target_mean = remaining_df[target_column].mean()
        # Calculate the category-specific mean for each category in the remaining dataframe
        column_mean = remaining_df.groupby(col)[target_column].mean()
        # Calculate the number of occurrences for each category in the remaining dataframe
        number_of_unique = remaining_df[col].value_counts()
        # Apply the weighted mean formula to the current fold. If not use get method we got error because in subset maybe that variable not exists so we make that zero encoded value.
        subset_encoded = subset[col].map(
            lambda x: (
                number_of_unique.get(x, 0) * column_mean.get(x, target_mean)
                + smoothing * target_mean
            )
            / (number_of_unique.get(x, 0) + smoothing)
        )

        df.loc[subset.index, col] = subset_encoded
    return df
