import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt


def feature_outlier_analysis(df, col, target):

    sns.set_style("whitegrid")
    figs = []

    # Boxplot of Feature by Target + stripplot
    if target != col:
        fig_by_target, ax_by_target = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=target, y=col, data=df, ax=ax_by_target, fliersize=0)
        ax_by_target.set_title(f"Boxplot of '{col}' by Target '{target}'")
        ax_by_target.set_xlabel("Target")
        ax_by_target.set_ylabel(col)
        ax_by_target.grid(True, linestyle="--", alpha=0.6)
        fig_by_target.tight_layout()
        figs.append(fig_by_target)

        # KDE Plot of Feature by Target
        fig_kde, ax_kde = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=df, x=col, hue=target, fill=True, ax=ax_kde)
        ax_kde.set_title(f"KDE Plot of '{col}' by Target '{target}'")
        ax_kde.set_xlabel(col)
        ax_kde.grid(True, linestyle="--", alpha=0.6)
        fig_kde.tight_layout()
        figs.append(fig_kde)

    # Boxplot of Feature Itself
    fig_by_inside, ax_by_inside = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=col, data=df, ax=ax_by_inside)
    ax_by_inside.set_title(f"Boxplot of Feature '{col}'")
    ax_by_inside.set_ylabel(col)
    ax_by_inside.grid(True, linestyle="--", alpha=0.6)
    fig_by_inside.tight_layout()
    figs.append(fig_by_inside)

    return figs
