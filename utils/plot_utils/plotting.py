import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt


def feature_outlier_analysis(df, col, target):
    sns.set_style("whitegrid")
    figs = []

    if target != col:
        # 1. Violin plot by Target
        fig_violin, ax_violin = plt.subplots(figsize=(8, 5))
        sns.violinplot(x=target, y=col, data=df, ax=ax_violin, inner="quartile")
        ax_violin.set_title(f"Violin Plot of '{col}' by Target '{target}'")
        ax_violin.set_xlabel("Target")
        ax_violin.set_ylabel(col)
        ax_violin.grid(True, linestyle="--", alpha=0.6)
        fig_violin.tight_layout()
        figs.append(fig_violin)

        # 2. KDE Plot of Feature by Target
        fig_kde, ax_kde = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=df, x=col, hue=target, fill=True, ax=ax_kde, common_norm=False)
        ax_kde.set_title(f"KDE Plot of '{col}' by Target '{target}'")
        ax_kde.set_xlabel(col)
        ax_kde.grid(True, linestyle="--", alpha=0.6)
        fig_kde.tight_layout()
        figs.append(fig_kde)

        # 3. Scatter Plot of Feature vs Target
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=col, y=target, data=df, alpha=0.6, edgecolor=None)
        ax_scatter.set_title(f"Scatter Plot of '{col}' vs Target '{target}'")
        ax_scatter.set_xlabel(col)
        ax_scatter.set_ylabel(target)
        ax_scatter.grid(True, linestyle="--", alpha=0.6)
        fig_scatter.tight_layout()
        figs.append(fig_scatter)

    # 4. Stripplot of Feature
    fig_strip, ax_strip = plt.subplots(figsize=(8, 4))
    sns.stripplot(data=df, y=col, ax=ax_strip, jitter=True, alpha=0.3, color="darkblue")
    ax_strip.set_title(f"Strip Plot of '{col}'")
    ax_strip.set_ylabel(col)
    ax_strip.grid(True, linestyle="--", alpha=0.6)
    fig_strip.tight_layout()
    figs.append(fig_strip)

    # 5. Histogram + Rugplot
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
    sns.histplot(df[col], bins=30, kde=True, ax=ax_hist, color="skyblue")
    sns.rugplot(df[col], ax=ax_hist, color="black", alpha=0.2)
    ax_hist.set_title(f"Distribution of '{col}'")
    ax_hist.set_xlabel(col)
    ax_hist.grid(True, linestyle="--", alpha=0.6)
    fig_hist.tight_layout()
    figs.append(fig_hist)

    return figs
