# src/puv/visualise.py
from __future__ import annotations


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, widgets
from dython.nominal import associations
from matplotlib.offsetbox import AnchoredText

import warnings



def interactive_correlation_heatmap(df, title):
    """
    Generate an interactive correlation heatmap for a given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - title (str): The title of the correlation heatmap.

    Returns:
    None
    """

    # Calculate correlation matrix
    corr_matrix = df.corr()

    def plot_upper_triangular_correlation(feature):
        # Calculate correlations with selected feature
        correlations = corr_matrix[feature].sort_values(ascending=False)
        sorted_features = correlations.index.tolist()
        
        # Create a mask to hide the lower triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Plot correlation heatmap with masked lower triangle
        plt.figure(figsize=(40, 32))
        sns.heatmap(corr_matrix[sorted_features].loc[sorted_features], 
                    annot=False, cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
        plt.title(f'Correlation Matrix (sorted by {feature})', fontsize=30)
        plt.suptitle(title, fontsize=50)
        plt.show()

    # Get list of features for dropdown widget
    features_list = df.columns.tolist()

    # Create interactive dropdown
    interact(plot_upper_triangular_correlation, feature=widgets.Dropdown(options=features_list, description='Select Feature:'))


def plot_feature_distributions(df, name, save_path=None):
    """
    Plot the feature distributions for a given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the features.
    - name (str): The name/title of the feature set.

    Returns:
    None
    """

    feature_num = len(df.columns)
    nrows_needed = feature_num // 4 + 1

    # Step 2: Create bar charts for each feature
    fig, axes = plt.subplots(nrows=nrows_needed, ncols=4, figsize=(20, 60))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        df[col].value_counts().sort_index().plot(kind='bar', ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xticklabels([])  # Hide x-axis tick labels
        axes[i].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()
    plt.suptitle('Feature Distributions ' + name, fontsize=52)
    plt.subplots_adjust(top=0.95)

        # Save or show the plot
    if save_path != None:
        plt.savefig(save_path, dpi=300)
        plt.show()
    else:
        plt.show()





def plot_correlation_heatmap(df, title, save_path=None, first_column=None):
    """
    Plot a correlation heatmap for a given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - title (str): The title of the correlation heatmap.
    - save_path (str, optional): The path to save the plot. If not provided, the plot will be displayed but not saved.

    Returns:
    None
    """

    if first_column in df.columns:
    # Get a list of columns excluding the specific column
        cols = [col for col in df.columns if col != first_column]
        # Append the specific column to the end of the list
        cols.insert(0, first_column)
        # Reorder the DataFrame columns
        df = df[cols]
    else:
        print(f"'{first_column}' column does not exist in the DataFrame")


    # Calculate the Pearson correlation matrix
    corr_matrix = df.corr(method='pearson')

    # Create a heatmap of the sorted correlation matrix
    plt.figure(figsize=(max(len(df.columns)*2, 10), max(5, len(df.columns))))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=False,        # Display correlation coefficients
        cmap='coolwarm',    # Color map
        fmt=".2f",          # Format of annotations
        linewidths=0.5,     # Width of grid lines
        square=True,        # Ensure the heatmap cells are square
        cbar_kws={"shrink": .8},  # Color bar size
        vmax=1, 
        vmin=-1
    )

    # Set the title
    plt.suptitle(title, fontsize=max (14, len(df.columns)*2))

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save_path != None:
        plt.savefig(save_path, dpi=300)
        plt.show()
    else:
        plt.show()






def get_binary_and_categorical_columns(dataframe):

    
    """
    Extract binary and categorical column names from a DataFrame.
    
    Parameters:
    - dataframe: pd.DataFrame, the dataset
    
    Returns:
    - binary_columns: list, columns containing binary data
    - categorical_columns: list, columns containing categorical data
    
    """
    binary_columns = []
    categorical_columns = []
    
    for column in dataframe.columns:
        unique_vals = dataframe[column].nunique()
        dtype = dataframe[column].dtype
        
        # Binary columns (either numeric with 2 unique values or categorical with 2 unique values)
        if unique_vals == 2:
            binary_columns.append(column)
        # Categorical columns (non-numeric with more than 2 unique values or numeric with more than 2 categories)
        elif unique_vals > 2 and not pd.api.types.is_numeric_dtype(dataframe[column]):
            categorical_columns.append(column)

    binary_and_categorical_columns = binary_columns + categorical_columns
    
    return binary_and_categorical_columns



