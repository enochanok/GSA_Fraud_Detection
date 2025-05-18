# Research Paper Visualizations for Fraud Detection Analysis

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import pandas as pd

def create_research_visualizations(merged_df, anomaly_df, threshold):
    """
    Create publication-quality visualizations for fraud detection analysis.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        The merged transaction data
    anomaly_df : pandas.DataFrame
        The anomaly detection results
    threshold : float
        The anomaly threshold value
    """
    # Set the style for publication quality plots
    plt.style.use('default')  # Use default style instead of seaborn
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': '#f8f8f8',
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, figure=fig)

    # 1. Distribution of Transaction Amounts with Anomaly Highlight
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data=merged_df, x='transaction amount', bins=50, ax=ax1, color='#2ecc71')
    ax1.set_title('Distribution of Transaction Amounts')
    ax1.set_xlabel('Transaction Amount ($)')
    ax1.set_ylabel('Frequency')
    # Add vertical lines for mean and median
    ax1.axvline(merged_df['transaction amount'].mean(), color='red', linestyle='--', label='Mean')
    ax1.axvline(merged_df['transaction amount'].median(), color='green', linestyle='--', label='Median')
    ax1.legend()

    # 2. Anomaly Score Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=anomaly_df, x='ensemble_score', bins=50, ax=ax2, color='#3498db')
    ax2.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    ax2.set_title('Distribution of Anomaly Scores')
    ax2.set_xlabel('Ensemble Anomaly Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # 3. Correlation between Different Detection Methods
    ax3 = fig.add_subplot(gs[1, 0])
    scatter = ax3.scatter(anomaly_df['kmeans_score_norm'], 
                         anomaly_df['iso_score_norm'],
                         c=anomaly_df['ensemble_score'],
                         cmap='viridis',
                         alpha=0.6)
    ax3.set_title('Correlation: KMeans vs Isolation Forest Scores')
    ax3.set_xlabel('KMeans Score (normalized)')
    ax3.set_ylabel('Isolation Forest Score (normalized)')
    plt.colorbar(scatter, ax=ax3, label='Ensemble Score')

    # 4. LSTM Reconstruction Error Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(anomaly_df['recon_error'],
                         anomaly_df['ensemble_score'],
                         c=anomaly_df['is_anomaly'],
                         cmap='coolwarm',
                         alpha=0.6)
    ax4.set_title('LSTM Reconstruction Error vs Ensemble Score')
    ax4.set_xlabel('Reconstruction Error')
    ax4.set_ylabel('Ensemble Score')
    plt.colorbar(scatter, ax=ax4, label='Is Anomaly')

    # 5. Top Anomalies by Transaction Amount
    ax5 = fig.add_subplot(gs[2, 0])
    top_anomalies = merged_df.iloc[anomaly_df[anomaly_df['is_anomaly'] == 1].index]
    top_anomalies_sorted = top_anomalies.nlargest(10, 'transaction amount')
    sns.barplot(data=top_anomalies_sorted,
                x='transaction amount',
                y='merchant name',
                ax=ax5,
                color='#e74c3c')
    ax5.set_title('Top 10 Anomalies by Transaction Amount')
    ax5.set_xlabel('Transaction Amount ($)')
    ax5.set_ylabel('Merchant Name')

    # 6. Anomaly Distribution by Region
    ax6 = fig.add_subplot(gs[2, 1])
    region_anomalies = merged_df.iloc[anomaly_df[anomaly_df['is_anomaly'] == 1].index]['region'].value_counts()
    sns.barplot(x=region_anomalies.index, y=region_anomalies.values, ax=ax6, color='#9b59b6')
    ax6.set_title('Distribution of Anomalies by Region')
    ax6.set_xlabel('Region')
    ax6.set_ylabel('Number of Anomalies')
    plt.xticks(rotation=45)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_temporal_analysis(merged_df, anomaly_df):
    """
    Create temporal analysis visualizations.
    """
    plt.figure(figsize=(15, 6))
    merged_df['transaction date'] = pd.to_datetime(merged_df['transaction date'])
    anomaly_dates = merged_df.iloc[anomaly_df[anomaly_df['is_anomaly'] == 1].index]['transaction date']
    sns.histplot(data=anomaly_dates, bins=30, color='#2ecc71')
    plt.title('Temporal Distribution of Detected Anomalies')
    plt.xlabel('Transaction Date')
    plt.ylabel('Number of Anomalies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('anomaly_temporal_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_mcc_analysis(merged_df, anomaly_df):
    """
    Create merchant category analysis visualizations.
    """
    plt.figure(figsize=(12, 6))
    mcc_anomalies = merged_df.iloc[anomaly_df[anomaly_df['is_anomaly'] == 1].index]['mcc description'].value_counts().head(10)
    sns.barplot(x=mcc_anomalies.values, y=mcc_anomalies.index, color='#3498db')
    plt.title('Top 10 Merchant Categories with Anomalies')
    plt.xlabel('Number of Anomalies')
    plt.ylabel('Merchant Category')
    plt.tight_layout()
    plt.savefig('mcc_anomaly_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_correlation_analysis(merged_df):
    """
    Create correlation analysis visualizations.
    """
    numerical_features = ['transaction amount', 'mcc', 'merchant zip']
    corr_matrix = merged_df[numerical_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_boxplot_analysis(merged_df, anomaly_df):
    """
    Create boxplot analysis visualizations.
    """
    # Reset indices to avoid duplicate label issues
    plot_data = merged_df.reset_index(drop=True).copy()
    anomaly_data = anomaly_df.reset_index(drop=True).copy()
    
    # Create a new column for anomaly status
    plot_data['is_anomaly'] = anomaly_data['is_anomaly']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_data, x='is_anomaly', y='transaction amount', color='#2ecc71')
    plt.title('Transaction Amount Distribution: Normal vs Anomalous')
    plt.xlabel('Is Anomaly (0: Normal, 1: Anomaly)')
    plt.ylabel('Transaction Amount ($)')
    plt.tight_layout()
    plt.savefig('transaction_amount_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(merged_df, anomaly_df):
    """
    Print summary statistics of the analysis.
    """
    # Reset indices to avoid duplicate label issues
    merged_data = merged_df.reset_index(drop=True)
    anomaly_data = anomaly_df.reset_index(drop=True)
    
    region_anomalies = merged_data.iloc[anomaly_data[anomaly_data['is_anomaly'] == 1].index]['region'].value_counts()
    
    print("\nSummary Statistics of Detected Anomalies:")
    print(f"Total number of anomalies detected: {len(anomaly_data[anomaly_data['is_anomaly'] == 1])}")
    print(f"Percentage of transactions flagged as anomalies: {(len(anomaly_data[anomaly_data['is_anomaly'] == 1]) / len(anomaly_data) * 100):.2f}%")
    print("\nTop 5 regions with most anomalies:")
    print(region_anomalies.head())
    print("\nAverage transaction amount of anomalies: $", 
          f"{merged_data.iloc[anomaly_data[anomaly_data['is_anomaly'] == 1].index]['transaction amount'].mean():.2f}")
    print("Average transaction amount of normal transactions: $", 
          f"{merged_data.iloc[anomaly_data[anomaly_data['is_anomaly'] == 0].index]['transaction amount'].mean():.2f}")

# Example usage:
# create_research_visualizations(merged_df, anomaly_df, threshold)
# create_temporal_analysis(merged_df, anomaly_df)
# create_mcc_analysis(merged_df, anomaly_df)
# create_correlation_analysis(merged_df)
# create_boxplot_analysis(merged_df, anomaly_df)
# print_summary_statistics(merged_df, anomaly_df) 