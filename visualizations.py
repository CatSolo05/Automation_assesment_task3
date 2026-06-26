"""Visualization module for Academic Predictor analysis."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def create_correlation_scatterplot(df, feature='Maths_Advanced', target='Software_Engineering_Final', 
                                    output_path='correlation_scatterplot.png',
                                    rmse_override=None,
                                    rmse_label='RMSE',
                                    big_rmse_text=True):
    """
    Generate a 2D scatterplot proving correlation between a feature and target variable.
    
    Args:
        df: DataFrame with cleaned data
        feature: Feature column name (X-axis)
        target: Target column name (Y-axis)
        output_path: Path to save the plot image
        rmse_override: Optional externally evaluated RMSE (e.g., holdout/test RMSE)
        rmse_label: Label shown next to RMSE value in the stats box
        big_rmse_text: Whether to render RMSE as a large prominent overlay
        
    Returns:
        Tuple of (figure, axes, correlation_coefficient, rmse)
    """
    # Extract data
    X = df[[feature]].values
    y = df[target].values
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(X.flatten(), y)[0, 1]
    
    # Fit regression line
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    ax.scatter(X, y, alpha=0.6, s=80, color='#2E86AB', edgecolors='#1E1E2E', linewidth=0.5)
    
    # Regression line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    ax.plot(X_line, y_line, color='#A23B72', linewidth=2.5, label='Regression Line')
    
    # Formatting
    ax.set_xlabel(f'{feature} Score', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{target} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Correlation: {feature} vs {target}', fontsize=14, fontweight='bold', pad=20)
    
    # Use the externally evaluated RMSE when provided.
    rmse_to_display = float(rmse_override) if rmse_override is not None else float(rmse)

    # Add statistics box
    stats_text = f'Correlation: {correlation:.3f}\n{rmse_label}: {rmse_to_display:.2f}\nSamples: {len(df)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    if big_rmse_text:
        ax.text(
            0.98,
            0.08,
            f'RMSE = {rmse_to_display:.2f}',
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=24,
            fontweight='bold',
            color='#A23B72',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#A23B72', linewidth=2, alpha=0.95),
        )
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_facecolor('#F5F5F5')
    fig.patch.set_facecolor('white')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Scatterplot saved to {output_path}')
    print(f'  Correlation Coefficient: {correlation:.4f}')
    print(f'  Displayed {rmse_label}: {rmse_to_display:.2f}')
    
    return output_path, correlation, rmse_to_display


def create_multifeature_scatterplot(df, features=['Maths_Advanced', 'Physics'], 
                                     target='Software_Engineering_Final',
                                     output_path='multifeature_scatterplot.png'):
    """
    Generate scatterplots for multiple features vs target (2x1 subplot).
    
    Args:
        df: DataFrame with cleaned data
        features: List of feature column names
        target: Target column name
        output_path: Path to save the plot image
        
    Returns:
        Path to saved image
    """
    fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
    
    for idx, feature in enumerate(features):
        X = df[[feature]].values
        y = df[target].values
        
        # Correlation and regression
        correlation = np.corrcoef(X.flatten(), y)[0, 1]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        
        ax = axes[idx] if len(features) > 1 else axes
        
        # Scatter plot
        ax.scatter(X, y, alpha=0.6, s=70, color='#2E86AB', edgecolors='#1E1E2E', linewidth=0.5)
        
        # Regression line
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(X_line)
        ax.plot(X_line, y_line, color='#A23B72', linewidth=2.5)
        
        # Formatting
        ax.set_xlabel(f'{feature}', fontsize=11, fontweight='bold')
        ax.set_ylabel(target if idx == 0 else '', fontsize=11, fontweight='bold')
        ax.set_title(f'r = {correlation:.3f}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#F5F5F5')
    
    fig.suptitle(f'Multi-Feature Correlation Analysis vs {target}', 
                 fontsize=14, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'✓ Multi-feature scatterplot saved to {output_path}')
    return output_path
