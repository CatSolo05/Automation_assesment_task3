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


def create_actual_vs_predicted_plot(actual, predicted, output_path='actual_vs_predicted.png', title='Predicted vs Actual Marks'):
    """Generate a plot comparing predicted marks against actual marks."""
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    min_val = float(min(actual.min(), predicted.min()))
    max_val = float(max(actual.max(), predicted.max()))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(actual, predicted, s=85, alpha=0.75, color='#2E86AB', edgecolors='#1E1E2E', linewidth=0.5)
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', linewidth=2.5, color='#A23B72', label='Ideal Line')

    ax.set_xlabel('Actual Marks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Marks', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=18)

    stats_text = f'RMSE: {rmse:.2f}\nSamples: {len(actual)}'
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        family='monospace',
    )

    ax.text(
        0.98,
        0.08,
        f'RMSE = {rmse:.2f}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=24,
        fontweight='bold',
        color='#A23B72',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#A23B72', linewidth=2, alpha=0.95),
    )

    ax.set_xlim(min_val - 2, max_val + 2)
    ax.set_ylim(min_val - 2, max_val + 2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_facecolor('#F5F5F5')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Actual-vs-predicted plot saved to {output_path}')
    print(f'  RMSE: {rmse:.4f}')
    return output_path, rmse


def create_uml_class_diagram(output_path='uml_class_diagram.png'):
    """Generate a UML-style class diagram for the project architecture."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    def draw_class(x, y, width, height, title, attributes, methods, title_color='#1E1E2E'):
        rect = plt.Rectangle((x, y), width, height, fill=True, facecolor='#F5F5F5', edgecolor='#1E1E2E', linewidth=2)
        ax.add_patch(rect)

        title_h = 0.75
        attr_h = max(1.25, 0.42 * len(attributes) + 0.35)
        method_h = max(1.5, 0.42 * len(methods) + 0.35)

        ax.add_line(plt.Line2D([x, x + width], [y + height - title_h, y + height - title_h], color='#1E1E2E', linewidth=1.5))
        ax.add_line(plt.Line2D([x, x + width], [y + height - title_h - attr_h, y + height - title_h - attr_h], color='#1E1E2E', linewidth=1.5))

        ax.text(x + width / 2, y + height - 0.38, title, ha='center', va='center', fontsize=14, fontweight='bold', color=title_color)

        attr_text = '\n'.join(attributes)
        method_text = '\n'.join(methods)
        ax.text(x + 0.2, y + height - title_h - 0.18, attr_text, ha='left', va='top', fontsize=10, family='monospace', color='#333333')
        ax.text(x + 0.2, y + height - title_h - attr_h - 0.18, method_text, ha='left', va='top', fontsize=10, family='monospace', color='#333333')

        return {'x': x, 'y': y, 'w': width, 'h': height}

    predictor_box = draw_class(
        0.8,
        5.7,
        4.4,
        3.4,
        'AcademicPredictor',
        [
            '+ feature_columns: list[str]',
            '+ target_column: str',
            '+ model: Pipeline',
        ],
        [
            '+ load_data(path)',
            '+ clean_data(df)',
            '+ prepare_data(df)',
            '+ train(X_train, y_train)',
            '+ evaluate_rmse(X_test, y_test)',
            '+ cross_validate(df)',
            '+ predict_student(student_row)',
        ],
        title_color='#A23B72',
    )

    mark_box = draw_class(
        8.0,
        6.0,
        4.1,
        2.8,
        'MarkPredictor',
        [
            '+ model: LinearRegression',
        ],
        [
            '+ fit(X, y)',
            '+ predict(X)',
        ],
        title_color='#2E86AB',
    )

    utilities_box = draw_class(
        4.3,
        1.0,
        5.5,
        3.6,
        'Assessment Utilities',
        [
            '+ check_data_reliability()',
            '+ check_data_privacy()',
            '+ anonymize_and_save()',
            '+ calculate_file_sha256()',
            '+ secure_load_clean_and_anonymize()',
            '+ run_bias_audit()',
        ],
        [
            '+ log_rmse()',
            '+ bias_audit_report()',
            '+ cross_validation_check()',
            '+ detailed_cross_validation_report()',
        ],
        title_color='#1E1E2E',
    )

    # Relationship arrows
    def arrow(start, end):
        ax.annotate(
            '',
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle='->', lw=2.2, color='#1E1E2E', shrinkA=0, shrinkB=0),
        )

    arrow((5.2, 7.0), (8.0, 7.2))
    arrow((3.0, 5.7), (6.1, 4.6))
    arrow((9.6, 6.0), (8.0, 4.8))

    ax.text(6.6, 9.4, 'UML Class Diagram: Predictive Assessment System', ha='center', va='center', fontsize=18, fontweight='bold', color='#1E1E2E')
    ax.text(6.6, 8.95, 'Encapsulation, modularity, and reusable validation logic', ha='center', va='center', fontsize=11, color='#555555')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ UML class diagram saved to {output_path}')
    return output_path
