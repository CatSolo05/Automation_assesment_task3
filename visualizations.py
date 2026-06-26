"""Visualization module for Academic Predictor analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    """Generate a clean UML diagram showing the real project classes and how they interact."""

    import textwrap

    TITLE_H = 0.65
    SECTION_H = 0.38
    LINE_H = 0.28
    PAD = 0.16
    BOX_EDGE = '#1E1E2E'

    def wrap_rows(items, max_chars=34):
        rows = []
        for item in items:
            wrapped = textwrap.wrap(item, width=max_chars, break_long_words=False, break_on_hyphens=False)
            rows.extend(wrapped or [''])
        return rows

    def estimate_height(attributes, methods):
        attr_rows = wrap_rows(attributes)
        method_rows = wrap_rows(methods)
        return TITLE_H + SECTION_H + (len(attr_rows) * LINE_H) + SECTION_H + (len(method_rows) * LINE_H) + (PAD * 6)

    def draw_class(ax, x, y, width, title, stereotype, attributes, methods, color):
        attr_rows = wrap_rows(attributes)
        method_rows = wrap_rows(methods)

        total_h = estimate_height(attributes, methods)
        title_y = y + total_h - TITLE_H
        attr_section_top = title_y - SECTION_H
        attr_section_bottom = attr_section_top - len(attr_rows) * LINE_H - PAD * 2
        method_section_top = attr_section_bottom - SECTION_H
        method_section_bottom = method_section_top - len(method_rows) * LINE_H - PAD * 2

        ax.add_patch(plt.Rectangle((x, y), width, total_h, facecolor='#FAFAFA', edgecolor=BOX_EDGE, linewidth=2.0, zorder=2))
        ax.add_patch(plt.Rectangle((x, title_y), width, TITLE_H, facecolor=color, edgecolor=BOX_EDGE, linewidth=2.0, zorder=3))

        ax.add_line(plt.Line2D([x, x + width], [attr_section_top, attr_section_top], color='#8C8C8C', linewidth=1.1, zorder=3))
        ax.add_line(plt.Line2D([x, x + width], [method_section_top, method_section_top], color='#8C8C8C', linewidth=1.1, zorder=3))

        ax.text(x + width / 2, title_y + TITLE_H * 0.70, stereotype, ha='center', va='center', fontsize=8.5, color='#EAEAEA', style='italic', zorder=4)
        ax.text(x + width / 2, title_y + TITLE_H * 0.33, title, ha='center', va='center', fontsize=12.5, fontweight='bold', color='white', zorder=4)

        ax.text(x + PAD, attr_section_top - SECTION_H / 2, 'Attributes', ha='left', va='center', fontsize=10, fontweight='bold', color='#333333', zorder=4)
        for i, row in enumerate(attr_rows):
            row_y = attr_section_top - SECTION_H - PAD - i * LINE_H
            ax.text(x + PAD, row_y, row, ha='left', va='top', fontsize=9.0, family='monospace', color='#222222', zorder=4)

        ax.text(x + PAD, method_section_top - SECTION_H / 2, 'Methods', ha='left', va='center', fontsize=10, fontweight='bold', color='#333333', zorder=4)
        for i, row in enumerate(method_rows):
            row_y = method_section_top - SECTION_H - PAD - i * LINE_H
            ax.text(x + PAD, row_y, row, ha='left', va='top', fontsize=9.0, family='monospace', color='#222222', zorder=4)

        return total_h

    def arrow(ax, x1, y1, x2, y2, label='', dashed=False):
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                lw=1.7,
                color='#333333',
                linestyle='dashed' if dashed else 'solid',
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=5,
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha='center', va='center', fontsize=8.5, color='#555555', style='italic', zorder=6)

    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    academic_x, academic_y, academic_w = 0.7, 1.1, 6.0
    mark_x, mark_y, mark_w = 11.1, 5.9, 6.0

    academic_h = draw_class(
        ax,
        academic_x,
        academic_y,
        academic_w,
        title='AcademicPredictor',
        stereotype='«class»',
        attributes=[
            'feature_columns: list[str]',
            'target_column: str',
            'model: Pipeline',
            '  (StandardScaler + LinearRegression)',
        ],
        methods=[
            '__init__(feature_columns, target_column)',
            'load_data(path) -> DataFrame',
            'clean_data(df) -> DataFrame',
            'save_cleaned_data(df, path)',
            'prepare_data(df, test_size, random_state) -> splits',
            'train(X_train, y_train)',
            'evaluate_rmse(X_test, y_test) -> (rmse, predictions)',
            'cross_validate(df, cv) -> float',
            'predict_student(student_row) -> float',
        ],
        color='#A23B72',
    )

    mark_h = draw_class(
        ax,
        mark_x,
        mark_y,
        mark_w,
        title='MarkPredictor',
        stereotype='«class»',
        attributes=[
            'model: LinearRegression',
        ],
        methods=[
            '__init__()',
            'fit(X, y)',
            'predict(X)',
        ],
        color='#2E86AB',
    )

    # Dependency notes showing how the classes actually interact
    dep_x, dep_y, dep_w, dep_h = 6.9, 1.1, 3.6, 2.0
    ax.add_patch(plt.Rectangle((dep_x, dep_y), dep_w, dep_h, facecolor='#F7F7F7', edgecolor='#666666', linewidth=1.4, linestyle='--', zorder=2))
    ax.text(dep_x + dep_w / 2, dep_y + dep_h - 0.25, 'External classes', ha='center', va='top', fontsize=10.5, fontweight='bold', color='#333333')
    ax.text(dep_x + 0.2, dep_y + 1.15, 'Pipeline', ha='left', va='center', fontsize=10, family='monospace', color='#222222')
    ax.text(dep_x + 0.2, dep_y + 0.85, 'StandardScaler', ha='left', va='center', fontsize=10, family='monospace', color='#222222')
    ax.text(dep_x + 0.2, dep_y + 0.55, 'LinearRegression', ha='left', va='center', fontsize=10, family='monospace', color='#222222')

    # Interaction arrows
    arrow(ax, academic_x + academic_w, academic_y + academic_h * 0.72, dep_x, dep_y + dep_h * 0.70, label='uses', dashed=True)
    arrow(ax, mark_x, mark_y + mark_h * 0.70, dep_x + dep_w, dep_y + dep_h * 0.42, label='uses', dashed=True)
    arrow(ax, 3.8, 9.8, academic_x + academic_w / 2, academic_y + academic_h, label='main() / load_and_clean_data()', dashed=False)
    arrow(ax, 14.1, 9.8, mark_x + mark_w / 2, mark_y + mark_h, label='train_level2_ai()', dashed=False)

    ax.text(9.0, 10.55, 'UML Class Diagram — Academic Predictor System', ha='center', va='center', fontsize=16, fontweight='bold', color='#1E1E2E')
    ax.text(9.0, 10.15, 'Attributes and methods are separated into clear compartments for each class', ha='center', va='center', fontsize=10.5, color='#555555')

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ UML class diagram saved to {output_path}')
    return output_path


def create_bias_error_audit_outputs(
    df,
    group_a_filter=None,
    group_b_filter=None,
    features=('Maths_Advanced', 'Physics'),
    target='Software_Engineering_Final',
    error_threshold=5.0,
    chart_path='bias_error_margin_chart.png',
    table_path='bias_error_audit_table.csv',
):
    """Create a fairness chart/table comparing prediction error margins across two groups.

    Disparate impact is computed on the favorable outcome:
    prediction error <= error_threshold.
    """
    if group_a_filter is None:
        group_a_filter = lambda d: d['Physics'] > 70
    if group_b_filter is None:
        group_b_filter = lambda d: d['Modern_History'] > 70

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression()),
    ])

    X = df[list(features)].to_numpy()
    y = df[target].to_numpy()

    # Out-of-fold predictions avoid evaluating on the same rows used for fitting.
    y_pred_oof = cross_val_predict(model, X, y, cv=5)
    abs_error = np.abs(y - y_pred_oof)

    group_a_mask = group_a_filter(df)
    group_b_mask = group_b_filter(df)

    group_stats = []
    for name, mask in [('Group A (Physics > 70)', group_a_mask), ('Group B (Modern_History > 70)', group_b_mask)]:
        group_errors = abs_error[mask.to_numpy()]
        size = int(group_errors.size)
        mae = float(np.mean(group_errors)) if size else 0.0
        rmse = float(np.sqrt(np.mean(group_errors ** 2))) if size else 0.0
        favorable_rate = float(np.mean(group_errors <= error_threshold)) if size else 0.0
        group_stats.append({
            'group': name,
            'n': size,
            'mean_abs_error': mae,
            'rmse_error': rmse,
            'within_threshold_rate': favorable_rate,
        })

    rate_a = group_stats[0]['within_threshold_rate']
    rate_b = group_stats[1]['within_threshold_rate']
    disparate_impact_ratio = (rate_b / rate_a) if rate_a > 0 else 0.0

    summary_row = {
        'group': 'Disparate Impact (B/A)',
        'n': int(len(df)),
        'mean_abs_error': np.nan,
        'rmse_error': np.nan,
        'within_threshold_rate': float(disparate_impact_ratio),
    }

    table_df = pd.DataFrame(group_stats + [summary_row])
    table_df.to_csv(table_path, index=False)

    # Chart: two clear panels (single axis each) to avoid multi-axis confusion.
    labels = [group_stats[0]['group'], group_stats[1]['group']]
    maes = [group_stats[0]['mean_abs_error'], group_stats[1]['mean_abs_error']]
    rates = [group_stats[0]['within_threshold_rate'] * 100, group_stats[1]['within_threshold_rate'] * 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.8))
    x = np.arange(len(labels))

    bars1 = ax1.bar(x, maes, width=0.55, color='#2E86AB')
    ax1.set_title('Error Magnitude by Group', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error (marks)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Group A\n(Physics > 70)', 'Group B\n(Modern_History > 70)'], fontsize=9)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.25)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.06, f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, rates, width=0.55, color='#A23B72')
    ax2.set_title(f'Fairness Outcome: Within ±{error_threshold:.0f} Marks', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Favorable rate (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Group A\n(Physics > 70)', 'Group B\n(Modern_History > 70)'], fontsize=9)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.25)
    for b in bars2:
        h = b.get_height()
        # Place percentage labels above bars with a safe cap inside the axis range.
        y_pos = min(h + 1.2, 108)
        va = 'bottom'
        ax2.text(b.get_x() + b.get_width() / 2, y_pos, f'{h:.1f}%', ha='center', va=va, fontsize=9)

    fig.suptitle('Bias Audit: Group Error Margin Comparison', fontsize=15, fontweight='bold', y=0.97)
    di_text = f'Disparate Impact (B/A) on favorable outcome: {disparate_impact_ratio:.2f}'
    fig.text(
        0.5,
        0.03,
        di_text,
        ha='center',
        va='bottom',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
    )

    plt.subplots_adjust(wspace=0.25, bottom=0.14, top=0.88)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Bias audit chart saved to {chart_path}')
    print(f'✓ Bias audit table saved to {table_path}')
    print(f'  Disparate Impact (B/A): {disparate_impact_ratio:.4f}')

    return {
        'chart_path': chart_path,
        'table_path': table_path,
        'disparate_impact_ratio': float(disparate_impact_ratio),
        'group_stats': group_stats,
        'error_threshold': float(error_threshold),
    }


def create_subject_disparate_impact_chart(
    df,
    features=('Maths_Advanced', 'Physics'),
    target='Software_Engineering_Final',
    subjects=('Maths_Advanced', 'Physics', 'Modern_History'),
    cutoff=70,
    error_threshold=5.0,
    chart_path='slide5_subject_disparate_impact_chart.png',
    table_path='slide5_subject_disparate_impact_table.csv',
):
    """Create a Slide-5-ready chart for subject-wise disparate impact using OOF error margins."""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression()),
    ])

    X = df[list(features)].to_numpy()
    y = df[target].to_numpy()
    y_pred_oof = cross_val_predict(model, X, y, cv=5)
    abs_error = np.abs(y - y_pred_oof)

    rows = []
    for subject in subjects:
        mask = (df[subject] > cutoff).to_numpy()
        group_errors = abs_error[mask]
        rest_errors = abs_error[~mask]

        fav_group = float(np.mean(group_errors <= error_threshold)) if group_errors.size else 0.0
        fav_rest = float(np.mean(rest_errors <= error_threshold)) if rest_errors.size else 0.0
        di_ratio = float(fav_group / fav_rest) if fav_rest > 0 else 0.0

        rows.append({
            'subject_group': f'{subject} > {cutoff}',
            'n_group': int(group_errors.size),
            'n_rest': int(rest_errors.size),
            'mae_group': float(np.mean(group_errors)) if group_errors.size else 0.0,
            'mae_rest': float(np.mean(rest_errors)) if rest_errors.size else 0.0,
            'favorable_rate_group': fav_group,
            'favorable_rate_rest': fav_rest,
            'disparate_impact': di_ratio,
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(table_path, index=False)

    labels = [r['subject_group'].replace('_', '\n') for r in rows]
    di_values = [r['disparate_impact'] for r in rows]
    mae_group = [r['mae_group'] for r in rows]
    mae_rest = [r['mae_rest'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.8))

    target_label = target.replace('_', ' ')

    # Panel 1: Disparate impact ratios by subject group.
    x = np.arange(len(labels))
    colors = ['#2E86AB' if v >= 0.8 else '#D95F5F' for v in di_values]
    bars = ax1.bar(x, di_values, color=colors, width=0.58)
    ax1.axhline(0.8, color='#A23B72', linestyle='--', linewidth=1.6, label='80% fairness threshold')
    ax1.set_title(f'Disparate Impact for Predicting {target_label}', fontsize=12.5, fontweight='bold')
    ax1.set_ylabel('DI Ratio (Group / Rest)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, max(1.5, max(di_values) + 0.2))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.25)
    ax1.legend(loc='upper right', fontsize=9)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.03, f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    # Panel 2: Error margins for context.
    w = 0.34
    x2 = np.arange(len(labels))
    bars_g = ax2.bar(x2 - w / 2, mae_group, width=w, color='#7AA6C2', label='Group MAE')
    bars_r = ax2.bar(x2 + w / 2, mae_rest, width=w, color='#C2D6E2', label='Rest MAE')
    ax2.set_title(f'Error Margin Context for {target_label} (MAE)', fontsize=12.5, fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error (marks)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.25)
    ax2.legend(loc='upper right', fontsize=9)
    for b in list(bars_g) + list(bars_r):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.04, f'{h:.2f}', ha='center', va='bottom', fontsize=8.5)

    fig.suptitle(f'Slide 5: Fairness Audit for {target_label} Predictions', fontsize=15, fontweight='bold', y=0.98)
    fig.text(
        0.5,
        0.02,
        f'Favorable outcome = absolute prediction error <= {error_threshold:.0f} marks (out-of-fold, 5-fold CV)',
        ha='center',
        va='bottom',
        fontsize=10.5,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
    )

    plt.subplots_adjust(wspace=0.28, bottom=0.14, top=0.90)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Subject disparate impact chart saved to {chart_path}')
    print(f'✓ Subject disparate impact table saved to {table_path}')
    return {
        'chart_path': chart_path,
        'table_path': table_path,
        'rows': rows,
        'cutoff': float(cutoff),
        'error_threshold': float(error_threshold),
    }


def create_slide6_verdict_chart(
    baseline_rmse,
    level2_rmse,
    cv_rmse,
    predicted_score,
    disparate_impact_ratio,
    output_path='slide6_final_verdict_chart.png',
):
    """Generate a final-verdict chart for Slide 6.

    Left panel: model error summary (RMSE).
    Right panel: final predicted score with academic thresholds.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.8))

    # Panel 1: RMSE summary
    labels = [
        'Baseline model\nInput: Maths only',
        'Level 2 model\nInput: Maths + Physics',
        '5-Fold CV Mean\n(generalization check)',
    ]
    values = [float(baseline_rmse), float(level2_rmse), float(cv_rmse)]
    colors = ['#7AA6C2', '#2E86AB', '#A23B72']
    bars = ax1.bar(labels, values, color=colors, width=0.62)
    ax1.set_title('RMSE for Predicting Software Engineering Final', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Prediction Error (marks, lower is better)', fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.25)
    ax1.set_ylim(0, max(values) + 2)

    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.08, f'{h:.2f}', ha='center', va='bottom', fontsize=10)

    # Panel 2: Final prediction
    ax2.set_title('Predicted Software Engineering Final (Alex)', fontsize=13, fontweight='bold')
    ax2.barh(['Predicted mark'], [float(predicted_score)], color='#2E86AB', height=0.55)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Software Engineering Final mark out of 100', fontsize=11, fontweight='bold')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.25)

    # Academic thresholds
    ax2.axvline(50, color='#999999', linestyle='--', linewidth=1.2)
    ax2.axvline(75, color='#999999', linestyle='--', linewidth=1.2)
    ax2.text(50, 0.35, 'Pass', ha='center', va='bottom', fontsize=9, color='#555555')
    ax2.text(75, 0.35, 'Distinction', ha='center', va='bottom', fontsize=9, color='#555555')

    ax2.text(
        float(predicted_score) + 1.0,
        0,
        f'{float(predicted_score):.1f}',
        va='center',
        ha='left',
        fontsize=12,
        fontweight='bold',
        color='#1E1E2E',
    )

    # Footer fairness note
    fairness_status = 'PASS' if float(disparate_impact_ratio) >= 0.8 else 'REVIEW'
    fig.text(
        0.5,
        0.02,
        f'Fairness check (Disparate Impact B/A): {float(disparate_impact_ratio):.2f} -> {fairness_status}',
        ha='center',
        va='bottom',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
    )

    fig.suptitle('Final Verdict Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.28, bottom=0.13, top=0.90)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Slide 6 verdict chart saved to {output_path}')
    return output_path


def create_final_mark_generation_chart(
    student_row,
    scaler,
    model,
    feature_cols=('Maths_Advanced', 'Physics'),
    output_path='slide6_mark_generation_chart.png',
):
    """Create a chart showing how a final mark prediction is generated for one student.

    The chart uses a simple contribution view for a linear model:
    prediction = intercept + sum(coef_i * scaled_feature_i)
    """
    if student_row is None:
        raise ValueError('student_row cannot be None')

    feature_cols = list(feature_cols)
    X_student = student_row[feature_cols].to_numpy().reshape(1, -1)
    X_scaled = scaler.transform(X_student)

    intercept = float(model.model.intercept_)
    coefs = model.model.coef_.astype(float)
    contributions = (coefs * X_scaled.flatten()).astype(float)
    predicted = float(model.predict(X_scaled)[0])

    step_labels = ['Intercept'] + [f'{c} contrib' for c in feature_cols]
    step_values = [intercept] + contributions.tolist()
    cumulative = np.cumsum(step_values)

    # Waterfall-like contribution chart (left) + input breakdown table (right).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.8))

    x = np.arange(len(step_values))
    starts = np.r_[0.0, cumulative[:-1]]
    colors = ['#7AA6C2'] + ['#2E86AB' if v >= 0 else '#D95F5F' for v in contributions]

    for i, (start, val, color) in enumerate(zip(starts, step_values, colors)):
        ax1.bar(i, val, bottom=start, color=color, width=0.58, edgecolor='white')
        y_text = start + val + (0.6 if val >= 0 else -0.8)
        va = 'bottom' if val >= 0 else 'top'
        ax1.text(i, y_text, f'{val:+.2f}', ha='center', va=va, fontsize=9)

    # Final predicted bar
    ax1.bar(len(step_values), predicted, color='#A23B72', width=0.62, alpha=0.9)
    ax1.text(len(step_values), predicted + 0.7, f'{predicted:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_xticks(np.arange(len(step_values) + 1))
    ax1.set_xticklabels(step_labels + ['Final Prediction'], rotation=12, ha='right', fontsize=9)
    ax1.set_ylabel('Marks', fontsize=11, fontweight='bold')
    ax1.set_title('How the Final Mark is Generated', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.25)
    ax1.set_ylim(min(0, np.min(cumulative) - 3), max(predicted, np.max(cumulative)) + 5)

    # Right-side explanatory panel
    ax2.axis('off')
    student_name = str(student_row.get('Student_Name', 'Selected Student'))
    raw_vals = [float(student_row[c]) for c in feature_cols]
    scaled_vals = X_scaled.flatten().tolist()

    eq_lines = [
        f'Student: {student_name}',
        '',
        'Linear model equation:',
        'Predicted = Intercept + sum(coef x scaled_feature)',
        '',
        f'Intercept: {intercept:+.3f}',
    ]
    for c, rv, sv, coef, contrib in zip(feature_cols, raw_vals, scaled_vals, coefs, contributions):
        eq_lines.append(f'{c}: raw={rv:.1f}, scaled={sv:+.3f}')
        eq_lines.append(f'  coef={coef:+.3f} -> contrib={contrib:+.3f}')
    eq_lines += [
        '',
        f'Final predicted Software_Engineering_Final = {predicted:.2f}',
    ]

    ax2.text(
        0.02,
        0.98,
        '\n'.join(eq_lines),
        transform=ax2.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='#F7F7F7', edgecolor='#999999', alpha=0.95),
    )

    fig.suptitle('Slide 6: Final Mark Generation Trace', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.12, bottom=0.12, top=0.90)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Final mark generation chart saved to {output_path}')
    print(f'  Predicted score: {predicted:.4f}')
    return output_path, predicted
