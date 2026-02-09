"""Visualization functions for cloud pipeline output.

Generates publication-quality matplotlib figures from pipeline results
(themes, topics, firms).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_theme_overview(themes_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a horizontal bar chart summarizing cross-firm themes.

    Produces a grouped horizontal bar chart with two bars per theme: one for
    the number of topics (blue) and one for the number of firms (orange),
    sorted by n_topics descending.

    Args:
        themes_df: DataFrame with columns: theme_name, n_topics, n_firms.
        output_dir: Path to directory for saving the chart PNG.
    """
    if themes_df is None or themes_df.empty:
        logger.warning("No theme data to plot")
        return

    logger.info("Creating theme overview chart...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart generation")
        logger.warning("Install with: pip install matplotlib")
        return

    # Sort themes by n_topics descending (plot bottom-to-top so largest is at top)
    df = themes_df.sort_values('n_topics', ascending=True).reset_index(drop=True)

    theme_names = df['theme_name']
    n_topics = df['n_topics']
    n_firms = df['n_firms']

    y_pos = np.arange(len(theme_names))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    bars_topics = ax.barh(
        y_pos + bar_height / 2, n_topics, bar_height,
        label='Topics', color='#1f77b4', edgecolor='white', linewidth=0.5
    )
    bars_firms = ax.barh(
        y_pos - bar_height / 2, n_firms, bar_height,
        label='Firms', color='#ff7f0e', edgecolor='white', linewidth=0.5
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(theme_names, fontsize=10)
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Firm Theme Overview', fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    chart_path = output_dir / 'theme_overview.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved theme overview chart to {chart_path}")


def plot_firm_theme_heatmap(topics_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a heatmap of firm-theme sentence contributions.

    Pivots the input data into a firm (rows) x theme (columns) matrix and
    renders it as an annotated heatmap with a sequential colormap.

    Args:
        topics_df: DataFrame with columns: firm_name, theme_name, n_sentences
            (count of sentences per firm-theme pair).
        output_dir: Path to directory for saving the chart PNG.
    """
    if topics_df is None or topics_df.empty:
        logger.warning("No topic data to plot heatmap")
        return

    logger.info("Creating firm-theme heatmap...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart generation")
        logger.warning("Install with: pip install matplotlib")
        return

    # Pivot into firm x theme matrix, filling missing pairs with 0
    pivot = topics_df.pivot_table(
        index='firm_name', columns='theme_name',
        values='n_sentences', aggfunc='sum', fill_value=0
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.cm.YlOrRd
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')

    # Annotate each cell with its count
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            # Use white text on dark cells, black on light cells
            text_color = 'white' if value > pivot.values.max() * 0.6 else 'black'
            ax.text(j, i, f'{int(value)}', ha='center', va='center',
                    fontsize=9, color=text_color, fontweight='bold')

    # Axis labels and ticks
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=10)

    ax.set_xlabel('Theme', fontsize=12, fontweight='bold')
    ax.set_ylabel('Firm', fontsize=12, fontweight='bold')
    ax.set_title('Firm-Theme Contribution Matrix', fontsize=14, fontweight='bold', pad=15)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Sentence Count', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    chart_path = output_dir / 'firm_theme_heatmap.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved firm-theme heatmap to {chart_path}")


def plot_topic_count_distribution(topics_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a bar chart showing the number of topics discovered per firm.

    Displays a bar for each firm along with a dashed mean line and a text box
    with summary statistics (mean, std, min, max).

    Args:
        topics_df: DataFrame with columns: firm_name, topic_id
            (one row per topic).
        output_dir: Path to directory for saving the chart PNG.
    """
    if topics_df is None or topics_df.empty:
        logger.warning("No topic data to plot distribution")
        return

    logger.info("Creating topic count distribution chart...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed - skipping chart generation")
        logger.warning("Install with: pip install matplotlib")
        return

    # Count topics per firm
    firm_counts = topics_df.groupby('firm_name')['topic_id'].nunique().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = np.arange(len(firm_counts))
    ax.bar(x_pos, firm_counts.values, color='#2ca02c', edgecolor='white', linewidth=0.5)

    # Mean line
    mean_val = firm_counts.mean()
    ax.axhline(y=mean_val, color='#d62728', linestyle='--', linewidth=2, label=f'Mean ({mean_val:.1f})')

    # Axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(firm_counts.index, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Number of Topics', fontsize=12, fontweight='bold')
    ax.set_xlabel('Firm', fontsize=12, fontweight='bold')
    ax.set_title('Topics Discovered per Firm', fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=10)

    # Stats text box
    std_val = firm_counts.std()
    min_val = firm_counts.min()
    max_val = firm_counts.max()

    stats_text = (
        f"Mean: {mean_val:.1f}\n"
        f"Std:  {std_val:.1f}\n"
        f"Min:  {min_val}\n"
        f"Max:  {max_val}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')

    plt.tight_layout()

    chart_path = output_dir / 'topic_count_distribution.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved topic count distribution chart to {chart_path}")
