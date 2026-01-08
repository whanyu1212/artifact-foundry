"""Utility functions for flashcard diagram generation.

This module provides helpers for creating consistent, educational diagrams
across all flashcards in the Artifact Foundry.

Usage:
    from diagram_generator.utils import setup_flashcard_figure, save_flashcard

    fig, ax = setup_flashcard_figure("My Diagram", (10, 6))
    ax.plot([0, 1], [0, 1])
    save_flashcard(fig, "my_diagram.png")
"""

from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Colorblind-friendly palette (Wong 2011)
ACCESSIBLE_COLORS = {
    'blue': '#0173B2',      # Primary blue
    'orange': '#DE8F05',    # Orange/yellow
    'green': '#029E73',     # Bluish green
    'yellow': '#ECE133',    # Yellow
    'red': '#D55E00',       # Vermillion
    'purple': '#CC78BC',    # Purple
    'cyan': '#56B4E9',      # Sky blue
    'brown': '#CA9161',     # Brown
}

# Standard colors from our existing visualizations
STANDARD_COLORS = {
    'primary_blue': '#3498DB',
    'primary_red': '#E74C3C',
    'primary_green': '#27AE60',
    'primary_orange': '#F39C12',
    'primary_purple': '#9B59B6',
    'secondary_blue': '#2874A6',
    'secondary_red': '#C0392B',
    'secondary_green': '#229954',
    'secondary_orange': '#E67E22',
    'secondary_purple': '#8E44AD',
}


def setup_flashcard_figure(
    title: str,
    figsize: Tuple[int, int] = (10, 6),
    add_grid: bool = True,
    grid_alpha: float = 0.3
) -> Tuple[Figure, Axes]:
    """Create a figure optimized for flashcard display.

    Sets up consistent styling across all flashcard diagrams including
    fonts, grid, and overall appearance.

    Args:
        title: Diagram title (should be descriptive and educational)
        figsize: Figure dimensions in inches (width, height)
            Common sizes:
            - (10, 6): Standard comparison/single plot
            - (14, 6): Side-by-side comparisons
            - (8, 8): Square diagrams (trees, matrices)
            - (10, 10): Stacked vertical comparisons
        add_grid: Whether to add gridlines (recommended for data plots)
        grid_alpha: Transparency of gridlines (0=invisible, 1=opaque)

    Returns:
        Tuple of (figure, axes) ready for plotting

    Example:
        >>> fig, ax = setup_flashcard_figure("Normal Distribution", (10, 6))
        >>> x = np.linspace(-4, 4, 1000)
        >>> ax.plot(x, stats.norm.pdf(x))
        >>> save_flashcard(fig, "normal_distribution.png")
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Set title with consistent styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add grid if requested
    if add_grid:
        ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5)

    return fig, ax


def save_flashcard(
    fig: Figure,
    output_path: str,
    tight: bool = True,
    dpi: int = 300
) -> Path:
    """Save figure in flashcard-optimized format.

    Saves with high DPI for clarity and consistent white background.

    Args:
        fig: Matplotlib figure object to save
        output_path: Where to save the PNG (relative or absolute path)
        tight: Whether to use tight_layout for optimal spacing
        dpi: Dots per inch (300 is publication quality)

    Returns:
        Path object of saved file

    Example:
        >>> fig, ax = setup_flashcard_figure("Example")
        >>> ax.plot([0, 1], [0, 1])
        >>> path = save_flashcard(fig, "example.png")
        >>> print(f"Saved to {path}")
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tight:
        plt.tight_layout()

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)

    return path


def get_color_palette(n_colors: int, accessible: bool = True) -> List[str]:
    """Get a colorblind-friendly color palette.

    Returns colors that are distinguishable for people with various
    types of color vision deficiency.

    Args:
        n_colors: Number of colors needed
        accessible: If True, use Wong colorblind-safe palette;
                   if False, use standard vibrant palette

    Returns:
        List of hex color codes

    Example:
        >>> colors = get_color_palette(3)
        >>> for i, color in enumerate(colors):
        ...     ax.plot(x, y[i], color=color, label=f'Series {i+1}')
    """
    if accessible:
        palette = list(ACCESSIBLE_COLORS.values())
    else:
        palette = list(STANDARD_COLORS.values())[:5]  # First 5 primary colors

    if n_colors <= len(palette):
        return palette[:n_colors]

    # Repeat if more colors needed (with warning)
    import warnings
    warnings.warn(
        f"Requested {n_colors} colors but palette only has {len(palette)}. "
        "Colors will repeat, consider using fewer series for clarity."
    )
    return (palette * (n_colors // len(palette) + 1))[:n_colors]


def annotate_point(
    ax: Axes,
    x: float,
    y: float,
    label: str,
    offset: Tuple[float, float] = (1, 1),
    color: str = '#E74C3C',
    fontsize: int = 11
) -> None:
    """Add an annotated arrow pointing to a specific point.

    Useful for highlighting important features in educational diagrams.

    Args:
        ax: Matplotlib axes to annotate
        x: X-coordinate of point to annotate
        y: Y-coordinate of point to annotate
        label: Text label for annotation
        offset: (dx, dy) offset for text placement
        color: Color for arrow and text
        fontsize: Size of annotation text

    Example:
        >>> fig, ax = setup_flashcard_figure("Example")
        >>> ax.plot(x, y)
        >>> annotate_point(ax, 0.5, 0.25, "Local minimum")
    """
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + offset[0], y + offset[1]),
        arrowprops=dict(arrowstyle='->', lw=2, color=color),
        fontsize=fontsize,
        fontweight='bold',
        color=color,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4)
    )


def add_text_box(
    ax: Axes,
    text: str,
    position: str = 'upper right',
    fontsize: int = 10,
    bgcolor: str = 'wheat',
    alpha: float = 0.5
) -> None:
    """Add a text box with key information to the plot.

    Useful for adding formulas, key insights, or interpretation notes.

    Args:
        ax: Matplotlib axes to add text to
        text: Text content (can include newlines)
        position: One of 'upper right', 'upper left', 'lower right', 'lower left'
        fontsize: Size of text
        bgcolor: Background color of box
        alpha: Transparency of background (0=transparent, 1=opaque)

    Example:
        >>> add_text_box(ax,
        ...     "Key Properties:\\n• Mean = 0\\n• Variance = 1",
        ...     position='upper left')
    """
    position_map = {
        'upper right': (0.98, 0.97),
        'upper left': (0.02, 0.97),
        'lower right': (0.98, 0.03),
        'lower left': (0.02, 0.03),
    }

    x, y = position_map.get(position, (0.98, 0.97))
    ha = 'right' if 'right' in position else 'left'
    va = 'top' if 'upper' in position else 'bottom'

    ax.text(
        x, y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(boxstyle='round', facecolor=bgcolor, alpha=alpha)
    )


def create_side_by_side(
    titles: Tuple[str, str],
    figsize: Tuple[int, int] = (14, 6)
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Create a figure with two side-by-side subplots.

    Useful for comparison diagrams (e.g., "small variance vs large variance").

    Args:
        titles: (left_title, right_title) for the two subplots
        figsize: Overall figure size

    Returns:
        Tuple of (figure, (left_ax, right_ax))

    Example:
        >>> fig, (ax1, ax2) = create_side_by_side(
        ...     ("Small Variance", "Large Variance"),
        ...     figsize=(14, 6)
        ... )
        >>> ax1.plot(x, pdf_small)
        >>> ax2.plot(x, pdf_large)
        >>> save_flashcard(fig, "variance_comparison.png")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.set_title(titles[0], fontsize=13, fontweight='bold', pad=15)
    ax2.set_title(titles[1], fontsize=13, fontweight='bold', pad=15)

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    return fig, (ax1, ax2)


def create_stacked(
    titles: Tuple[str, str],
    figsize: Tuple[int, int] = (10, 10)
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Create a figure with two vertically stacked subplots.

    Useful for showing relationships (e.g., "PDF to CDF").

    Args:
        titles: (top_title, bottom_title) for the two subplots
        figsize: Overall figure size

    Returns:
        Tuple of (figure, (top_ax, bottom_ax))

    Example:
        >>> fig, (ax_top, ax_bottom) = create_stacked(
        ...     ("PDF", "CDF"),
        ...     figsize=(10, 10)
        ... )
        >>> ax_top.plot(x, pdf)
        >>> ax_bottom.plot(x, cdf)
        >>> save_flashcard(fig, "pdf_cdf_relationship.png")
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.set_title(titles[0], fontsize=14, fontweight='bold', pad=15)
    ax2.set_title(titles[1], fontsize=14, fontweight='bold', pad=15)

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    return fig, (ax1, ax2)


def setup_mermaid_template(diagram_type: str) -> str:
    """Return a template for common Mermaid diagram types.

    Provides starting point templates for various Mermaid diagram styles.

    Args:
        diagram_type: One of 'flowchart', 'graph', 'tree', 'sequence', 'class'

    Returns:
        Template string with Mermaid syntax

    Example:
        >>> template = setup_mermaid_template('flowchart')
        >>> print(template)
        >>> # Customize the template for your specific diagram
    """
    templates = {
        'flowchart': '''```mermaid
graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action]
    B -->|No| D[Alternative]
    C --> E[End]
    D --> E

    style A fill:#3498DB,stroke:#2874A6,color:#fff
    style E fill:#27AE60,stroke:#229954,color:#fff
```''',

        'graph': '''```mermaid
graph LR
    A[Node A] -->|relationship| B[Node B]
    B --> C[Node C]
    A --> C

    style A fill:#3498DB,stroke:#2874A6,color:#fff
    style B fill:#E74C3C,stroke:#C0392B,color:#fff
    style C fill:#27AE60,stroke:#229954,color:#fff
```''',

        'tree': '''```mermaid
graph TD
    Root[Root Node] --> Left[Left Child]
    Root --> Right[Right Child]
    Left --> LL[Left-Left]
    Left --> LR[Left-Right]
    Right --> RL[Right-Left]
    Right --> RR[Right-Right]

    style Root fill:#3498DB,stroke:#2874A6,color:#fff
```''',

        'sequence': '''```mermaid
sequenceDiagram
    participant A as Actor A
    participant B as Actor B
    A->>B: Request
    B->>A: Response
```''',

        'class': '''```mermaid
classDiagram
    class ClassName {
        +attribute: type
        +method()
    }
    class OtherClass {
        +data: type
    }
    ClassName --> OtherClass
```''',
    }

    return templates.get(diagram_type, templates['flowchart'])


if __name__ == "__main__":
    # Example usage
    print("Diagram Generator Utils - Example Usage")
    print("=" * 60)

    # Example 1: Simple plot
    import numpy as np

    fig, ax = setup_flashcard_figure("Example Distribution", (10, 6))
    x = np.linspace(-4, 4, 1000)
    y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

    ax.plot(x, y, linewidth=2.5, color=STANDARD_COLORS['primary_blue'])
    ax.fill_between(x, y, alpha=0.3, color=STANDARD_COLORS['primary_blue'])
    ax.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')

    path = save_flashcard(fig, "example_distribution.png")
    print(f"✓ Saved example to: {path}")

    # Example 2: Color palette
    print(f"\n✓ Accessible color palette: {get_color_palette(5)}")

    # Example 3: Mermaid template
    print(f"\n✓ Flowchart template:\n{setup_mermaid_template('flowchart')}")
