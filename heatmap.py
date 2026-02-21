"""
CultureSense Resistance Evolution Heatmap
Visualization module for antibiotic resistance patterns over time.
"""

from typing import List, Optional
import base64
import io


def generate_resistance_heatmap(
    resistance_timeline: List[List[str]],
    report_dates: List[str],
) -> Optional[str]:
    """
    Generate a heatmap visualization of resistance evolution across time points.

    Args:
        resistance_timeline: List of resistance marker lists per report
        report_dates: List of ISO date strings (one per report)

    Returns:
        Base64-encoded PNG image string, or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return None

    if not resistance_timeline or not any(resistance_timeline):
        return None

    # Collect all unique resistance markers across all reports
    all_markers = set()
    for markers in resistance_timeline:
        all_markers.update(markers)

    if not all_markers:
        return None

    markers_list = sorted(all_markers)
    num_reports = len(resistance_timeline)

    # Build heatmap matrix: rows=markers, cols=time points
    # Value: 1 if marker present, 0 if absent
    matrix = []
    for marker in markers_list:
        row = [1 if marker in report_markers else 0 for report_markers in resistance_timeline]
        matrix.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(4, num_reports * 1.5), max(3, len(markers_list) * 0.5)))

    # Create heatmap using imshow
    cmap = plt.cm.colors.ListedColormap(["#F5F0EB", "#C1622F"])  # Cream to rust
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Set labels
    ax.set_xticks(range(num_reports))
    ax.set_xticklabels(report_dates, rotation=45, ha="right", fontsize=9)

    ax.set_yticks(range(len(markers_list)))
    ax.set_yticklabels(markers_list, fontsize=10)

    # Add gridlines
    ax.set_xticks([x - 0.5 for x in range(1, num_reports)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(markers_list))], minor=True)
    ax.grid(which="minor", color="#E8DDD6", linestyle="-", linewidth=0.5)

    # Labels and title
    ax.set_xlabel("Report Date", fontsize=10, labelpad=10)
    ax.set_ylabel("Resistance Marker", fontsize=10, labelpad=10)
    ax.set_title("Resistance Evolution Timeline", fontsize=11, fontweight="600", pad=12)

    # Add legend
    present_patch = mpatches.Patch(color="#C1622F", label="Present")
    absent_patch = mpatches.Patch(color="#F5F0EB", label="Absent")
    ax.legend(handles=[present_patch, absent_patch], loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8)

    # Style adjustments
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#E8DDD6")
    ax.spines["left"].set_color("#E8DDD6")

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#FDFAF7")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def get_heatmap_html(img_base64: str) -> str:
    """
    Wrap base64 image in HTML for display.

    Args:
        img_base64: Base64-encoded PNG image string

    Returns:
        HTML img tag with embedded image
    """
    return f'<img src="data:image/png;base64,{img_base64}" alt="Resistance Evolution Heatmap" style="max-width:100%;border:1px solid #E8DDD6;border-radius:4px;margin:12px 0;">'
