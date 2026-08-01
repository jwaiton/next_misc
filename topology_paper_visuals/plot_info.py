import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size


def apply_style(scale_factor=1.0):
    """
    scale_factor: ratio the figure will be scaled by in LaTeX
    (e.g. 1.4 for \\includegraphics[width=1.4\\textwidth]).
    Base sizes are chosen to match \\documentclass[a4paper,11pt]{article}:
    body text = 11pt, so axis labels/tick labels are set relative to
    that, with titles slightly larger.
    """
    base_body = 11  # matches \documentclass[...,11pt]{article}

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": base_body / scale_factor,
        "axes.titlesize": (base_body + 1) / scale_factor,   # slightly larger than body
        "axes.labelsize": base_body / scale_factor,          # match body text
        "xtick.labelsize": (base_body - 1) / scale_factor,   # slightly smaller, standard convention
        "ytick.labelsize": (base_body - 1) / scale_factor,
        "legend.fontsize": (base_body - 1) / scale_factor,
        "lines.linewidth": 1.0 / scale_factor,
        "axes.linewidth": 0.6 / scale_factor,
        "xtick.major.width": 0.6 / scale_factor,
        "ytick.major.width": 0.6 / scale_factor,
        "savefig.dpi": 300,
        "lines.markersize": 3,
        "errorbar.capsize": 1.5,
    })

def fixed_axes_figure(nrows=1, ncols=1, ax_width=2.2, ax_height=2.0,
                       wspace=0.4, hspace=0.4,
                       left=0.7, right=0.15, top=0.15, bottom=0.55):
    """
    Create a figure with one or more axes of a precisely fixed size,
    in inches, regardless of subplot layout or tick/axis label content.

    Unlike `plt.subplots(figsize=...)`, which fixes the *canvas* size
    and lets matplotlib shrink or grow the plot area to accommodate
    labels, this function fixes the *axes* (plot area) size directly
    and derives the canvas size around it. This guarantees that every
    panel produced by this function — across single-panel and
    multi-panel figures alike — has an identical plot area, so
    figures placed side by side in a paper look visually consistent
    (same tick length relative to axes, same font-to-plot-area ratio,
    etc.), which a globally fixed `figsize` cannot guarantee once
    label widths differ between plots.

    Parameters
    ----------
    nrows : int, default 1
        Number of subplot rows.
    ncols : int, default 1
        Number of subplot columns.
    ax_width : float, default 2.2
        Width of a single axes (plot area only, excluding margins
        and labels), in inches. Identical for every panel in the grid.
    ax_height : float, default 2.0
        Height of a single axes (plot area only), in inches.
        Identical for every panel in the grid.
    wspace : float, default 0.4
        Horizontal gap between adjacent columns of axes, in inches.
        Must be wide enough to fit y-axis tick labels of the axes
        to its right, if that axes has its own y-axis labels.
    hspace : float, default 0.4
        Vertical gap between adjacent rows of axes, in inches.
        Must be wide enough to fit x-axis tick labels / axis label
        of the axes above it, if shown.
    left : float, default 0.7
        Margin reserved on the left edge of the figure, in inches.
        Needs to fit the y-axis label and the widest y-tick labels
        across all figures you want to look consistent with this one.
    right : float, default 0.15
        Margin reserved on the right edge of the figure, in inches.
        Usually only needs to fit a bit of padding, unless you're
        adding a colorbar or legend outside the axes.
    top : float, default 0.15
        Margin reserved on the top edge of the figure, in inches.
        Increase if you plan to add a title.
    bottom : float, default 0.55
        Margin reserved on the bottom edge of the figure, in inches.
        Needs to fit the x-axis label and tick labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure, sized automatically so that all axes
        panels are exactly `ax_width` x `ax_height` inches.
    ax : matplotlib.axes.Axes or list or list of lists
        - If nrows == 1 and ncols == 1: a single Axes object.
        - If nrows == 1 and ncols > 1: a list of Axes, left to right.
        - If ncols == 1 and nrows > 1: a list of Axes, top to bottom.
        - Otherwise: a nested list of Axes indexed as axes[row][col],
          with axes[0] being the top row.

    Notes
    -----
    Because margins are specified in inches rather than as figure
    fractions (as `subplots_adjust` uses), the same `left`/`bottom`
    values reserve the *same physical space* regardless of how many
    panels or how large the canvas is. This is what keeps y-axis
    label distance-from-plot-edge, tick length, etc. visually
    identical across figures with different subplot counts.

    Avoid combining this with `savefig(bbox_inches='tight')`, since
    that recrops each figure individually based on its own content
    and will undo the fixed-size guarantee. Use the `left/right/
    top/bottom` margins to control padding instead, and choose them
    generously enough to fit your widest tick labels across *all*
    figures you want to remain consistent with one another.

    Examples
    --------
    Single-panel figure:

     fig, ax = fixed_axes_figure(1, 1, ax_width=2.2, ax_height=2.0)
     ax.plot(x, y)
     fig.savefig("single_panel.pdf")

    Two panels side by side (1 row, 2 columns), each identical in
    size to the single-panel figure above:

     fig, (axA, axB) = fixed_axes_figure(1, 2, ax_width=2.2, ax_height=2.0)
     axA.plot(x, y1)
     axB.plot(x, y2)
     fig.savefig("two_panel.pdf")

    2x2 grid of panels:

     fig, axes = fixed_axes_figure(2, 2, ax_width=2.2, ax_height=2.0)
     axes[0][0].plot(x, y1)   # top-left
     axes[0][1].plot(x, y2)   # top-right
     axes[1][0].plot(x, y3)   # bottom-left
     axes[1][1].plot(x, y4)   # bottom-right
     fig.savefig("grid_panel.pdf")

    Three stacked panels sharing a wide x-axis (3 rows, 1 column),
    with a larger bottom margin only needed on the last panel — note
    this function gives each panel its own bottom margin, so for a
    shared-axis look you may want to hide the tick labels on the
    upper panels manually after creation:

     fig, axes = fixed_axes_figure(3, 1, ax_width=3.0, ax_height=1.2,
                                    hspace=0.15)
     for ax in axes[:-1]:
         ax.tick_params(labelbottom=False)
    """
    fig_width = left + ncols*ax_width + (ncols-1)*wspace + right
    fig_height = bottom + nrows*ax_height + (nrows-1)*hspace + top

    fig = plt.figure(figsize=(fig_width, fig_height))

    h = [Size.Fixed(left)]
    for i in range(ncols):
        h.append(Size.Fixed(ax_width))
        if i < ncols-1:
            h.append(Size.Fixed(wspace))
    h.append(Size.Fixed(right))

    v = [Size.Fixed(bottom)]
    for i in range(nrows):
        v.append(Size.Fixed(ax_height))
        if i < nrows-1:
            v.append(Size.Fixed(hspace))
    v.append(Size.Fixed(top))

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    axes = []
    for row in range(nrows):
        row_axes = []
        for col in range(ncols):
            nx = 1 + 2*col
            ny = len(v) - 2 - 2*row
            ax = fig.add_axes(divider.get_position(),
                               axes_locator=divider.new_locator(nx=nx, ny=ny))
            row_axes.append(ax)
        axes.append(row_axes)

    if nrows == 1 and ncols == 1:
        return fig, axes[0][0]
    elif nrows == 1:
        return fig, axes[0]
    elif ncols == 1:
        return fig, [r[0] for r in axes]
    return fig, axes
