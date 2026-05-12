#!/usr/bin/env python3
"""
pk_plots.py — All matplotlib plot functions for PKMagic
========================================================
Each public function in this module draws one complete figure.
They all follow the same signature:

    plot_xxx(fig, pk_data)  ->  None

'fig' is a blank matplotlib Figure handed in by a tab widget.
'pk_data' is the loaded PKData object (see pk_data.py).

Keeping plotting code here — away from the Qt GUI code — means you can
call these functions in a plain Python script or Jupyter notebook without
ever starting a Qt window.  This separation of concerns is a common
and recommended design pattern.

Key Python and library concepts demonstrated here
--------------------------------------------------
- matplotlib Figure / Axes architecture: a Figure is the whole canvas;
  Axes are the individual plot panels within it.
- GridSpec: a flexible way to create subplot grids where panels can span
  multiple rows or columns.
- numpy broadcasting and boolean masking for fast array operations.
- Nested (inner) functions: _hist_by_sex and _trend are defined inside
  their parent function so they can share local variables.
- scipy.stats.linregress: ordinary least-squares linear regression.
- seaborn: a statistical visualisation library built on top of matplotlib
  that handles grouping and colour palettes automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm                  # provides ScalarMappable
from matplotlib import colormaps as mcmaps     # modern colormap registry
import matplotlib.colors as mcolors            # Normalize and colour utilities
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D            # used to build custom legend entries
from scipy import stats                        # linregress for trend lines

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Dictionaries map the numeric sex code (0/1) to a colour hex string and
# a display label.  Using a dictionary means we only need to change the
# colour in one place if we ever want to update it.
SEX_COLORS = {1: "#E221F3", 0: "#1EE92C"}   # blue = male, pink = female
SEX_LABELS = {1: 'Male',    0: 'Female'}
WT_CMAP    = 'viridis'   # colormap name for weight-coloured plots


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def _annotate(ax, text: str, fontsize: int = 7) -> None:
    """
    Place a small text box in the top-right corner of an axes panel.

    transform=ax.transAxes means the coordinates (0.97, 0.97) are in
    'axes fraction' — (0,0) is bottom-left and (1,1) is top-right,
    regardless of the actual data units on the axes.
    """
    ax.text(
        0.97, 0.97, text, transform=ax.transAxes,
        fontsize=fontsize, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.6),
    )


def _sex_color(sex_val) -> str:
    """
    Safely convert a sex code (0 or 1) to a colour string.

    Returns grey if sex_val is None, NaN, or any other unexpected value.
    The try/except catches TypeError (None) and ValueError (bad string).
    """
    try:
        return SEX_COLORS.get(int(sex_val), '#888888')
    except (TypeError, ValueError):
        return '#888888'


# ---------------------------------------------------------------------------
# Tab 0 — Data overview
# ---------------------------------------------------------------------------

def plot_overview(fig: Figure, pk_data) -> None:
    """
    Four-panel overview of the study population.

    Panel layout (2 rows × 2 columns):
        [0,0] Age histogram by sex
        [0,1] Weight histogram by sex
        [1,0] Dose vs weight scatter
        [1,1] Sex counts bar chart
    """
    # fig.subplots() creates a 2×2 grid of Axes and returns them as a
    # 2D NumPy array.  We access individual panels as axes[row, col].
    axes = fig.subplots(2, 2)
    demo = pk_data.demographics   # shorthand reference

    def _hist_by_sex(ax, col, xlabel):
        """
        Draw overlapping histograms — one for each sex — on the same axes.
        alpha=0.65 makes each bar semi-transparent so overlap is visible.
        """
        for sex_val, label in SEX_LABELS.items():
            # Boolean indexing: keep only rows where sex matches sex_val,
            # then select the column of interest and drop NaN values.
            sub = demo[demo['sex'] == sex_val][col].dropna()
            ax.hist(sub, bins=8, alpha=0.65, label=label,
                    color=SEX_COLORS[sex_val], edgecolor='white')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    _hist_by_sex(axes[0, 0], 'age', 'Age (years)')
    axes[0, 0].set_title('Age Distribution')

    _hist_by_sex(axes[0, 1], 'wt', 'Weight (kg)')
    axes[0, 1].set_title('Weight Distribution')

    # Scatter plot: dose vs weight, coloured by sex.
    ax = axes[1, 0]
    for sex_val, label in SEX_LABELS.items():
        sub = demo[demo['sex'] == sex_val]
        ax.scatter(sub['wt'], sub['dose'], label=label,
                   color=SEX_COLORS[sex_val], alpha=0.85, s=60, zorder=3)
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Dose (mg)')
    ax.set_title('Dose vs. Weight')
    ax.legend(fontsize=8)

    # Bar chart of sex counts.
    ax = axes[1, 1]
    counts = demo['sex_label'].value_counts()
    order  = [l for l in ('M', 'F') if l in counts.index]
    bar_colors = [SEX_COLORS[1] if l == 'M' else SEX_COLORS[0] for l in order]
    bars = ax.bar(order, [counts[l] for l in order],
                  color=bar_colors, alpha=0.85, edgecolor='white')
    # Annotate each bar with its count value.
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                str(int(bar.get_height())),
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Count')
    ax.set_title('Sex Distribution')

    n = len(pk_data.patient_ids)
    fig.suptitle(f'Population Demographics Overview  (n = {n})',
                 fontsize=12, fontweight='bold')
    # tight_layout() automatically adjusts spacing so labels don't overlap.
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Tab 1 — Individual concentration–time profiles
# ---------------------------------------------------------------------------

def plot_individual_ct(fig: Figure, pk_data, color_by: str = 'sex') -> None:
    """
    Draw one concentration–time subplot per patient, arranged in a grid.

    'color_by' controls how lines are coloured:
        'sex'    — blue for male, pink for female
        'weight' — continuous viridis colormap scaled to patient weight

    Layout trick: we always reserve a fixed right margin (right=0.88) for
    the colorbar or legend.  This means the subplot grid geometry stays
    exactly the same whether a colorbar is present or not, preventing the
    layout from jumping when the user switches between the two modes.
    """
    pids  = pk_data.patient_ids
    n     = len(pids)
    ncols = 6
    # Ceiling division: how many rows do we need to fit n plots in ncols columns?
    nrows = (n + ncols - 1) // ncols

    # Reserve a fixed right strip for the colorbar / legend.
    # subplots_adjust sets the outer margins of the whole subplot grid
    # as fractions of the figure size (0 = left/bottom edge, 1 = right/top).
    fig.subplots_adjust(left=0.05, right=0.88, top=0.93,
                        bottom=0.05, hspace=0.55, wspace=0.35)
    axes = fig.subplots(nrows, ncols, squeeze=False)

    # .set_index('id') makes the patient ID the row label so we can do
    # demo.loc[pid, 'sex'] instead of a boolean filter each time.
    demo = pk_data.demographics.set_index('id')

    # Build the colormap normaliser once before the loop (only needed for weight mode).
    if color_by == 'weight':
        wt_vals = pk_data.demographics['wt'].dropna().values
        # Normalize maps the weight range onto [0, 1] for the colormap.
        norm = mcolors.Normalize(vmin=wt_vals.min(), vmax=wt_vals.max())
        cmap = plt.colormaps[WT_CMAP]

    for idx, pid in enumerate(pids):
        # divmod(idx, ncols) returns (quotient, remainder), giving us
        # the (row, col) position in the grid.
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        obs = pk_data.get_patient_obs(pid)

        if obs.empty:
            ax.set_visible(False)
            continue

        # Pick the line colour for this patient.
        if color_by == 'weight':
            wt    = demo.loc[pid, 'wt'] if pid in demo.index else np.nan
            color = cmap(norm(float(wt))) if np.isfinite(float(wt)) else '#888888'
        else:
            sex_val = demo.loc[pid, 'sex'] if pid in demo.index else None
            color   = _sex_color(sex_val)

        # 'o-' means draw circles at data points and connect them with a line.
        ax.plot(obs['time'], obs['dv'], 'o-', color=color,
                markersize=3, linewidth=1.2, markerfacecolor=color)
        ax.set_title(f'ID {pid}', fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(pk_data.time_label, fontsize=6)
        ax.set_ylabel('mg/L', fontsize=6)

    # Hide any unused grid positions (the last row may not be full).
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Add colorbar or legend in the reserved right strip.
    # fig.add_axes([left, bottom, width, height]) places axes at an
    # exact position given in figure fractions — it does NOT steal space
    # from the existing subplots, which was the cause of the overlap bug.
    if color_by == 'weight':
        sm = mplcm.ScalarMappable(norm=norm, cmap=plt.colormaps[WT_CMAP])
        sm.set_array([])   # required but unused — tells matplotlib the data range
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
        fig.colorbar(sm, cax=cax, label='Weight (kg)')
    else:
        # Build custom legend entries using Line2D proxy artists.
        legend_els = [
            Line2D([0], [0], color=SEX_COLORS[1], marker='o', markersize=5, label='Male'),
            Line2D([0], [0], color=SEX_COLORS[0], marker='o', markersize=5, label='Female'),
        ]
        fig.legend(handles=legend_els, loc='upper right', fontsize=9,
                   bbox_to_anchor=(0.995, 0.93))

    # x=0.46 centres the title over the subplot grid (not the full figure
    # width which includes the colorbar strip).
    fig.suptitle('Individual Concentration–Time Profiles', fontsize=11,
                 fontweight='bold', x=0.46)


# ---------------------------------------------------------------------------
# Tab 2 — Population concentration–time
# ---------------------------------------------------------------------------

def _build_interp_matrix(pk_data, t_grid: np.ndarray,
                          pid_subset: list[str] | None = None) -> np.ndarray:
    """
    Build a 2-D array (matrix) where:
        rows = patients
        columns = time points on t_grid

    Each row contains the patient's concentration interpolated onto t_grid.
    np.nan is used where the patient has no data (before their first or
    after their last observation), so population statistics can ignore
    those positions.

    This interpolation approach lets us compute a meaningful mean and
    confidence interval across patients even though they were sampled at
    different time points.
    """
    pids = pid_subset if pid_subset is not None else pk_data.patient_ids
    rows = []
    for pid in pids:
        obs = pk_data.get_patient_obs(pid)
        if len(obs) < 2:
            # Not enough points to interpolate — fill row with NaN.
            rows.append(np.full(len(t_grid), np.nan))
            continue

        t  = obs['time'].values.astype(float)
        dv = obs['dv'].values.astype(float)

        # Start with all NaN, then fill in only the range we have data for.
        # Extrapolating outside observed range would introduce false data.
        interp    = np.full(len(t_grid), np.nan)
        in_range  = (t_grid >= t.min()) & (t_grid <= t.max())
        # np.interp does piecewise linear interpolation.
        interp[in_range] = np.interp(t_grid[in_range], t, dv)
        rows.append(interp)

    # np.vstack stacks the 1-D row arrays into a 2-D matrix.
    return np.vstack(rows) if rows else np.full((1, len(t_grid)), np.nan)


def _plot_mean_ci(ax, t_grid, matrix, color='black', label='Mean', min_n=5):
    """
    Draw the population mean curve with a shaded 5th–95th percentile band.

    We only draw at time points where at least min_n patients contribute,
    to avoid misleadingly smooth curves based on very few observations.

    np.errstate(all='ignore') suppresses the 'mean of empty slice' warning
    that numpy would otherwise print when nanmean encounters a column of
    all NaN — we handle that case with the 'mask' check instead.
    """
    # Count how many patients have valid (non-NaN) data at each time point.
    n_valid = np.sum(~np.isnan(matrix), axis=0)
    mask    = n_valid >= min_n
    if not mask.any():
        return   # nothing to draw

    mat_masked = matrix[:, mask]   # select only the well-populated columns
    with np.errstate(all='ignore'):
        mean_c = np.nanmean(mat_masked, axis=0)
        p5     = np.nanpercentile(mat_masked, 5,  axis=0)
        p95    = np.nanpercentile(mat_masked, 95, axis=0)

    ax.plot(t_grid[mask], mean_c, '-', color=color, linewidth=2, label=label)
    # fill_between shades the area between p5 and p95.
    ax.fill_between(t_grid[mask], p5, p95, alpha=0.18, color=color)


def plot_population_ct(fig: Figure, pk_data) -> None:
    """
    Three-panel population overview:
        Left (2/3 width) : all individual curves + population mean ± CI
        Top-right        : mean curves by sex
        Bottom-right     : mean curves by weight quartile

    GridSpec lets us define a grid where panels can span different numbers
    of rows and columns, unlike the simpler fig.subplots() approach.
    """
    # Build a common time grid covering the full observation window.
    t_grid = np.linspace(0, pk_data.obs_df['time'].max() * 1.02, 300)

    # GridSpec(rows, cols) — main panel spans both rows, right panels each span one.
    gs      = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.4)
    ax_main = fig.add_subplot(gs[:, :2])   # gs[:, :2] = all rows, first 2 columns
    ax_sex  = fig.add_subplot(gs[0, 2])    # top-right
    ax_wt   = fig.add_subplot(gs[1, 2])    # bottom-right

    demo = pk_data.demographics.set_index('id')

    # Draw individual patient curves (thin, semi-transparent) coloured by sex.
    for pid in pk_data.patient_ids:
        obs = pk_data.get_patient_obs(pid)
        if obs.empty:
            continue
        sex_val = demo.loc[pid, 'sex'] if pid in demo.index else None
        ax_main.plot(obs['time'], obs['dv'], '-',
                     color=_sex_color(sex_val), alpha=0.18, linewidth=0.8)

    # Population mean ± CI across all patients.
    mat_all = _build_interp_matrix(pk_data, t_grid)
    _plot_mean_ci(ax_main, t_grid, mat_all, color='#212121',
                  label='Pop. mean ± 5–95%ile')

    ax_main.set_xlabel(pk_data.time_label, fontsize=10)
    ax_main.set_ylabel('Concentration (mg/L)', fontsize=10)
    ax_main.set_title('Population Concentration–Time', fontsize=11)
    legend_els = [
        Line2D([0], [0], color=SEX_COLORS[1], alpha=0.6, label='Male (indiv.)'),
        Line2D([0], [0], color=SEX_COLORS[0], alpha=0.6, label='Female (indiv.)'),
        Line2D([0], [0], color='#212121', linewidth=2, label='Pop. mean'),
    ]
    ax_main.legend(handles=legend_els, fontsize=8)

    # --- Sex-stratified means ---
    for sex_val, label in SEX_LABELS.items():
        # Get the IDs for patients of this sex and convert to strings.
        sex_pids = demo[demo['sex'] == sex_val].index.tolist()
        sex_pids = [str(p) for p in sex_pids]
        mat      = _build_interp_matrix(pk_data, t_grid, sex_pids)
        _plot_mean_ci(ax_sex, t_grid, mat, color=SEX_COLORS[sex_val],
                      label=label, min_n=3)
    ax_sex.set_xlabel(pk_data.time_label, fontsize=8)
    ax_sex.set_ylabel('Mean Conc (mg/L)', fontsize=8)
    ax_sex.set_title('By Sex', fontsize=9)
    ax_sex.legend(fontsize=8)
    ax_sex.tick_params(labelsize=7)

    # --- Weight-quartile means ---
    # pd.qcut divides patients into 4 equal-sized groups (quartiles) by weight.
    # Q1 = lightest quarter, Q4 = heaviest quarter.
    wt_series = pk_data.demographics.set_index('id')['wt'].dropna()
    try:
        quartiles = pd.qcut(wt_series, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    except ValueError:
        # qcut fails when there are too few unique values to form 4 groups.
        quartiles = pd.Series(dtype=object)

    # Use a diverging red-yellow-green colormap: lighter patients = red,
    # heavier patients = green.  We divide by 3.0 to spread the 4 values
    # evenly across the [0, 1] colormap range.
    q_cmap = plt.colormaps['RdYlGn']
    for qi, q_label in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        q_pids = quartiles[quartiles == q_label].index.tolist()
        q_pids = [str(p) for p in q_pids]
        if not q_pids:
            continue
        mat = _build_interp_matrix(pk_data, t_grid, q_pids)
        _plot_mean_ci(ax_wt, t_grid, mat, color=q_cmap(qi / 3.0),
                      label=q_label, min_n=2)
    ax_wt.set_xlabel(pk_data.time_label, fontsize=8)
    ax_wt.set_ylabel('Mean Conc (mg/L)', fontsize=8)
    ax_wt.set_title('By Weight Quartile', fontsize=9)
    ax_wt.legend(fontsize=8)
    ax_wt.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Tab 3 — NCA parameter distributions
# ---------------------------------------------------------------------------

def plot_nca_summary(fig: Figure, pk_data) -> None:
    """
    Six histograms showing how each NCA parameter is distributed across
    the population.  A KDE (kernel density estimate) curve is overlaid
    to smooth out histogram bin-edge effects and give a better sense of
    the underlying distribution shape.

    zip(params, axes.flat) pairs each parameter spec with a subplot.
    axes.flat iterates over a 2-D array of Axes in row-major order.
    """
    if not pk_data.nca_results:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No NCA results available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return

    nca_df = pk_data.nca_dataframe()
    axes   = fig.subplots(2, 3)

    # Each tuple: (column name in the NCA dataframe, axis label for the plot).
    params = [
        ('Cmax',    'Cmax (mg/L)'),
        ('Tmax',    f'Tmax ({pk_data.time_label})'),
        ('AUC0-t',  f'AUC₀ₜ (mg/L·{pk_data.time_label})'),
        ('AUC0-inf',f'AUC₀∞ (mg/L·{pk_data.time_label})'),
        ('t_half',  f't½ ({pk_data.time_label})'),
        ('CL',      'CL (mg/L per dose unit)'),
    ]

    for (col, label), ax in zip(params, axes.flat):
        # pd.to_numeric with errors='coerce' turns any None or non-numeric
        # value into NaN, then .dropna() removes those NaN entries.
        data = pd.to_numeric(nca_df[col], errors='coerce').dropna()

        if data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='grey')
            ax.set_title(label, fontsize=9)
            continue

        try:
            # seaborn's histplot draws the histogram and optional KDE in one call.
            sns.histplot(data, ax=ax, kde=True, color='steelblue',
                         alpha=0.6, bins='auto')
        except Exception:
            ax.hist(data, bins=8, color='steelblue', alpha=0.6)

        median = data.median()
        std    = data.std()
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('')
        ax.tick_params(labelsize=7)
        # .3g format: up to 3 significant figures, drops trailing zeros.
        _annotate(ax, f'n={len(data)}\nMedian={median:.3g}\nSD={std:.3g}')

    fig.suptitle('NCA Parameter Distributions', fontsize=12, fontweight='bold')
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Tab 4 — Covariate analysis
# ---------------------------------------------------------------------------

def plot_covariate(fig: Figure, pk_data) -> None:
    """
    Three panels exploring how clearance (CL) relates to patient covariates.

    In pharmacokinetics, 'covariates' are patient characteristics (weight,
    age, sex, genetics, etc.) that explain some of the variability in how
    individuals process a drug.  Identifying covariates that significantly
    affect CL helps clinicians personalise doses.

    Panel layout:
        [0] CL vs weight  — expect positive relationship (heavier = faster CL)
        [1] CL vs age     — typically a negative relationship in elderly
        [2] CL by sex     — box plot with individual data points overlaid
    """
    axes   = fig.subplots(1, 3)
    nca_df = pk_data.nca_dataframe()
    demo   = pk_data.demographics

    # merge() combines two DataFrames on a shared key column, like a SQL JOIN.
    # left_on/right_on specify the key column in each DataFrame.
    merged = nca_df.merge(demo[['id', 'wt', 'age', 'sex', 'sex_label']],
                          left_on='ID', right_on='id', how='inner')
    merged['CL'] = pd.to_numeric(merged['CL'], errors='coerce')
    merged = merged.dropna(subset=['CL'])

    def _trend(ax, x, y, color='red'):
        """
        Fit and draw a linear trend line through scatter data, and annotate
        with R² and p-value.

        np.isfinite() creates a boolean mask that excludes NaN and infinity.
        We use & (bitwise AND) to combine two boolean arrays element-wise.
        """
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return   # not enough points for a meaningful regression

        slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
        xl = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xl, slope * xl + intercept, '--', color=color, linewidth=1.5)
        # R² (r-squared) measures what fraction of the variance in CL is
        # explained by the covariate.  p-value tests whether the slope is
        # significantly different from zero.
        _annotate(ax, f'R²={r**2:.2f}  p={p:.3f}')

    # --- CL vs weight ---
    ax = axes[0]
    for sex_val, label in SEX_LABELS.items():
        sub = merged[merged['sex'] == sex_val]
        ax.scatter(sub['wt'], sub['CL'], color=SEX_COLORS[sex_val],
                   alpha=0.8, s=55, label=label, zorder=3)
    _trend(ax, merged['wt'].values.astype(float), merged['CL'].values.astype(float))
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('CL')
    ax.set_title('CL vs. Weight')
    ax.legend(fontsize=8)

    # --- CL vs age ---
    ax = axes[1]
    for sex_val, label in SEX_LABELS.items():
        sub = merged[merged['sex'] == sex_val]
        ax.scatter(sub['age'], sub['CL'], color=SEX_COLORS[sex_val],
                   alpha=0.8, s=55, label=label, zorder=3)
    _trend(ax, merged['age'].values.astype(float), merged['CL'].values.astype(float),
           color='darkorange')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('CL')
    ax.set_title('CL vs. Age')
    ax.legend(fontsize=8)

    # --- CL by sex — box plot ---
    ax = axes[2]
    try:
        order   = [o for o in ('M', 'F') if o in merged['sex_label'].values]
        palette = {k: v for k, v in [('M', SEX_COLORS[1]), ('F', SEX_COLORS[0])]
                   if k in merged['sex_label'].unique()}
        # seaborn boxplot shows the median, IQR, and outliers.
        # hue='sex_label' is needed for the palette argument in newer seaborn versions.
        sns.boxplot(data=merged, x='sex_label', y='CL', hue='sex_label',
                    ax=ax, palette=palette, order=order, width=0.5, legend=False)
        # stripplot overlays individual data points on top of the box.
        sns.stripplot(data=merged, x='sex_label', y='CL', ax=ax,
                      color='black', size=4, alpha=0.6, order=order)
    except Exception:
        # Fallback to plain matplotlib if seaborn fails.
        groups = [merged[merged['sex_label'] == l]['CL'].dropna().values
                  for l in ('M', 'F')]
        ax.boxplot([g for g in groups if len(g) > 0])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Male', 'Female'])

    # Mann–Whitney U test: a non-parametric test for whether two groups
    # have the same distribution.  It does not assume normality, making
    # it appropriate for small PK datasets.
    male_cl   = merged[merged['sex_label'] == 'M']['CL'].dropna()
    female_cl = merged[merged['sex_label'] == 'F']['CL'].dropna()
    if len(male_cl) >= 2 and len(female_cl) >= 2:
        from scipy.stats import mannwhitneyu
        _, p_val = mannwhitneyu(male_cl, female_cl, alternative='two-sided')
        _annotate(ax, f'MW p={p_val:.3f}')
    ax.set_xlabel('Sex')
    ax.set_ylabel('CL')
    ax.set_title('CL by Sex')

    fig.suptitle('Covariate Analysis', fontsize=12, fontweight='bold')
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Tab 5 — One-compartment model fits
# ---------------------------------------------------------------------------

def plot_compartment(fig: Figure, pk_data) -> None:
    """
    Display the fitted one-compartment model curve alongside the observed
    data for every patient.

    A dashed red line in each panel shows the 'population mean' curve,
    computed using the median V and CL across all patients whose fit
    converged.  This makes it easy to see which patients are faster or
    slower drug eliminators compared to the typical patient.
    """
    pids  = pk_data.patient_ids
    n     = len(pids)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    axes  = fig.subplots(nrows, ncols, squeeze=False)

    # Build a dict for O(1) result lookup by patient ID.
    cpt_lookup = {r.patient_id: r for r in pk_data.compartment_results}
    demo       = pk_data.demographics.set_index('id')

    # Population median parameters (from patients whose fit converged).
    # A list comprehension filters the list in one readable line.
    converged = [r for r in pk_data.compartment_results if r.converged]
    pop_v     = np.median([r.v  for r in converged]) if converged else None
    pop_cl    = np.median([r.cl for r in converged]) if converged else None

    # Dense time grid for smooth model curves (300 points vs ~10 observations).
    t_dense = np.linspace(0, pk_data.obs_df['time'].max() * 1.05, 300)

    for idx, pid in enumerate(pids):
        row, col = divmod(idx, ncols)
        ax   = axes[row][col]
        obs  = pk_data.get_patient_obs(pid)
        dose = pk_data.get_patient_dose(pid)
        cpt  = cpt_lookup.get(pid)

        sex_val = demo.loc[pid, 'sex'] if pid in demo.index else None
        color   = _sex_color(sex_val)

        # Draw observed data as scatter points (zorder=4 puts them on top).
        ax.scatter(obs['time'], obs['dv'], color=color, s=14, zorder=4, alpha=0.9)

        if cpt is not None and cpt.converged and np.isfinite(dose) and dose > 0:
            # Evaluate the fitted model on the dense time grid.
            c_fit = (dose / cpt.v) * np.exp(-(cpt.cl / cpt.v) * t_dense)
            ax.plot(t_dense, c_fit, '-', color=color, linewidth=1.2, zorder=3)
            _annotate(ax, f't½={cpt.t_half:.1f}\nR²={cpt.r_squared:.2f}', fontsize=6)
        else:
            ax.text(0.5, 0.5, 'no fit', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='grey', style='italic')

        # Draw the population mean curve using the median V and CL, scaled
        # to this patient's actual dose so the curves are comparable.
        if pop_v is not None and np.isfinite(dose) and dose > 0:
            c_pop = (dose / pop_v) * np.exp(-(pop_cl / pop_v) * t_dense)
            ax.plot(t_dense, c_pop, 'r--', linewidth=0.7, alpha=0.5, zorder=2)

        ax.set_title(f'ID {pid}', fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(pk_data.time_label, fontsize=5)
        ax.set_ylabel('mg/L', fontsize=5)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Build a figure-level legend with proxy Line2D artists.
    legend_els = [
        Line2D([0], [0], color=SEX_COLORS[1], marker='o', markersize=4,
               label='Male obs/fit'),
        Line2D([0], [0], color=SEX_COLORS[0], marker='o', markersize=4,
               label='Female obs/fit'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1,
               label='Pop. mean'),
    ]
    fig.legend(handles=legend_els, loc='lower right', fontsize=8)
    fig.suptitle('1-Compartment IV Bolus Model Fits', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])
