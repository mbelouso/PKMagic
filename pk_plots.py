from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib import colormaps as mcmaps
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

SEX_COLORS = {1: '#2196F3', 0: '#E91E63'}
SEX_LABELS = {1: 'Male', 0: 'Female'}
WT_CMAP = 'viridis'


def _annotate(ax, text: str, fontsize: int = 7) -> None:
    ax.text(
        0.97, 0.97, text, transform=ax.transAxes,
        fontsize=fontsize, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.6),
    )


def _sex_color(sex_val) -> str:
    try:
        return SEX_COLORS.get(int(sex_val), '#888888')
    except (TypeError, ValueError):
        return '#888888'


# ---------------------------------------------------------------------------
# Tab 0 — Data overview
# ---------------------------------------------------------------------------

def plot_overview(fig: Figure, pk_data) -> None:
    axes = fig.subplots(2, 2)
    demo = pk_data.demographics

    def _hist_by_sex(ax, col, xlabel):
        for sex_val, label in SEX_LABELS.items():
            sub = demo[demo['sex'] == sex_val][col].dropna()
            ax.hist(sub, bins=8, alpha=0.65, label=label, color=SEX_COLORS[sex_val], edgecolor='white')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    _hist_by_sex(axes[0, 0], 'age', 'Age (years)')
    axes[0, 0].set_title('Age Distribution')

    _hist_by_sex(axes[0, 1], 'wt', 'Weight (kg)')
    axes[0, 1].set_title('Weight Distribution')

    ax = axes[1, 0]
    for sex_val, label in SEX_LABELS.items():
        sub = demo[demo['sex'] == sex_val]
        ax.scatter(sub['wt'], sub['dose'], label=label,
                   color=SEX_COLORS[sex_val], alpha=0.85, s=60, zorder=3)
    ax.set_xlabel('Weight (kg)')
    ax.set_ylabel('Dose (mg)')
    ax.set_title('Dose vs. Weight')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    counts = demo['sex_label'].value_counts()
    order = [l for l in ('M', 'F') if l in counts.index]
    bar_colors = [SEX_COLORS[1] if l == 'M' else SEX_COLORS[0] for l in order]
    bars = ax.bar(order, [counts[l] for l in order], color=bar_colors, alpha=0.85, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Count')
    ax.set_title('Sex Distribution')

    n = len(pk_data.patient_ids)
    fig.suptitle(f'Population Demographics Overview  (n = {n})', fontsize=12, fontweight='bold')
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Tab 1 — Individual concentration–time profiles
# ---------------------------------------------------------------------------

def plot_individual_ct(fig: Figure, pk_data, color_by: str = 'sex') -> None:
    pids = pk_data.patient_ids
    n = len(pids)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    axes = fig.subplots(nrows, ncols, squeeze=False)

    demo = pk_data.demographics.set_index('id')

    if color_by == 'weight':
        wt_vals = pk_data.demographics['wt'].dropna().values
        norm = mcolors.Normalize(vmin=wt_vals.min(), vmax=wt_vals.max())
        cmap = plt.colormaps[WT_CMAP]

    for idx, pid in enumerate(pids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        obs = pk_data.get_patient_obs(pid)
        if obs.empty:
            ax.set_visible(False)
            continue

        if color_by == 'weight':
            wt = demo.loc[pid, 'wt'] if pid in demo.index else np.nan
            color = cmap(norm(float(wt))) if np.isfinite(float(wt)) else '#888888'
        else:
            sex_val = demo.loc[pid, 'sex'] if pid in demo.index else None
            color = _sex_color(sex_val)

        ax.plot(obs['time'], obs['dv'], 'o-', color=color,
                markersize=3, linewidth=1.2, markerfacecolor=color)
        ax.set_title(f'ID {pid}', fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(pk_data.time_label, fontsize=6)
        ax.set_ylabel('mg/L', fontsize=6)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if color_by == 'weight':
        sm = mplcm.ScalarMappable(norm=norm, cmap=plt.colormaps[WT_CMAP])
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5,
                     label='Weight (kg)', pad=0.01)
    else:
        legend_els = [
            Line2D([0], [0], color=SEX_COLORS[1], marker='o', markersize=5, label='Male'),
            Line2D([0], [0], color=SEX_COLORS[0], marker='o', markersize=5, label='Female'),
        ]
        fig.legend(handles=legend_els, loc='lower right', fontsize=9)

    fig.suptitle('Individual Concentration–Time Profiles', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])


# ---------------------------------------------------------------------------
# Tab 2 — Population concentration–time
# ---------------------------------------------------------------------------

def _build_interp_matrix(pk_data, t_grid: np.ndarray,
                         pid_subset: list[str] | None = None) -> np.ndarray:
    """
    Returns matrix (n_patients × len(t_grid)) of interpolated concentrations.
    NaN where a patient has no data at a given time point.
    """
    pids = pid_subset if pid_subset is not None else pk_data.patient_ids
    rows = []
    for pid in pids:
        obs = pk_data.get_patient_obs(pid)
        if len(obs) < 2:
            rows.append(np.full(len(t_grid), np.nan))
            continue
        t = obs['time'].values.astype(float)
        dv = obs['dv'].values.astype(float)
        interp = np.full(len(t_grid), np.nan)
        in_range = (t_grid >= t.min()) & (t_grid <= t.max())
        interp[in_range] = np.interp(t_grid[in_range], t, dv)
        rows.append(interp)
    return np.vstack(rows) if rows else np.full((1, len(t_grid)), np.nan)


def _plot_mean_ci(ax, t_grid, matrix, color='black', label='Mean', min_n=5):
    n_valid = np.sum(~np.isnan(matrix), axis=0)
    mask = n_valid >= min_n
    if not mask.any():
        return
    # Only compute stats on columns that have enough valid data
    mat_masked = matrix[:, mask]
    with np.errstate(all='ignore'):
        mean_c = np.nanmean(mat_masked, axis=0)
        p5 = np.nanpercentile(mat_masked, 5, axis=0)
        p95 = np.nanpercentile(mat_masked, 95, axis=0)
    ax.plot(t_grid[mask], mean_c, '-', color=color, linewidth=2, label=label)
    ax.fill_between(t_grid[mask], p5, p95, alpha=0.18, color=color)


def plot_population_ct(fig: Figure, pk_data) -> None:
    t_grid = np.linspace(0, pk_data.obs_df['time'].max() * 1.02, 300)
    gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.4)
    ax_main = fig.add_subplot(gs[:, :2])
    ax_sex = fig.add_subplot(gs[0, 2])
    ax_wt = fig.add_subplot(gs[1, 2])

    demo = pk_data.demographics.set_index('id')

    # Individual curves (thin, coloured by sex)
    for pid in pk_data.patient_ids:
        obs = pk_data.get_patient_obs(pid)
        if obs.empty:
            continue
        sex_val = demo.loc[pid, 'sex'] if pid in demo.index else None
        ax_main.plot(obs['time'], obs['dv'], '-',
                     color=_sex_color(sex_val), alpha=0.18, linewidth=0.8)

    # Population mean ± 95% CI
    mat_all = _build_interp_matrix(pk_data, t_grid)
    _plot_mean_ci(ax_main, t_grid, mat_all, color='#212121', label='Pop. mean ± 5–95%ile')

    ax_main.set_xlabel(pk_data.time_label, fontsize=10)
    ax_main.set_ylabel('Concentration (mg/L)', fontsize=10)
    ax_main.set_title('Population Concentration–Time', fontsize=11)
    legend_els = [
        Line2D([0], [0], color=SEX_COLORS[1], alpha=0.6, label='Male (indiv.)'),
        Line2D([0], [0], color=SEX_COLORS[0], alpha=0.6, label='Female (indiv.)'),
        Line2D([0], [0], color='#212121', linewidth=2, label='Pop. mean'),
    ]
    ax_main.legend(handles=legend_els, fontsize=8)

    # --- Sex-stratified panel ---
    for sex_val, label in SEX_LABELS.items():
        sex_pids = demo[demo['sex'] == sex_val].index.tolist()
        sex_pids = [str(p) for p in sex_pids]
        mat = _build_interp_matrix(pk_data, t_grid, sex_pids)
        _plot_mean_ci(ax_sex, t_grid, mat, color=SEX_COLORS[sex_val], label=label, min_n=3)
    ax_sex.set_xlabel(pk_data.time_label, fontsize=8)
    ax_sex.set_ylabel('Mean Conc (mg/L)', fontsize=8)
    ax_sex.set_title('By Sex', fontsize=9)
    ax_sex.legend(fontsize=8)
    ax_sex.tick_params(labelsize=7)

    # --- Weight-quartile panel ---
    wt_series = pk_data.demographics.set_index('id')['wt'].dropna()
    try:
        quartiles = pd.qcut(wt_series, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    except ValueError:
        quartiles = pd.Series(dtype=object)

    q_cmap = plt.colormaps['RdYlGn']
    for qi, q_label in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        q_pids = quartiles[quartiles == q_label].index.tolist()
        q_pids = [str(p) for p in q_pids]
        if not q_pids:
            continue
        mat = _build_interp_matrix(pk_data, t_grid, q_pids)
        _plot_mean_ci(ax_wt, t_grid, mat, color=q_cmap(qi / 3.0), label=q_label, min_n=2)
    ax_wt.set_xlabel(pk_data.time_label, fontsize=8)
    ax_wt.set_ylabel('Mean Conc (mg/L)', fontsize=8)
    ax_wt.set_title('By Weight Quartile', fontsize=9)
    ax_wt.legend(fontsize=8)
    ax_wt.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Tab 3 — NCA summary
# ---------------------------------------------------------------------------

def plot_nca_summary(fig: Figure, pk_data) -> None:
    if not pk_data.nca_results:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No NCA results available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return

    nca_df = pk_data.nca_dataframe()
    axes = fig.subplots(2, 3)

    params = [
        ('Cmax', 'Cmax (mg/L)'),
        ('Tmax', f'Tmax ({pk_data.time_label})'),
        ('AUC0-t', f'AUC₀ₜ (mg/L·{pk_data.time_label})'),
        ('AUC0-inf', f'AUC₀∞ (mg/L·{pk_data.time_label})'),
        ('t_half', f't½ ({pk_data.time_label})'),
        ('CL', 'CL (mg/L per dose unit)'),
    ]

    for (col, label), ax in zip(params, axes.flat):
        data = pd.to_numeric(nca_df[col], errors='coerce').dropna()
        if data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='grey')
            ax.set_title(label, fontsize=9)
            continue
        try:
            sns.histplot(data, ax=ax, kde=True, color='steelblue', alpha=0.6, bins='auto')
        except Exception:
            ax.hist(data, bins=8, color='steelblue', alpha=0.6)
        median = data.median()
        std = data.std()
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('')
        ax.tick_params(labelsize=7)
        _annotate(ax, f'n={len(data)}\nMedian={median:.3g}\nSD={std:.3g}')

    fig.suptitle('NCA Parameter Distributions', fontsize=12, fontweight='bold')
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Tab 4 — Covariate analysis
# ---------------------------------------------------------------------------

def plot_covariate(fig: Figure, pk_data) -> None:
    axes = fig.subplots(1, 3)
    nca_df = pk_data.nca_dataframe()
    demo = pk_data.demographics

    merged = nca_df.merge(demo[['id', 'wt', 'age', 'sex', 'sex_label']],
                          left_on='ID', right_on='id', how='inner')
    merged['CL'] = pd.to_numeric(merged['CL'], errors='coerce')
    merged = merged.dropna(subset=['CL'])

    def _trend(ax, x, y, color='red'):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return
        slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
        xl = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xl, slope * xl + intercept, '--', color=color, linewidth=1.5)
        _annotate(ax, f'R²={r**2:.2f}  p={p:.3f}')

    # CL vs weight
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

    # CL vs age
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

    # CL by sex
    ax = axes[2]
    try:
        order = [o for o in ('M', 'F') if o in merged['sex_label'].values]
        palette = {k: v for k, v in [('M', SEX_COLORS[1]), ('F', SEX_COLORS[0])]
                   if k in merged['sex_label'].unique()}
        sns.boxplot(data=merged, x='sex_label', y='CL', hue='sex_label',
                    ax=ax, palette=palette, order=order, width=0.5, legend=False)
        sns.stripplot(data=merged, x='sex_label', y='CL', ax=ax,
                      color='black', size=4, alpha=0.6, order=order)
    except Exception:
        groups = [merged[merged['sex_label'] == l]['CL'].dropna().values for l in ('M', 'F')]
        ax.boxplot([g for g in groups if len(g) > 0])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Male', 'Female'])

    # Mann–Whitney U test
    male_cl = merged[merged['sex_label'] == 'M']['CL'].dropna()
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
# Tab 5 — 1-compartment model fits
# ---------------------------------------------------------------------------

def plot_compartment(fig: Figure, pk_data) -> None:
    pids = pk_data.patient_ids
    n = len(pids)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    axes = fig.subplots(nrows, ncols, squeeze=False)

    cpt_lookup = {r.patient_id: r for r in pk_data.compartment_results}
    demo = pk_data.demographics.set_index('id')

    # Population median parameters from converged fits
    converged = [r for r in pk_data.compartment_results if r.converged]
    pop_v = np.median([r.v for r in converged]) if converged else None
    pop_cl = np.median([r.cl for r in converged]) if converged else None

    t_dense = np.linspace(0, pk_data.obs_df['time'].max() * 1.05, 300)

    for idx, pid in enumerate(pids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        obs = pk_data.get_patient_obs(pid)
        dose = pk_data.get_patient_dose(pid)
        cpt = cpt_lookup.get(pid)
        sex_val = demo.loc[pid, 'sex'] if pid in demo.index else None
        color = _sex_color(sex_val)

        ax.scatter(obs['time'], obs['dv'], color=color, s=14, zorder=4, alpha=0.9)

        if cpt is not None and cpt.converged and np.isfinite(dose) and dose > 0:
            c_fit = (dose / cpt.v) * np.exp(-(cpt.cl / cpt.v) * t_dense)
            ax.plot(t_dense, c_fit, '-', color=color, linewidth=1.2, zorder=3)
            _annotate(ax, f't½={cpt.t_half:.1f}\nR²={cpt.r_squared:.2f}', fontsize=6)
        else:
            ax.text(0.5, 0.5, 'no fit', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7, color='grey', style='italic')

        # Population mean curve overlay
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

    legend_els = [
        Line2D([0], [0], color=SEX_COLORS[1], marker='o', markersize=4, label='Male obs/fit'),
        Line2D([0], [0], color=SEX_COLORS[0], marker='o', markersize=4, label='Female obs/fit'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='Pop. mean'),
    ]
    fig.legend(handles=legend_els, loc='lower right', fontsize=8)
    fig.suptitle('1-Compartment IV Bolus Model Fits', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])
