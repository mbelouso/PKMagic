#!/usr/bin/env python3
"""
pk_data.py — Data model for PKMagic
====================================
This module is responsible for everything that touches the raw data:
  1. Loading and cleaning a NONMEM-style CSV file
  2. Splitting rows into dose events and concentration observations
  3. Running Non-Compartmental Analysis (NCA) per patient
  4. Fitting a one-compartment pharmacokinetic model per patient

The results are stored in a single PKData object that every plot and
table in the GUI can read from.

Key Python concepts demonstrated here
--------------------------------------
- @dataclass   : a decorator that auto-generates __init__, __repr__, etc.
- Type hints   : e.g. `pid: str` — they don't change how the code runs
                 but make it easier to understand what each variable holds
- try/except   : graceful error handling so one bad patient doesn't crash
                 the whole analysis
- List comprehensions : compact syntax for building lists, e.g.
                 [str(int(x)) for x in values]
- np.nan       : NumPy's "Not a Number" — used for any missing/invalid
                 numeric value so arithmetic on it propagates gracefully
"""

from __future__ import annotations  # allows type hints to reference the
                                     # class being defined (forward refs)

from dataclasses import dataclass   # gives us @dataclass
from typing import Optional          # Optional[X] means the value is X or None

import numpy as np                   # numerical arrays and maths
import pandas as pd                  # tabular data (DataFrames)
from scipy import stats              # statistical functions (linear regression)
from scipy.integrate import trapezoid as _trapz   # area under a curve
from scipy.optimize import curve_fit              # non-linear least-squares fitting


# ---------------------------------------------------------------------------
# Column name mapping
# ---------------------------------------------------------------------------
# NONMEM files can use different column headers across studies.
# This dictionary maps any recognised raw header to our internal short name.
# If a column is not in COLUMN_MAP it is left unchanged.
COLUMN_MAP = {
    '#ID':            'id',
    'PatientID':      'id',
    'time(min)':      'time',
    'wt(kg)':         'wt',
    'dv(mg/L)':       'dv',
    'dependtvariable':'dvid',
}

# After renaming, every one of these internal names must be present.
# Using a Python set means we can use set-difference ( - ) to find what's missing.
REQUIRED_INTERNAL = {'id', 'time', 'wt', 'age', 'sex', 'amt', 'rate', 'dvid', 'dv', 'mdv'}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
# A @dataclass is just a regular class where Python writes the __init__
# method for you, based on the field names and type hints you declare.
# Think of each field as a column in a results table.

@dataclass
class NCAResult:
    """
    Holds the Non-Compartmental Analysis (NCA) output for one patient.

    NCA derives PK parameters directly from the observed data — no model
    assumptions needed.  Key parameters:

    Cmax      : peak concentration observed (mg/L)
    Tmax      : time of peak concentration
    AUC0-t    : area under the concentration–time curve from 0 to last
                measured time point — total drug exposure (mg/L · time)
    lambda_z  : terminal elimination rate constant (1/time).  Found by
                fitting a straight line to log(C) vs time in the terminal
                (declining) phase.  Slope = -lambda_z.
    t_half    : elimination half-life = ln(2) / lambda_z
    AUC0-inf  : AUC extrapolated to infinity = AUC0-t + Clast / lambda_z
    CL        : apparent clearance = Dose / AUC0-inf  (volume per time)
    Vd        : apparent volume of distribution = CL / lambda_z
    r2_lambdaz: R² of the log-linear regression used to estimate lambda_z
                (closer to 1.0 = better fit to the terminal phase)
    cmax_is_observed: False when the first sample was taken so late that
                the true peak was likely missed (e.g. first obs at 24 h)
    """
    patient_id: str
    dose: float
    n_obs: int
    cmax: float
    tmax: float
    auc0t: float
    lambda_z: float
    t_half: float
    auc0inf: float
    cl: float
    vd: float
    r2_lambdaz: float
    cmax_is_observed: bool


@dataclass
class CompartmentResult:
    """
    Holds the one-compartment model fit results for one patient.

    The model assumes the body is a single well-mixed compartment.
    After an IV bolus dose D the concentration decays exponentially:

        C(t) = (D / V) * exp(-(CL/V) * t)

    Parameters fitted:
    V    : volume of distribution (L)
    CL   : clearance (L / time)
    k    : elimination rate constant = CL / V  (1/time)
    t_half : half-life = ln(2) / k
    r_squared : how well the model curve fits the observed data
    converged : True if scipy's curve_fit found a solution
    """
    patient_id: str
    dose: float
    v: float
    cl: float
    k: float
    t_half: float
    r_squared: float
    converged: bool


# ---------------------------------------------------------------------------
# Main data class
# ---------------------------------------------------------------------------

class PKData:
    """
    Central data store for one loaded CSV file.

    Call .load(filepath) to populate all attributes.  The GUI reads
    from this object; it never writes back to it.

    Attributes (all populated after load())
    ----------------------------------------
    raw_df      : the full cleaned DataFrame (all rows)
    obs_df      : only valid concentration observations (mdv == 0)
    dose_df     : only dose-event rows (amt is not NaN)
    patient_ids : sorted list of patient ID strings, e.g. ['0','1',...]
    demographics: one row per patient — weight, age, sex, dose, etc.
    nca_results : list of NCAResult, one per patient
    compartment_results : list of CompartmentResult, one per patient
    warnings    : list of human-readable warnings produced during loading
    time_label  : the original time column header (e.g. 'time(min)')
                  used for axis labels so plots stay accurate
    """

    def __init__(self):
        # Initialise all attributes to empty/default values so the object
        # is always in a valid state, even before load() is called.
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.obs_df: pd.DataFrame = pd.DataFrame()
        self.dose_df: pd.DataFrame = pd.DataFrame()
        self.patient_ids: list[str] = []
        self.demographics: pd.DataFrame = pd.DataFrame()
        self.nca_results: list[NCAResult] = []
        self.compartment_results: list[CompartmentResult] = []
        self.warnings: list[str] = []
        self.time_label: str = 'time'

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def load(self, filepath: str) -> None:
        """
        Read a CSV file and run the full analysis pipeline.

        Steps performed in order:
            1. Pre-filter: skip any body line that starts with '#'
               (those are commented-out / excluded data points)
            2. Parse into a pandas DataFrame
            3. Clean column names and cast to correct types
            4. Split into dose_df and obs_df
            5. Build a per-patient demographics table
            6. Run NCA for every patient
            7. Fit the one-compartment model for every patient
        """
        self.warnings = []

        # --- Step 1: read the file and drop commented rows ---
        # The 'with' statement opens the file and automatically closes
        # it when the block finishes, even if an error occurs.
        import io
        with open(filepath, 'r') as f:
            lines = f.readlines()   # list of strings, one per line

        # The very first line is the column header — keep it always.
        # Body lines (lines[1:]) that start with '#' are excluded data.
        header = lines[0]
        body = [l for l in lines[1:] if not l.lstrip().startswith('#')]

        n_excluded = len(lines) - 1 - len(body)
        if n_excluded:
            self.warnings.append(
                f"Excluded {n_excluded} commented-out data row(s) from analysis"
            )

        # io.StringIO turns a string into a file-like object so
        # pd.read_csv can parse it without writing a temporary file.
        # dtype=str reads every cell as text so we control type conversion.
        df = pd.read_csv(io.StringIO(header + ''.join(body)), dtype=str)

        # --- Steps 2–7 ---
        df = self._clean(df)
        self.raw_df = df
        self._split_dose_obs()
        self._build_demographics()
        self.compute_nca()
        self.fit_one_compartment()

    # ------------------------------------------------------------------
    # Private helpers — data cleaning
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise a raw DataFrame so downstream code sees consistent names
        and types regardless of the exact CSV format used.
        """
        # Remove accidental leading/trailing spaces from column headers.
        df.columns = [c.strip() for c in df.columns]

        # Remove spaces from every cell in string (object-dtype) columns.
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()

        # Rename known column variants to our internal short names.
        # The dict comprehension only renames columns that actually exist
        # in this file — unknown columns are left as-is.
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

        # Remember the original time column header (e.g. 'time(min)' or
        # 'time(h)') so we can use it as an axis label on every plot.
        orig_time_cols = [c for c in COLUMN_MAP if 'time' in c.lower()]
        if orig_time_cols:
            self.time_label = orig_time_cols[0]

        # Raise a clear error if any required column is missing.
        # Set difference ( - ) returns elements in the first set that
        # are not in the second.
        missing = REQUIRED_INTERNAL - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns after rename: {missing}")

        # NONMEM uses '.' to mean "not applicable / missing".
        # Replace those with np.nan so pandas treats them as missing numbers.
        for col in ['amt', 'rate', 'dv']:
            df[col] = df[col].replace('.', np.nan)

        # Convert columns to numeric types.
        # errors='coerce' turns anything that cannot be converted into NaN
        # instead of raising an exception.
        for col in ['time', 'wt', 'age', 'sex', 'mdv', 'dvid']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['amt', 'rate', 'dv']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Patient IDs: cast to float first for numeric sort order, then
        # drop any row where the ID could not be parsed.
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df = df.dropna(subset=['id'])

        return df

    def _split_dose_obs(self) -> None:
        """
        Separate the single flat DataFrame into two specialised views:

        dose_df — one row per patient containing the administered dose.
                  In NONMEM format these rows have a non-NaN 'amt' value
                  and mdv=1 (dependent variable is "missing" because we
                  haven't measured a concentration at dose time).

        obs_df  — all valid concentration measurements (mdv == 0, dv not
                  NaN).  If a patient has duplicate observations at the
                  same time point, we average them — pharmacologically
                  these represent replicate samples taken from the same
                  blood draw.
        """
        df = self.raw_df

        # Boolean indexing: keep only rows where 'amt' is not NaN.
        # .copy() creates an independent DataFrame so later changes don't
        # accidentally affect raw_df.
        self.dose_df = df[df['amt'].notna()].copy()

        # Keep rows where the dependent variable is valid.
        # mdv == 0 means "this row contains a real observation".
        obs = df[(df['mdv'] == 0) & (df['dv'].notna())].copy()

        # Handle duplicate (patient, time) pairs by averaging their dv.
        # groupby groups rows that share the same (id, time) values;
        # agg then applies a function to each group — 'mean' for dv,
        # 'first' for covariates (they are the same within a patient).
        n_before = len(obs)
        agg_dict = {
            'dv':   'mean',
            'wt':   'first',
            'age':  'first',
            'sex':  'first',
            'dvid': 'first',
        }
        obs = obs.groupby(['id', 'time'], as_index=False).agg(agg_dict)
        n_dup = n_before - len(obs)
        if n_dup > 0:
            self.warnings.append(
                f"Averaged {n_dup} duplicate (id, time) observation(s) before analysis"
            )

        # Sort by patient then time so every analysis sees data in
        # chronological order. reset_index gives a clean 0-based index.
        self.obs_df = obs.sort_values(['id', 'time']).reset_index(drop=True)

        # Build a list of patient IDs sorted numerically as integers,
        # then stored as strings so they work as dict keys and labels.
        all_ids = sorted(df['id'].dropna().unique())
        self.patient_ids = [str(int(pid)) for pid in all_ids]

    def _build_demographics(self) -> None:
        """
        Collect one row of summary information per patient:
        weight, age, sex, dose, number of observations, and the first
        and last observation times.

        We build a list of dicts and then convert it to a DataFrame at
        the end — this pattern is more efficient than appending rows to
        a DataFrame one at a time inside a loop.
        """
        rows = []
        for pid in self.patient_ids:
            pid_f = float(pid)   # float for DataFrame boolean comparisons

            # Retrieve this patient's observations and dose row.
            obs      = self.obs_df[self.obs_df['id'] == pid_f]
            dose_row = self.dose_df[self.dose_df['id'] == pid_f]

            # .iloc[0] gets the first element of a Series.
            # We guard with 'if not <df>.empty' to avoid IndexError when
            # a patient has no data.
            dose    = float(dose_row['amt'].iloc[0]) if not dose_row.empty else np.nan
            wt      = float(obs['wt'].iloc[0])       if not obs.empty else np.nan
            age     = float(obs['age'].iloc[0])      if not obs.empty else np.nan
            sex_raw = float(obs['sex'].iloc[0])      if not obs.empty else np.nan

            # np.isfinite() returns False for NaN and infinity, so this
            # safely converts a valid sex code (0 or 1) to int, or None.
            sex_int = int(sex_raw) if np.isfinite(sex_raw) else None

            rows.append({
                'id':         pid,
                'sex':        sex_int,
                'sex_label':  ('M' if sex_int == 1 else 'F') if sex_int is not None else '?',
                'wt':         wt,
                'age':        age,
                'dose':       dose,
                'n_obs':      len(obs),
                'first_time': float(obs['time'].min()) if not obs.empty else np.nan,
                'last_time':  float(obs['time'].max()) if not obs.empty else np.nan,
            })

        self.demographics = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_patient_obs(self, pid: str) -> pd.DataFrame:
        """Return all concentration observations for one patient, sorted by time."""
        return self.obs_df[self.obs_df['id'] == float(pid)].copy()

    def get_patient_dose(self, pid: str) -> float:
        """Return the administered dose (mg) for one patient, or NaN if not found."""
        row = self.dose_df[self.dose_df['id'] == float(pid)]
        return float(row['amt'].iloc[0]) if not row.empty else np.nan

    # ------------------------------------------------------------------
    # Non-Compartmental Analysis (NCA)
    # ------------------------------------------------------------------

    def compute_nca(self) -> None:
        """
        Run NCA for every patient and collect results in self.nca_results.

        We wrap each patient's analysis in try/except so that if one
        patient's data is malformed the rest still get analysed.
        """
        self.nca_results = []
        for pid in self.patient_ids:
            try:
                self.nca_results.append(self._nca_patient(pid))
            except Exception as exc:
                self.warnings.append(f"NCA failed for patient {pid}: {exc}")

    def _nca_patient(self, pid: str) -> NCAResult:
        """
        Calculate NCA parameters for a single patient.

        The core steps are:
        1. Cmax and Tmax  — simple max of the observed concentrations.
        2. AUC0-t         — trapezoidal integration: the area under the
                            curve between observed time points.
        3. lambda_z       — terminal elimination rate constant.  We fit a
                            straight line to log(C) vs time in the terminal
                            declining phase.  Because C(t) ∝ exp(-lambda_z·t),
                            log(C) decreases linearly with slope -lambda_z.
        4. t_half         — ln(2) / lambda_z  (time for concentration to
                            halve, independent of the starting level).
        5. AUC0-inf       — extrapolate beyond the last observation:
                            AUC0-t + Clast / lambda_z
        6. CL, Vd         — derived from dose and AUC0-inf.
        """
        obs  = self.get_patient_obs(pid)
        dose = self.get_patient_dose(pid)

        # Convert pandas Series to plain NumPy arrays for fast maths.
        # .astype(float) ensures no integer or object dtype surprises.
        time = obs['time'].values.astype(float)
        dv   = obs['dv'].values.astype(float)
        n_obs = len(obs)

        # --- Step 1: Cmax and Tmax ---
        cmax = float(np.max(dv))
        tmax = float(time[np.argmax(dv)])   # argmax returns the index of the max

        # --- Step 2: AUC0-t by trapezoidal rule ---
        # Each trapezoid has area = (C1+C2)/2 * (t2-t1).
        # scipy's trapezoid sums these for us.
        auc0t = float(_trapz(dv, time))

        # Flag patients whose first observation is so late (≥ 24 time
        # units) that the true peak concentration was probably missed.
        cmax_is_observed = float(time[0]) < 24.0

        # --- Step 3: terminal lambda_z ---
        # We can only take log of positive concentrations, so filter first.
        pos_mask = dv > 0
        pos_time = time[pos_mask]
        pos_dv   = dv[pos_mask]

        # Initialise all terminal-phase results to NaN.
        # They remain NaN if we cannot compute a valid lambda_z.
        lambda_z   = np.nan
        t_half     = np.nan
        auc0inf    = np.nan
        cl         = np.nan
        vd         = np.nan
        r2_lambdaz = np.nan

        if len(pos_time) >= 3:
            # Adaptive selection: start with the last 3 positive-dv points
            # and extend backward one step at a time.  Accept the extension
            # if R² does not drop by more than 0.01 (i.e. the new point
            # still lies on the same log-linear line).
            best_r2 = -np.inf
            best_lz = np.nan

            for n_pts in range(3, len(pos_time) + 1):
                # pos_time[-n_pts:] slices the last n_pts elements
                t_seg    = pos_time[-n_pts:]
                lndv_seg = np.log(pos_dv[-n_pts:])   # natural logarithm

                # Fit a straight line: log(C) = intercept + slope * t
                # slope will be negative for a declining concentration.
                slope, _, r, _, _ = stats.linregress(t_seg, lndv_seg)
                lz = -slope        # lambda_z must be positive
                r2 = r ** 2        # coefficient of determination

                if lz <= 0:
                    break  # concentration is not declining — stop here

                if r2 >= best_r2 - 0.01:
                    best_r2 = r2
                    best_lz = lz
                else:
                    break  # adding this point degraded the fit — stop

            if np.isfinite(best_lz) and best_lz > 0:
                lambda_z   = best_lz
                r2_lambdaz = best_r2
                t_half     = np.log(2.0) / lambda_z

                # Clast is the last quantifiable (positive) concentration.
                # The tail beyond the last observation is extrapolated as:
                # AUC(tlast→∞) = Clast / lambda_z  (integral of exp decay)
                clast   = float(pos_dv[-1])
                auc0inf = auc0t + clast / lambda_z

                if np.isfinite(dose) and auc0inf > 0:
                    cl = dose / auc0inf          # clearance
                    vd = cl / lambda_z           # volume of distribution

        # Return a populated NCAResult dataclass.
        # Named arguments make it clear which value goes where.
        return NCAResult(
            patient_id       = pid,
            dose             = dose,
            n_obs            = n_obs,
            cmax             = cmax,
            tmax             = tmax,
            auc0t            = auc0t,
            lambda_z         = lambda_z,
            t_half           = t_half,
            auc0inf          = auc0inf,
            cl               = cl,
            vd               = vd,
            r2_lambdaz       = r2_lambdaz,
            cmax_is_observed = cmax_is_observed,
        )

    # ------------------------------------------------------------------
    # One-compartment model fitting
    # ------------------------------------------------------------------

    def fit_one_compartment(self) -> None:
        """
        Fit a one-compartment IV-bolus model to every patient's data.

        We build a lookup dictionary from NCA results first so we can
        use each patient's NCA estimates as starting values (initial
        guesses) for the curve fitter — this greatly improves convergence.

        A dictionary comprehension  { key: value  for item in iterable }
        is used to build the lookup in one readable line.
        """
        self.compartment_results = []
        nca_lookup = {r.patient_id: r for r in self.nca_results}

        for pid in self.patient_ids:
            try:
                nca = nca_lookup.get(pid)   # None if NCA failed for this patient
                self.compartment_results.append(self._fit_patient(pid, nca))
            except Exception as exc:
                self.warnings.append(f"1-cpt fit failed for patient {pid}: {exc}")

    def _fit_patient(self, pid: str, nca: Optional[NCAResult]) -> CompartmentResult:
        """
        Fit C(t) = (Dose/V) * exp(-(CL/V) * t) to one patient's data
        using scipy's curve_fit (non-linear least squares).

        curve_fit adjusts V and CL until the sum of squared differences
        between the model predictions and the observed concentrations is
        minimised.

        Initial guesses (p0) come from the NCA results when available —
        a good starting point helps the optimiser converge to the right
        solution.  Bounds prevent physiologically impossible values.
        """
        obs  = self.get_patient_obs(pid)
        dose = self.get_patient_dose(pid)
        time = obs['time'].values.astype(float)
        dv   = obs['dv'].values.astype(float)

        # Cannot fit a model without a valid dose.
        if not np.isfinite(dose) or dose <= 0:
            return CompartmentResult(
                patient_id=pid, dose=dose, v=np.nan, cl=np.nan,
                k=np.nan, t_half=np.nan, r_squared=np.nan, converged=False,
            )

        # Define the model as a nested function so it can "see" the dose
        # variable from the enclosing scope (a Python closure).
        # curve_fit requires the signature f(x, *params).
        def model(t: np.ndarray, V: float, CL: float) -> np.ndarray:
            # One-compartment IV bolus equation
            return (dose / V) * np.exp(-(CL / V) * t)

        # Choose initial parameter guesses, preferring NCA-derived values.
        # getattr(obj, 'attr', default) safely reads an attribute, returning
        # the default if the attribute does not exist.
        if nca is not None and np.isfinite(getattr(nca, 'vd', np.nan)) and np.isfinite(getattr(nca, 'cl', np.nan)):
            v_init  = max(float(nca.vd), 0.1)
            cl_init = max(float(nca.cl), 0.001)
        elif nca is not None and np.isfinite(getattr(nca, 'cmax', np.nan)) and nca.cmax > 0:
            # If only Cmax is known: at t=0 the model gives C = Dose/V,
            # so V ≈ Dose/Cmax is a reasonable first guess.
            v_init  = dose / nca.cmax
            cl_init = 5.0
        else:
            v_init  = 10.0
            cl_init = 1.0

        p0     = [max(v_init, 0.1), max(cl_init, 0.001)]
        # bounds = ([lower bounds], [upper bounds]) — physiological limits
        bounds = ([0.01, 0.0001], [500.0, 200.0])

        try:
            # popt contains the optimal [V, CL]; we ignore the covariance
            # matrix (second return value, captured as _).
            popt, _ = curve_fit(model, time, dv, p0=p0, bounds=bounds, maxfev=10000)
            V_fit, CL_fit = popt
            k_fit      = CL_fit / V_fit
            t_half_fit = np.log(2.0) / k_fit

            # Calculate R² (coefficient of determination) to measure
            # goodness of fit.  R² = 1 means a perfect fit; 0 means the
            # model is no better than predicting the mean.
            dv_pred = model(time, V_fit, CL_fit)
            ss_res  = float(np.sum((dv - dv_pred) ** 2))   # residual sum of squares
            ss_tot  = float(np.sum((dv - np.mean(dv)) ** 2))  # total sum of squares
            r_sq    = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            return CompartmentResult(
                patient_id=pid, dose=dose, v=V_fit, cl=CL_fit,
                k=k_fit, t_half=t_half_fit, r_squared=r_sq, converged=True,
            )

        except (RuntimeError, ValueError):
            # curve_fit raises RuntimeError when it cannot converge within
            # maxfev iterations, and ValueError for invalid inputs.
            # We store a "failed" result rather than crashing the program.
            return CompartmentResult(
                patient_id=pid, dose=dose, v=np.nan, cl=np.nan,
                k=np.nan, t_half=np.nan, r_squared=np.nan, converged=False,
            )

    # ------------------------------------------------------------------
    # Table export helpers
    # ------------------------------------------------------------------

    def demographics_summary(self) -> pd.DataFrame:
        """Return a copy of the per-patient demographics table."""
        return self.demographics.copy()

    def nca_dataframe(self) -> pd.DataFrame:
        """Convert the list of NCAResult objects to a display-ready DataFrame."""
        if not self.nca_results:
            return pd.DataFrame()
        rows = []
        for r in self.nca_results:
            rows.append({
                'ID':       r.patient_id,
                'Dose':     _fmt(r.dose, 1),
                'N_obs':    r.n_obs,
                'Cmax':     _fmt(r.cmax, 3),
                'Tmax':     _fmt(r.tmax, 1),
                'AUC0-t':   _fmt(r.auc0t, 1),
                'AUC0-inf': _fmt(r.auc0inf, 1),
                't_half':   _fmt(r.t_half, 1),
                'CL':       _fmt(r.cl, 4),
                'Vd':       _fmt(r.vd, 2),
                'lz_R2':    _fmt(r.r2_lambdaz, 3),
                'Cmax_obs': r.cmax_is_observed,
            })
        return pd.DataFrame(rows)

    def compartment_dataframe(self) -> pd.DataFrame:
        """Convert the list of CompartmentResult objects to a display-ready DataFrame."""
        if not self.compartment_results:
            return pd.DataFrame()
        rows = []
        for r in self.compartment_results:
            rows.append({
                'ID':       r.patient_id,
                'Dose':     _fmt(r.dose, 1),
                'V(L)':     _fmt(r.v, 2),
                'CL':       _fmt(r.cl, 4),
                'k':        _fmt(r.k, 5),
                't_half':   _fmt(r.t_half, 1),
                'R2':       _fmt(r.r_squared, 3),
                'Converged': r.converged,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _fmt(val: float, decimals: int) -> Optional[float]:
    """
    Round a float to 'decimals' places, returning None for NaN/inf.

    Returning None (rather than NaN) makes the GUI table display '—'
    instead of 'nan', which is friendlier for non-technical users.
    """
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), decimals)
