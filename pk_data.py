from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid as _trapz
from scipy.optimize import curve_fit

COLUMN_MAP = {
    '#ID': 'id',
    'PatientID': 'id',
    'time(min)': 'time',
    'wt(kg)': 'wt',
    'dv(mg/L)': 'dv',
    'dependtvariable': 'dvid',
}

REQUIRED_INTERNAL = {'id', 'time', 'wt', 'age', 'sex', 'amt', 'rate', 'dvid', 'dv', 'mdv'}


@dataclass
class NCAResult:
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
    patient_id: str
    dose: float
    v: float
    cl: float
    k: float
    t_half: float
    r_squared: float
    converged: bool


class PKData:
    def __init__(self):
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.obs_df: pd.DataFrame = pd.DataFrame()
        self.dose_df: pd.DataFrame = pd.DataFrame()
        self.patient_ids: list[str] = []
        self.demographics: pd.DataFrame = pd.DataFrame()
        self.nca_results: list[NCAResult] = []
        self.compartment_results: list[CompartmentResult] = []
        self.warnings: list[str] = []
        self.time_label: str = 'time'

    def load(self, filepath: str) -> None:
        self.warnings = []
        import io
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Keep header (first line) unconditionally; skip any body line starting with '#'
        header = lines[0]
        body = [l for l in lines[1:] if not l.lstrip().startswith('#')]
        n_excluded = len(lines) - 1 - len(body)
        if n_excluded:
            self.warnings.append(
                f"Excluded {n_excluded} commented-out data row(s) from analysis"
            )

        df = pd.read_csv(io.StringIO(header + ''.join(body)), dtype=str)
        df = self._clean(df)
        self.raw_df = df
        self._split_dose_obs()
        self._build_demographics()
        self.compute_nca()
        self.fit_one_compartment()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip column name whitespace
        df.columns = [c.strip() for c in df.columns]

        # Strip whitespace from all string cell values
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()

        # Rename known columns; ignore unknown extras
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

        # Detect time unit label from original column name before rename
        orig_time_cols = [c for c in COLUMN_MAP if 'time' in c.lower()]
        if orig_time_cols:
            self.time_label = orig_time_cols[0]

        missing = REQUIRED_INTERNAL - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns after rename: {missing}")

        # Replace '.' sentinel with NaN in numeric columns
        for col in ['amt', 'rate', 'dv']:
            df[col] = df[col].replace('.', np.nan)

        # Cast all numeric columns
        for col in ['time', 'wt', 'age', 'sex', 'mdv', 'dvid']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in ['amt', 'rate', 'dv']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Cast id to numeric for proper sorting; keep as float internally
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df = df.dropna(subset=['id'])

        return df

    def _split_dose_obs(self) -> None:
        df = self.raw_df

        # dose_df: rows with a recorded dose amount
        self.dose_df = df[df['amt'].notna()].copy()

        # obs_df: valid concentration observations only
        obs = df[(df['mdv'] == 0) & (df['dv'].notna())].copy()

        # Average duplicate (id, time) rows — handles patient 8 replicate samples
        n_before = len(obs)
        agg_dict = {'dv': 'mean', 'wt': 'first', 'age': 'first', 'sex': 'first', 'dvid': 'first'}
        obs = obs.groupby(['id', 'time'], as_index=False).agg(agg_dict)
        n_dup = n_before - len(obs)
        if n_dup > 0:
            self.warnings.append(
                f"Averaged {n_dup} duplicate (id, time) observation(s) before analysis"
            )

        self.obs_df = obs.sort_values(['id', 'time']).reset_index(drop=True)

        # Build sorted string patient ID list
        all_ids = sorted(df['id'].dropna().unique())
        self.patient_ids = [str(int(pid)) for pid in all_ids]

    def _build_demographics(self) -> None:
        rows = []
        for pid in self.patient_ids:
            pid_f = float(pid)
            obs = self.obs_df[self.obs_df['id'] == pid_f]
            dose_row = self.dose_df[self.dose_df['id'] == pid_f]
            dose = float(dose_row['amt'].iloc[0]) if not dose_row.empty else np.nan
            wt = float(obs['wt'].iloc[0]) if not obs.empty else np.nan
            age = float(obs['age'].iloc[0]) if not obs.empty else np.nan
            sex_raw = float(obs['sex'].iloc[0]) if not obs.empty else np.nan
            sex_int = int(sex_raw) if np.isfinite(sex_raw) else None
            rows.append({
                'id': pid,
                'sex': sex_int,
                'sex_label': ('M' if sex_int == 1 else 'F') if sex_int is not None else '?',
                'wt': wt,
                'age': age,
                'dose': dose,
                'n_obs': len(obs),
                'first_time': float(obs['time'].min()) if not obs.empty else np.nan,
                'last_time': float(obs['time'].max()) if not obs.empty else np.nan,
            })
        self.demographics = pd.DataFrame(rows)

    def get_patient_obs(self, pid: str) -> pd.DataFrame:
        return self.obs_df[self.obs_df['id'] == float(pid)].copy()

    def get_patient_dose(self, pid: str) -> float:
        row = self.dose_df[self.dose_df['id'] == float(pid)]
        return float(row['amt'].iloc[0]) if not row.empty else np.nan

    # ------------------------------------------------------------------
    # NCA
    # ------------------------------------------------------------------

    def compute_nca(self) -> None:
        self.nca_results = []
        for pid in self.patient_ids:
            try:
                self.nca_results.append(self._nca_patient(pid))
            except Exception as exc:
                self.warnings.append(f"NCA failed for patient {pid}: {exc}")

    def _nca_patient(self, pid: str) -> NCAResult:
        obs = self.get_patient_obs(pid)
        dose = self.get_patient_dose(pid)
        time = obs['time'].values.astype(float)
        dv = obs['dv'].values.astype(float)
        n_obs = len(obs)

        cmax = float(np.max(dv))
        tmax = float(time[np.argmax(dv)])
        auc0t = float(_trapz(dv, time))
        cmax_is_observed = float(time[0]) < 24.0

        # Terminal elimination rate constant via adaptive log-linear regression
        pos_mask = dv > 0
        pos_time = time[pos_mask]
        pos_dv = dv[pos_mask]

        lambda_z = np.nan
        t_half = np.nan
        auc0inf = np.nan
        cl = np.nan
        vd = np.nan
        r2_lambdaz = np.nan

        if len(pos_time) >= 3:
            best_r2 = -np.inf
            best_lz = np.nan

            for n_pts in range(3, len(pos_time) + 1):
                t_seg = pos_time[-n_pts:]
                lndv_seg = np.log(pos_dv[-n_pts:])
                slope, _, r, _, _ = stats.linregress(t_seg, lndv_seg)
                lz = -slope
                r2 = r ** 2

                if lz <= 0:
                    break  # must be a declining phase

                if r2 >= best_r2 - 0.01:
                    best_r2 = r2
                    best_lz = lz
                else:
                    break

            if np.isfinite(best_lz) and best_lz > 0:
                lambda_z = best_lz
                r2_lambdaz = best_r2
                t_half = np.log(2.0) / lambda_z
                clast = float(pos_dv[-1])
                auc0inf = auc0t + clast / lambda_z
                if np.isfinite(dose) and auc0inf > 0:
                    cl = dose / auc0inf
                    vd = cl / lambda_z

        return NCAResult(
            patient_id=pid,
            dose=dose,
            n_obs=n_obs,
            cmax=cmax,
            tmax=tmax,
            auc0t=auc0t,
            lambda_z=lambda_z,
            t_half=t_half,
            auc0inf=auc0inf,
            cl=cl,
            vd=vd,
            r2_lambdaz=r2_lambdaz,
            cmax_is_observed=cmax_is_observed,
        )

    # ------------------------------------------------------------------
    # 1-compartment fitting
    # ------------------------------------------------------------------

    def fit_one_compartment(self) -> None:
        self.compartment_results = []
        nca_lookup = {r.patient_id: r for r in self.nca_results}
        for pid in self.patient_ids:
            try:
                nca = nca_lookup.get(pid)
                self.compartment_results.append(self._fit_patient(pid, nca))
            except Exception as exc:
                self.warnings.append(f"1-cpt fit failed for patient {pid}: {exc}")

    def _fit_patient(self, pid: str, nca: Optional[NCAResult]) -> CompartmentResult:
        obs = self.get_patient_obs(pid)
        dose = self.get_patient_dose(pid)
        time = obs['time'].values.astype(float)
        dv = obs['dv'].values.astype(float)

        if not np.isfinite(dose) or dose <= 0:
            return CompartmentResult(
                patient_id=pid, dose=dose, v=np.nan, cl=np.nan,
                k=np.nan, t_half=np.nan, r_squared=np.nan, converged=False,
            )

        def model(t: np.ndarray, V: float, CL: float) -> np.ndarray:
            return (dose / V) * np.exp(-(CL / V) * t)

        # Warm-start initial guesses from NCA where available
        if nca is not None and np.isfinite(getattr(nca, 'vd', np.nan)) and np.isfinite(getattr(nca, 'cl', np.nan)):
            v_init = max(float(nca.vd), 0.1)
            cl_init = max(float(nca.cl), 0.001)
        elif nca is not None and np.isfinite(getattr(nca, 'cmax', np.nan)) and nca.cmax > 0:
            v_init = dose / nca.cmax
            cl_init = 5.0
        else:
            v_init = 10.0
            cl_init = 1.0

        p0 = [max(v_init, 0.1), max(cl_init, 0.001)]
        bounds = ([0.01, 0.0001], [500.0, 200.0])

        try:
            popt, _ = curve_fit(model, time, dv, p0=p0, bounds=bounds, maxfev=10000)
            V_fit, CL_fit = popt
            k_fit = CL_fit / V_fit
            t_half_fit = np.log(2.0) / k_fit

            dv_pred = model(time, V_fit, CL_fit)
            ss_res = float(np.sum((dv - dv_pred) ** 2))
            ss_tot = float(np.sum((dv - np.mean(dv)) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            return CompartmentResult(
                patient_id=pid, dose=dose, v=V_fit, cl=CL_fit,
                k=k_fit, t_half=t_half_fit, r_squared=r_sq, converged=True,
            )
        except (RuntimeError, ValueError):
            return CompartmentResult(
                patient_id=pid, dose=dose, v=np.nan, cl=np.nan,
                k=np.nan, t_half=np.nan, r_squared=np.nan, converged=False,
            )

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def demographics_summary(self) -> pd.DataFrame:
        return self.demographics.copy()

    def nca_dataframe(self) -> pd.DataFrame:
        if not self.nca_results:
            return pd.DataFrame()
        rows = []
        for r in self.nca_results:
            rows.append({
                'ID': r.patient_id,
                'Dose': _fmt(r.dose, 1),
                'N_obs': r.n_obs,
                'Cmax': _fmt(r.cmax, 3),
                'Tmax': _fmt(r.tmax, 1),
                'AUC0-t': _fmt(r.auc0t, 1),
                'AUC0-inf': _fmt(r.auc0inf, 1),
                't_half': _fmt(r.t_half, 1),
                'CL': _fmt(r.cl, 4),
                'Vd': _fmt(r.vd, 2),
                'lz_R2': _fmt(r.r2_lambdaz, 3),
                'Cmax_obs': r.cmax_is_observed,
            })
        return pd.DataFrame(rows)

    def compartment_dataframe(self) -> pd.DataFrame:
        if not self.compartment_results:
            return pd.DataFrame()
        rows = []
        for r in self.compartment_results:
            rows.append({
                'ID': r.patient_id,
                'Dose': _fmt(r.dose, 1),
                'V(L)': _fmt(r.v, 2),
                'CL': _fmt(r.cl, 4),
                'k': _fmt(r.k, 5),
                't_half': _fmt(r.t_half, 1),
                'R2': _fmt(r.r_squared, 3),
                'Converged': r.converged,
            })
        return pd.DataFrame(rows)


def _fmt(val: float, decimals: int) -> Optional[float]:
    if val is None or not np.isfinite(val):
        return None
    return round(float(val), decimals)
