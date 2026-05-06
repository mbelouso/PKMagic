from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pk_plots


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width: int = 14, height: int = 9, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class BaseTab(QWidget):
    """
    Layout (top to bottom):
        optional controls row
        MplCanvas  (stretch 3)
        NavigationToolbar2QT
        optional QTableWidget  (stretch 1)
    """

    def __init__(self, parent=None, table_rows: int = 0) -> None:
        super().__init__(parent)
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.table: QTableWidget | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        controls = self._build_controls()
        if controls is not None:
            outer.addWidget(controls)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)

        # Toolbar row: matplotlib nav toolbar + export button
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.addWidget(self.toolbar)
        export_btn = QPushButton('Export Plot…')
        export_btn.setFixedWidth(110)
        export_btn.clicked.connect(self._on_export)
        toolbar_row.addWidget(export_btn)
        plot_layout.addLayout(toolbar_row)

        plot_layout.addWidget(self.canvas)

        if table_rows > 0:
            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(plot_widget)
            self.table = QTableWidget()
            self.table.setMinimumHeight(150)
            splitter.addWidget(self.table)
            splitter.setSizes([650, 200])
            outer.addWidget(splitter)
        else:
            outer.addWidget(plot_widget)

    def _build_controls(self) -> QWidget | None:
        return None

    def _on_export(self) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            'Export Plot',
            '',
            'PNG Image (*.png);;PDF Document (*.pdf)',
        )
        if not path:
            return
        # Ensure correct extension when user types a name without one
        if selected_filter.startswith('PNG') and not path.lower().endswith('.png'):
            path += '.png'
        elif selected_filter.startswith('PDF') and not path.lower().endswith('.pdf'):
            path += '.pdf'
        try:
            self.canvas.fig.savefig(path, dpi=150, bbox_inches='tight')
        except Exception as exc:
            QMessageBox.critical(self, 'Export failed', str(exc))

    def refresh(self, pk_data) -> None:
        self.canvas.fig.clear()
        try:
            self._draw(pk_data)
        except Exception as exc:
            ax = self.canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Rendering error:\n{exc}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='red')
        self.canvas.draw()

        if self.table is not None:
            df = self._get_table_data(pk_data)
            _fill_table(self.table, df)

    def _draw(self, pk_data) -> None:
        raise NotImplementedError

    def _get_table_data(self, pk_data) -> pd.DataFrame:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Tab 0
# ---------------------------------------------------------------------------

class DataOverviewTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=1)

    def _draw(self, pk_data):
        pk_plots.plot_overview(self.canvas.fig, pk_data)

    def _get_table_data(self, pk_data) -> pd.DataFrame:
        df = pk_data.demographics_summary()
        return df[['id', 'sex_label', 'wt', 'age', 'dose', 'n_obs',
                   'first_time', 'last_time']].rename(columns={
            'id': 'ID', 'sex_label': 'Sex', 'wt': 'Weight(kg)', 'age': 'Age',
            'dose': 'Dose(mg)', 'n_obs': 'N_obs',
            'first_time': 'T_first', 'last_time': 'T_last',
        })


# ---------------------------------------------------------------------------
# Tab 1
# ---------------------------------------------------------------------------

class IndividualCTTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=0)
        self._color_by = 'sex'
        self._pk_data = None

    def _build_controls(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QLabel('Color by:'))
        self._combo = QComboBox()
        self._combo.addItems(['Sex', 'Weight'])
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        layout.addWidget(self._combo)
        layout.addStretch()
        return w

    def _on_combo_changed(self, text: str) -> None:
        self._color_by = text.lower()
        if self._pk_data is not None and self._pk_data.patient_ids:
            super().refresh(self._pk_data)

    def refresh(self, pk_data) -> None:
        self._pk_data = pk_data
        super().refresh(pk_data)

    def _draw(self, pk_data):
        pk_plots.plot_individual_ct(self.canvas.fig, pk_data, color_by=self._color_by)


# ---------------------------------------------------------------------------
# Tab 2
# ---------------------------------------------------------------------------

class PopulationCTTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=0)

    def _draw(self, pk_data):
        pk_plots.plot_population_ct(self.canvas.fig, pk_data)


# ---------------------------------------------------------------------------
# Tab 3
# ---------------------------------------------------------------------------

class NCATab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=1)

    def _draw(self, pk_data):
        pk_plots.plot_nca_summary(self.canvas.fig, pk_data)

    def _get_table_data(self, pk_data) -> pd.DataFrame:
        return pk_data.nca_dataframe()


# ---------------------------------------------------------------------------
# Tab 4
# ---------------------------------------------------------------------------

class CovariateTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=0)

    def _draw(self, pk_data):
        pk_plots.plot_covariate(self.canvas.fig, pk_data)


# ---------------------------------------------------------------------------
# Tab 5
# ---------------------------------------------------------------------------

class CompartmentTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=1)

    def _draw(self, pk_data):
        pk_plots.plot_compartment(self.canvas.fig, pk_data)

    def _get_table_data(self, pk_data) -> pd.DataFrame:
        return pk_data.compartment_dataframe()


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------

def _fill_table(table: QTableWidget, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        return
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels(list(df.columns))
    for r, (_, row) in enumerate(df.iterrows()):
        for c, val in enumerate(row):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                text = '—'
            elif isinstance(val, bool):
                text = 'Yes' if val else 'No'
            else:
                text = str(val)
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(r, c, item)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeToContents)
    header.setStretchLastSection(True)
