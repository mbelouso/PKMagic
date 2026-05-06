#!/usr/bin/env python3
"""
pk_tabs.py — PyQt5 tab widgets for PKMagic
==========================================
This module defines the graphical tab panels that appear in the main window.

Each tab follows the same pattern (the 'Template Method' design pattern):
  1. A shared base class (BaseTab) handles the common layout — toolbar,
     matplotlib canvas, optional data table, export button.
  2. Six subclasses each override just two small methods:
       _draw()           — which plot function to call
       _get_table_data() — which data to show in the table below the plot

This keeps each subclass very short and prevents code duplication.

Key Python and Qt concepts demonstrated here
--------------------------------------------
- Inheritance      : BaseTab is the parent class; all six tab classes
                     inherit from it using class MyTab(BaseTab).
- super()          : calls the parent class's version of a method.
                     Essential when overriding __init__ so the parent
                     can do its own initialisation first.
- Signals & slots  : Qt's event system.  button.clicked.connect(method)
                     means "call method() whenever this button is clicked".
- QSplitter        : a resizable divider between two widgets.  The user
                     can drag it to give more space to the plot or the table.
- QFileDialog      : a standard OS file-browser dialog for saving files.
- QTableWidget     : a spreadsheet-style widget for displaying tabular data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,        # dropdown selector
    QFileDialog,      # file browser dialog
    QHBoxLayout,      # horizontal box layout
    QHeaderView,      # table header (for resize modes)
    QLabel,           # static text widget
    QMessageBox,      # pop-up message/error dialog
    QPushButton,      # clickable button
    QSplitter,        # resizable divider between two widgets
    QTableWidget,     # spreadsheet-style data table
    QTableWidgetItem, # individual cell in a QTableWidget
    QVBoxLayout,      # vertical box layout
    QWidget,          # base class for all Qt widgets
)

import pk_plots   # all the matplotlib drawing functions


# ---------------------------------------------------------------------------
# Matplotlib canvas wrapper
# ---------------------------------------------------------------------------

class MplCanvas(FigureCanvasQTAgg):
    """
    A Qt widget that contains a matplotlib Figure.

    FigureCanvasQTAgg is the 'bridge' class provided by matplotlib that
    makes a Figure behave like a Qt widget so it can be placed inside
    layouts alongside buttons, labels, etc.

    We expose self.fig so that plot functions can draw into it directly.
    """
    def __init__(self, parent=None, width: int = 14, height: int = 9, dpi: int = 100):
        # Create the matplotlib Figure first, then initialise the Qt side.
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)   # super() calls FigureCanvasQTAgg.__init__
        self.setParent(parent)


# ---------------------------------------------------------------------------
# Base tab class — shared layout and behaviour
# ---------------------------------------------------------------------------

class BaseTab(QWidget):
    """
    Parent class for all six analysis tabs.

    Visual layout (top to bottom inside each tab):
        [optional controls row — e.g. a combo box]
        [matplotlib NavigationToolbar2QT + Export button]
        [MplCanvas — the actual plot]
        [optional QTableWidget — data table]  ← shown when table_rows > 0

    Subclasses should override:
        _draw(pk_data)          — call the appropriate pk_plots function
        _get_table_data(pk_data)— return a DataFrame for the table
        _build_controls()       — return a QWidget with extra controls,
                                  or None if no controls are needed
    """

    def __init__(self, parent=None, table_rows: int = 0) -> None:
        # Always call the parent __init__ first when subclassing a Qt widget.
        super().__init__(parent)

        self.canvas = MplCanvas(self)
        # NavigationToolbar2QT adds zoom, pan, and save buttons to the canvas.
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.table: QTableWidget | None = None   # only created if table_rows > 0

        # QVBoxLayout stacks widgets vertically (top to bottom).
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # _build_controls() returns None in BaseTab but subclasses can
        # override it to add extra widgets (like a combo box) at the top.
        controls = self._build_controls()
        if controls is not None:
            outer.addWidget(controls)

        # Group the toolbar and canvas into a container widget.
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)

        # Horizontal row: matplotlib's toolbar on the left, Export button on the right.
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.addWidget(self.toolbar)
        export_btn = QPushButton('Export Plot…')
        export_btn.setFixedWidth(110)
        # .connect() links the button's 'clicked' signal to our _on_export method.
        export_btn.clicked.connect(self._on_export)
        toolbar_row.addWidget(export_btn)
        plot_layout.addLayout(toolbar_row)

        plot_layout.addWidget(self.canvas)

        if table_rows > 0:
            # QSplitter lets the user drag a divider to resize the plot vs table.
            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(plot_widget)

            self.table = QTableWidget()
            self.table.setMinimumHeight(150)
            splitter.addWidget(self.table)
            splitter.setSizes([650, 200])   # initial pixel sizes [plot, table]

            outer.addWidget(splitter)
        else:
            outer.addWidget(plot_widget)

    # ------------------------------------------------------------------
    # Methods subclasses can override
    # ------------------------------------------------------------------

    def _build_controls(self) -> QWidget | None:
        """Return an optional controls widget to show above the toolbar.
        Override in subclasses that need extra controls (e.g. a combo box).
        Returning None (default) means no controls row is shown."""
        return None

    # ------------------------------------------------------------------
    # Export to PNG or PDF
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        """
        Open a 'Save As' dialog and write the current figure to disk.

        QFileDialog.getSaveFileName() returns a tuple:
            (chosen file path as a string, the selected filter string).
        If the user cancels the dialog, path will be an empty string.
        """
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            'Export Plot',
            '',
            'PNG Image (*.png);;PDF Document (*.pdf)',
        )
        if not path:
            return   # user cancelled — do nothing

        # If the user typed a name without an extension, add the correct one
        # based on whichever filter they had selected in the dialog.
        if selected_filter.startswith('PNG') and not path.lower().endswith('.png'):
            path += '.png'
        elif selected_filter.startswith('PDF') and not path.lower().endswith('.pdf'):
            path += '.pdf'

        try:
            # fig.savefig() writes the figure to a file.
            # dpi=150 gives a higher-resolution image than the screen default.
            # bbox_inches='tight' trims unnecessary white space around the plot.
            self.canvas.fig.savefig(path, dpi=150, bbox_inches='tight')
        except Exception as exc:
            # Show a pop-up error message if saving fails (e.g. permission denied).
            QMessageBox.critical(self, 'Export failed', str(exc))

    # ------------------------------------------------------------------
    # Refresh — called whenever a new CSV is loaded
    # ------------------------------------------------------------------

    def refresh(self, pk_data) -> None:
        """
        Clear the figure and redraw it with the newly loaded data.

        fig.clear() removes all axes and artists from the figure so we
        start from a blank canvas each time.  This is simpler than trying
        to update individual plot elements in place.
        """
        self.canvas.fig.clear()
        try:
            self._draw(pk_data)
        except Exception as exc:
            # If drawing fails, show a red error message inside the canvas
            # rather than crashing the whole application.
            ax = self.canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Rendering error:\n{exc}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='red')

        # canvas.draw() tells Qt to repaint the widget with the new figure contents.
        self.canvas.draw()

        # Update the data table if this tab has one.
        if self.table is not None:
            df = self._get_table_data(pk_data)
            _fill_table(self.table, df)

    def _draw(self, pk_data) -> None:
        """Override this in each subclass to call the correct plot function."""
        raise NotImplementedError

    def _get_table_data(self, pk_data) -> pd.DataFrame:
        """Override this in subclasses that display a data table."""
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Tab 0 — Data Overview
# ---------------------------------------------------------------------------

class DataOverviewTab(BaseTab):
    """
    Shows population demographics: age/weight histograms, dose vs weight
    scatter, sex distribution bar chart, and a per-patient summary table.
    """
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=1)   # table_rows=1 enables the table

    def _draw(self, pk_data):
        pk_plots.plot_overview(self.canvas.fig, pk_data)

    def _get_table_data(self, pk_data) -> pd.DataFrame:
        # Select and rename columns for a cleaner table display.
        df = pk_data.demographics_summary()
        return df[['id', 'sex_label', 'wt', 'age', 'dose', 'n_obs',
                   'first_time', 'last_time']].rename(columns={
            'id':         'ID',
            'sex_label':  'Sex',
            'wt':         'Weight(kg)',
            'age':        'Age',
            'dose':       'Dose(mg)',
            'n_obs':      'N_obs',
            'first_time': 'T_first',
            'last_time':  'T_last',
        })


# ---------------------------------------------------------------------------
# Tab 1 — Individual Concentration–Time Profiles
# ---------------------------------------------------------------------------

class IndividualCTTab(BaseTab):
    """
    Grid of one concentration–time plot per patient.
    A combo box lets the user switch the colour scheme between sex and weight.
    """
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=0)
        self._color_by = 'sex'    # current colour mode
        self._pk_data  = None     # store a reference so we can redraw on combo change

    def _build_controls(self) -> QWidget:
        """Return a horizontal row containing the 'Color by:' combo box."""
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QLabel('Color by:'))

        self._combo = QComboBox()
        self._combo.addItems(['Sex', 'Weight'])
        # Connect the combo box's change signal to our handler method.
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        layout.addWidget(self._combo)
        layout.addStretch()   # pushes widgets to the left
        return w

    def _on_combo_changed(self, text: str) -> None:
        """Called automatically by Qt whenever the user selects a different item."""
        self._color_by = text.lower()
        # Only redraw if data has already been loaded.
        if self._pk_data is not None and self._pk_data.patient_ids:
            super().refresh(self._pk_data)

    def refresh(self, pk_data) -> None:
        # Save the reference so _on_combo_changed can trigger a redraw later.
        self._pk_data = pk_data
        super().refresh(pk_data)

    def _draw(self, pk_data):
        pk_plots.plot_individual_ct(self.canvas.fig, pk_data, color_by=self._color_by)


# ---------------------------------------------------------------------------
# Tab 2 — Population Concentration–Time
# ---------------------------------------------------------------------------

class PopulationCTTab(BaseTab):
    """
    Shows the population mean curve with a 95% band, plus mean curves
    stratified by sex and body weight quartile.
    """
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=0)

    def _draw(self, pk_data):
        pk_plots.plot_population_ct(self.canvas.fig, pk_data)


# ---------------------------------------------------------------------------
# Tab 3 — NCA Results
# ---------------------------------------------------------------------------

class NCATab(BaseTab):
    """
    Histograms of the NCA parameter distributions across the population,
    plus a per-patient NCA results table.
    """
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=1)

    def _draw(self, pk_data):
        pk_plots.plot_nca_summary(self.canvas.fig, pk_data)

    def _get_table_data(self, pk_data) -> pd.DataFrame:
        return pk_data.nca_dataframe()


# ---------------------------------------------------------------------------
# Tab 4 — Covariate Analysis
# ---------------------------------------------------------------------------

class CovariateTab(BaseTab):
    """
    Scatter plots showing how clearance (CL) relates to weight and age,
    plus a box plot comparing CL between sexes.
    """
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=0)

    def _draw(self, pk_data):
        pk_plots.plot_covariate(self.canvas.fig, pk_data)


# ---------------------------------------------------------------------------
# Tab 5 — One-Compartment Model Fits
# ---------------------------------------------------------------------------

class CompartmentTab(BaseTab):
    """
    Observed data overlaid with the fitted one-compartment model curve for
    each patient, plus a parameter table (V, CL, t½, R²).
    """
    def __init__(self, parent=None):
        super().__init__(parent, table_rows=1)

    def _draw(self, pk_data):
        pk_plots.plot_compartment(self.canvas.fig, pk_data)

    def _get_table_data(self, pk_data) -> pd.DataFrame:
        return pk_data.compartment_dataframe()


# ---------------------------------------------------------------------------
# Shared table-filling helper
# ---------------------------------------------------------------------------

def _fill_table(table: QTableWidget, df: pd.DataFrame) -> None:
    """
    Populate a QTableWidget from a pandas DataFrame.

    We iterate over every (row, column) cell and create a QTableWidgetItem
    for each one.  Items are set to read-only so users cannot accidentally
    edit the displayed results.

    The .flags() method returns a bitmask of item properties; we use
    bitwise AND with the bitwise NOT (~) of ItemIsEditable to clear that
    flag while leaving all other flags unchanged.
    """
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        return

    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels(list(df.columns))

    for r, (_, row) in enumerate(df.iterrows()):
        for c, val in enumerate(row):
            # Convert each value to a display string.
            if val is None or (isinstance(val, float) and np.isnan(val)):
                text = '—'          # em dash for missing values
            elif isinstance(val, bool):
                text = 'Yes' if val else 'No'
            else:
                text = str(val)

            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(r, c, item)

    # Resize each column to fit its content, then let the last column
    # stretch to fill any remaining space.
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeToContents)
    header.setStretchLastSection(True)
