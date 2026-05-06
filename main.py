#!/usr/bin/env python3
"""
main.py — Application entry point for PKMagic
==============================================
This file creates the main window and starts the Qt event loop.

It is intentionally kept short.  All the heavy lifting lives in the
other three modules:
    pk_data.py  — loading, cleaning, and analysing the CSV data
    pk_plots.py — drawing matplotlib figures
    pk_tabs.py  — the Qt widgets that contain those figures

Key Python and Qt concepts demonstrated here
--------------------------------------------
- QMainWindow  : a specialised window class that supports a menu bar,
                 toolbars, a status bar, and a single 'central widget'.
- QToolBar     : a row of buttons (and other widgets) docked to the edge
                 of the window.
- QTabWidget   : a container that holds multiple child widgets and shows
                 one at a time, selected by clickable tabs.
- QStatusBar   : the narrow bar along the bottom of the window; good for
                 showing progress messages and results.
- QApplication : the global Qt object that must exist before any widgets
                 are created and whose exec_() method runs the event loop.
- Event loop   : Qt works by running an infinite loop (exec_()) that
                 waits for events (mouse clicks, key presses, timers) and
                 dispatches them to the right widget.  Your code only runs
                 in response to those events — that is why we connect
                 signals to methods rather than writing a sequential script.
- if __name__ == '__main__': guards the entry-point code so that this
                 file can be safely imported without launching the GUI.
"""

from __future__ import annotations

import sys         # sys.argv passes command-line arguments to QApplication;
                   # sys.exit() ensures the process exit code is propagated.
import traceback   # traceback.print_exc() prints the full call stack to the
                   # terminal — helpful when debugging unexpected exceptions.

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,   # must be created before any other Qt object
    QFileDialog,    # standard OS file-open dialog
    QLabel,         # read-only text widget
    QMainWindow,    # top-level window with toolbar / status bar support
    QPushButton,    # clickable button
    QStatusBar,     # bottom-of-window message area
    QTabWidget,     # multi-tab container
    QToolBar,       # horizontal bar of widgets docked to the window edge
)

from pk_data import PKData       # our data model (loads and analyses the CSV)
from pk_tabs import (            # the six analysis tab widgets
    CovariateTab,
    DataOverviewTab,
    CompartmentTab,
    IndividualCTTab,
    NCATab,
    PopulationCTTab,
)


class MainWindow(QMainWindow):
    """
    The application's single top-level window.

    Responsibilities:
      1. Build the toolbar (file browse button + path label).
      2. Build and register all six analysis tabs.
      3. Respond to the 'Open CSV' button by loading data and
         asking every tab to redraw itself.
    """

    def __init__(self) -> None:
        # QMainWindow.__init__ must be called first so Qt can set up
        # the internal C++ object that underlies every Qt widget.
        super().__init__()

        # PKData holds all the loaded/computed data.  We create an empty
        # instance here; it is filled when the user opens a CSV file.
        self._pk_data = PKData()

        # Keep references to all tab widgets so we can call .refresh()
        # on each one after loading a new file.
        self._tabs: list = []

        self._build_ui()
        self.setWindowTitle('PKMagic — Pharmacokinetics Analysis')
        self.resize(1440, 920)   # default window size in pixels

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create and arrange all child widgets."""

        # --- Toolbar ---------------------------------------------------
        # QToolBar is a strip of widgets that docks to the window edge.
        # setMovable(False) prevents users from accidentally dragging it off.
        toolbar = QToolBar('File')
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        # A plain QPushButton works inside a toolbar just like anywhere else.
        browse_btn = QPushButton('Open CSV…')
        browse_btn.setFixedWidth(110)
        # .connect() wires the button's 'clicked' signal to our handler.
        # Signals are Qt's version of events/callbacks: when the button is
        # clicked, Qt automatically calls self._on_load_file().
        browse_btn.clicked.connect(self._on_load_file)

        # QLabel shows static (or dynamically updated) text.
        # TextSelectableByMouse lets the user copy the path by dragging.
        self._file_label = QLabel('   No file loaded')
        self._file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        toolbar.addWidget(browse_btn)
        toolbar.addWidget(self._file_label)
        self.addToolBar(toolbar)   # dock the toolbar to the top of the window

        # --- Tabs ------------------------------------------------------
        # QTabWidget displays one child widget at a time, with named tabs.
        tab_widget = QTabWidget()

        # A list of (tab label, tab widget) pairs makes it easy to add
        # or reorder tabs without touching the loop below.
        tab_specs = [
            ('Data Overview',     DataOverviewTab()),
            ('Individual C-t',    IndividualCTTab()),
            ('Population PK',     PopulationCTTab()),
            ('NCA',               NCATab()),
            ('Covariate',         CovariateTab()),
            ('1-Cpt Fit',         CompartmentTab()),
        ]

        # Unpack each (label, tab) pair and register it.
        for label, tab in tab_specs:
            self._tabs.append(tab)       # save reference for later refresh
            tab_widget.addTab(tab, label)

        # setCentralWidget defines the main content area of QMainWindow.
        self.setCentralWidget(tab_widget)

        # QStatusBar lives at the bottom of the window — we use it to
        # display loading progress and result summaries.
        self.setStatusBar(QStatusBar())

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _on_load_file(self) -> None:
        """
        Called when the user clicks 'Open CSV…'.

        Steps:
          1. Show a file-open dialog and get the chosen path.
          2. Ask PKData to load and analyse the file.
          3. Tell every tab to redraw with the new data.
          4. Update the status bar with a summary.
        """
        # QFileDialog.getOpenFileName() shows the OS file browser.
        # It returns a tuple: (chosen path, selected filter).
        # If the user cancels, path is an empty string.
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open PK Data File',
            '',                                    # start in current directory
            'CSV Files (*.csv);;All Files (*)',     # file-type filter
        )
        if not path:
            return   # user cancelled — nothing to do

        self.statusBar().showMessage('Loading …')
        # processEvents() flushes the Qt event queue so the 'Loading…'
        # message actually appears before we block on the file I/O.
        QApplication.processEvents()

        try:
            # PKData.load() reads the CSV, cleans it, runs NCA, and fits
            # the one-compartment model.  This may take a second or two.
            self._pk_data.load(path)
            self._file_label.setText(f'   {path}')
            self._refresh_all_tabs()

            # Build a summary message for the status bar.
            n = len(self._pk_data.patient_ids)
            msg = f'Loaded {n} patients  |  {path}'
            if self._pk_data.warnings:
                msg += f'  [{len(self._pk_data.warnings)} warning(s) — see console]'
                for w in self._pk_data.warnings:
                    print(f'[PKMagic warning] {w}')
            self.statusBar().showMessage(msg)

        except Exception as exc:
            # Catch any error during loading so the application doesn't
            # crash.  Show the error in the status bar and print the full
            # traceback to the terminal for debugging.
            self.statusBar().showMessage(f'Error loading file: {exc}')
            traceback.print_exc()

    def _refresh_all_tabs(self) -> None:
        """Ask every tab widget to redraw itself with the current data."""
        for tab in self._tabs:
            tab.refresh(self._pk_data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Create the Qt application and show the main window.

    QApplication must be created before any QWidget.  It manages the
    event loop and shared resources (fonts, clipboard, etc.).

    sys.argv passes command-line arguments to Qt (some Qt plugins use
    flags like --display or --style).

    app.exec_() starts the event loop and blocks until the user closes
    the main window, then returns the exit code.

    sys.exit() propagates that exit code to the operating system — useful
    for scripting (e.g. checking whether the app exited cleanly).
    """
    app = QApplication(sys.argv)
    # 'Fusion' is a cross-platform Qt style that looks identical on
    # Windows, macOS, and Linux — good for a teaching demo.
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


# ---------------------------------------------------------------------------
# Guard: only run main() when this file is executed directly
# ---------------------------------------------------------------------------

# When Python executes a file directly (python main.py), it sets the
# special variable __name__ to '__main__'.  When another module imports
# this file, __name__ is 'main' instead.  This guard means the GUI won't
# accidentally launch if pk_tabs.py or another module imports main.py.
if __name__ == '__main__':
    main()
