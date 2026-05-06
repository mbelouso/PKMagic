#!/usr/bin/env python3
from __future__ import annotations

import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QToolBar,
)

from pk_data import PKData
from pk_tabs import (
    CovariateTab,
    DataOverviewTab,
    CompartmentTab,
    IndividualCTTab,
    NCATab,
    PopulationCTTab,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._pk_data = PKData()
        self._tabs: list = []
        self._build_ui()
        self.setWindowTitle('PKMagic — Pharmacokinetics Analysis')
        self.resize(1440, 920)

    def _build_ui(self) -> None:
        # Toolbar -------------------------------------------------------
        toolbar = QToolBar('File')
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        browse_btn = QPushButton('Open CSV…')
        browse_btn.setFixedWidth(110)
        browse_btn.clicked.connect(self._on_load_file)

        self._file_label = QLabel('   No file loaded')
        self._file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        toolbar.addWidget(browse_btn)
        toolbar.addWidget(self._file_label)
        self.addToolBar(toolbar)

        # Tabs -----------------------------------------------------------
        tab_widget = QTabWidget()
        tab_specs = [
            ('Data Overview',     DataOverviewTab()),
            ('Individual C-t',    IndividualCTTab()),
            ('Population PK',     PopulationCTTab()),
            ('NCA',               NCATab()),
            ('Covariate',         CovariateTab()),
            ('1-Cpt Fit',         CompartmentTab()),
        ]
        for label, tab in tab_specs:
            self._tabs.append(tab)
            tab_widget.addTab(tab, label)

        self.setCentralWidget(tab_widget)
        self.setStatusBar(QStatusBar())

    # ------------------------------------------------------------------

    def _on_load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open PK Data File', '',
            'CSV Files (*.csv);;All Files (*)',
        )
        if not path:
            return

        self.statusBar().showMessage('Loading …')
        QApplication.processEvents()

        try:
            self._pk_data.load(path)
            self._file_label.setText(f'   {path}')
            self._refresh_all_tabs()
            n = len(self._pk_data.patient_ids)
            msg = f'Loaded {n} patients  |  {path}'
            if self._pk_data.warnings:
                msg += f'  [{len(self._pk_data.warnings)} warning(s) — see console]'
                for w in self._pk_data.warnings:
                    print(f'[PKMagic warning] {w}')
            self.statusBar().showMessage(msg)
        except Exception as exc:
            self.statusBar().showMessage(f'Error loading file: {exc}')
            traceback.print_exc()

    def _refresh_all_tabs(self) -> None:
        for tab in self._tabs:
            tab.refresh(self._pk_data)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
