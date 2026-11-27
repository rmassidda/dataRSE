"""
Command-line interface for JMLR experiments.
"""

import fire

from jmlr_exp.experiment import run
from jmlr_exp.mise import run_mise
from jmlr_exp.plot import figure3, plot_abs, plot_compare, summary_grid

fire.Fire(
    {
        "run": run,
        "run_mise": run_mise,
        "figure3": figure3,
        "summary_grid": summary_grid,
        "plot_abs": plot_abs,
        "plot_compare": plot_compare,
    }
)
