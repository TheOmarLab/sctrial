from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from anndata import AnnData

from .design import TrialDesign
from .stats.did import DiDConfig, did_table
from .stats.summary import summarize_did_results


@dataclass
class DiDAnalyzer:
    """High-level interface for DiD analysis with stored results."""

    adata: AnnData
    design: TrialDesign
    results_: pd.DataFrame | None = None

    def fit(
        self,
        features: list[str],
        *,
        visits: tuple[str, str],
        config: DiDConfig | None = None,
        celltype: str | None = None,
    ) -> pd.DataFrame:
        """Run DiD and store results."""
        self.results_ = did_table(
            self.adata,
            features=features,
            design=self.design,
            visits=visits,
            celltype=celltype,
            config=config,
        )
        return self.results_

    def summarize(self) -> str:
        """Summarize the last DiD results."""
        if self.results_ is None:
            raise ValueError("No results available. Call fit() first.")
        return summarize_did_results(self.results_)

    def plot_forest(self, **kwargs):
        """Plot a forest plot of the last DiD results."""
        if self.results_ is None:
            raise ValueError("No results available. Call fit() first.")
        from .plotting import plot_did_forest

        return plot_did_forest(self.results_, **kwargs)
