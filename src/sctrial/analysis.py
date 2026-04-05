"""High-level analysis helpers for trial-aware inference."""

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
        """Run DiD and store results.

        Returns a *copy* of the results DataFrame so that external
        mutations do not alter the analyzer's internal state.

        Parameters
        ----------
        features
            List of features to analyze.
        visits
            Tuple of (baseline, followup) visit labels.
        config
            Configuration for the DiD analysis. See `sctrial.stats.did.DiDConfig` for more details.
        celltype
            Cell type to analyze. If None, all cell types are analyzed.

        Returns
        -------
        pd.DataFrame
            A copy of the results DataFrame.
        """
        self.results_ = did_table(
            self.adata,
            features=features,
            design=self.design,
            visits=visits,
            celltype=celltype,
            config=config,
        )
        return self.results_.copy()

    def summarize(self) -> str:
        """Summarize the last DiD results.

        Returns
        -------
        str
            A summary of the DiD results.
        """
        if self.results_ is None:
            raise ValueError("No results available. Call fit() first.")
        return summarize_did_results(self.results_)

    def plot_forest(self, **kwargs):
        """Plot a forest plot of the last DiD results.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to `sctrial.plotting.plot_did_forest`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the forest plot.
        """
        if self.results_ is None:
            raise ValueError("No results available. Call fit() first.")
        from .plotting import plot_did_forest

        return plot_did_forest(self.results_, **kwargs)
