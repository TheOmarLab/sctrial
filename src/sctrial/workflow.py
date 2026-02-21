from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from anndata import AnnData

from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets
from .design import TrialDesign
from .stats.did import DiDConfig, did_table


@dataclass
class TrialWorkflow:
    """Fluent API for common sctrial workflows.

    This class provides a minimal chainable interface for common tasks:
    preprocessing, gene scoring, and DiD analysis.
    """

    adata: AnnData
    last_result: Any | None = None

    def add_log1p_cpm_layer(self, counts_layer: str = "counts") -> TrialWorkflow:
        """Add a log1p-CPM layer to the workflow AnnData."""
        self.adata = add_log1p_cpm_layer(self.adata, counts_layer=counts_layer)
        return self

    def score_gene_sets(
        self,
        gene_sets: dict[str, list[str]],
        *,
        layer: str | None = None,
        method: Literal["zmean", "mean"] = "zmean",
        prefix: str = "ms_",
        min_genes: int = 5,
        overwrite: bool = False,
    ) -> TrialWorkflow:
        """Score gene sets and store module scores in `adata.obs`."""
        self.adata = score_gene_sets(
            self.adata,
            gene_sets,
            layer=layer,
            method=method,
            prefix=prefix,
            min_genes=min_genes,
            overwrite=overwrite,
        )
        return self

    def did_table(
        self,
        features: list[str],
        *,
        design: TrialDesign,
        visits: tuple[str, str],
        config: DiDConfig | None = None,
    ) -> TrialWorkflow:
        """Run DiD and store the result on the workflow."""
        self.last_result = did_table(
            self.adata,
            features=features,
            design=design,
            visits=visits,
            config=config,
        )
        return self

    def result(self) -> Any:
        """Return the last result computed in the workflow."""
        return self.last_result


def workflow(adata: AnnData) -> TrialWorkflow:
    """Create a TrialWorkflow for fluent chaining."""
    return TrialWorkflow(adata=adata)
