from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from anndata import AnnData

from .design import TrialDesign
from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets
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
        """Add a log1p-CPM layer to the workflow AnnData.

        Parameters
        ----------
        counts_layer
            Layer name in `adata.layers` to use for counts data.

        Returns
        -------
        TrialWorkflow
            The workflow object with the new log1p-CPM layer added.
        """
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
        """Score gene sets and store module scores in `adata.obs`.

        Parameters
        ----------
        gene_sets
            Dictionary mapping set names to lists of gene names.
            Each value must be a ``list`` (not a bare string).  Duplicate gene
            names within a set are automatically removed.
        layer
            Layer name in `adata.layers` to use for expression data.
            If None, uses `adata.X`.
            For log1p-CPM workflows, use layer="log1p_cpm".
        method
            Scoring method:
            - "zmean": Z-score each gene across cells (within the current AnnData),
              then average z-scores across genes. This is the recommended method
              as it accounts for different expression scales across genes.
            - "mean": Mean expression across genes.
        prefix
            Prefix to add to column names (e.g., ``ms_`` for module scores).
        min_genes
            Minimum number of genes from the set that must be present in the data.
            Default is 5.
        overwrite
            If False, skip gene sets that already have a column in `adata.obs`.
            Default is True.

        Returns
        -------
        TrialWorkflow
            The workflow object with the gene sets scored and module scores stored in ``adata.obs``.
        """
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
        """Run DiD and store the result on the workflow.

        Parameters
        ----------
        features
            List of feature names to analyze.
        design
            Trial design object.
        visits
            Tuple of (baseline, followup) visit labels.
        config
            Configuration for the DiD analysis. See `sctrial.stats.did.DiDConfig` for more details.
            If None, uses the default configuration.

        Returns
        -------
        TrialWorkflow
            The workflow object with the DiD result stored in ``self.last_result``.
        """
        self.last_result = did_table(
            self.adata,
            features=features,
            design=design,
            visits=visits,
            config=config,
        )
        return self

    def result(self) -> Any:
        """Return the last result computed in the workflow.

        Returns
        -------
        Any
            The last result computed in the workflow (e.g., a DataFrame).
        """
        return self.last_result


def workflow(adata: AnnData) -> TrialWorkflow:
    """Create a TrialWorkflow for fluent chaining.

    Parameters
    ----------
    adata
        AnnData object with trial data.

    Returns
    -------
    TrialWorkflow
        A TrialWorkflow object with the AnnData object set.
    """
    return TrialWorkflow(adata=adata)
