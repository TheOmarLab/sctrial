"""Integration tests for end-to-end workflows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st
from sctrial.design import TrialDesign


@pytest.fixture
def trial_adata_large():
    """Create a larger trial dataset for integration testing."""
    np.random.seed(42)

    n_participants = 20
    n_genes = 500
    n_cells_per_pv = 50

    participants = []
    visits = []
    arms = []
    celltypes = []
    cells_data = []

    for pid in range(n_participants):
        arm = "Treated" if pid < 10 else "Control"
        for visit in ["V1", "V2"]:
            for _ in range(n_cells_per_pv):
                participants.append(f"P{pid:02d}")
                visits.append(visit)
                arms.append(arm)
                celltypes.append(np.random.choice(["CD4_T", "CD8_T", "Mono", "B"]))

                # Generate expression with treatment effect
                if arm == "Treated" and visit == "V2":
                    # Add treatment effect to first 50 genes
                    expr = np.random.poisson(10, n_genes).astype(float)
                    expr[:50] += 5  # Treatment effect
                else:
                    expr = np.random.poisson(10, n_genes).astype(float)

                cells_data.append(expr)

    X = np.vstack(cells_data)
    obs = pd.DataFrame({
        "participant_id": participants,
        "visit": visits,
        "arm": arms,
        "celltype": celltypes,
    })
    var = pd.DataFrame(index=[f"Gene{i:03d}" for i in range(n_genes)])

    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()

    return adata


class TestEndToEndDidWorkflow:
    """Test complete DiD workflow from start to finish."""

    def test_complete_did_workflow(self, trial_adata_large):
        """Test full workflow: preprocess -> score -> DiD -> plot."""
        # 1. Create design
        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
            celltype_col="celltype",
        )

        # 2. Preprocessing
        adata = st.add_log1p_cpm_layer(
            trial_adata_large,
            counts_layer="counts",
            out_layer="log1p_cpm"
        )
        assert "log1p_cpm" in adata.layers

        # 3. Score gene sets
        gene_sets = {
            "EarlyGenes": [f"Gene{i:03d}" for i in range(25)],
            "LateGenes": [f"Gene{i:03d}" for i in range(25, 50)],
        }
        adata = st.score_gene_sets(
            adata,
            gene_sets,
            layer="log1p_cpm",
            method="zmean",
            prefix="ms_"
        )
        assert "ms_EarlyGenes" in adata.obs.columns
        assert "ms_LateGenes" in adata.obs.columns

        # 4. Run DiD
        features = ["ms_EarlyGenes", "ms_LateGenes"]
        res = st.did_table(
            adata,
            features=features,
            design=design,
            visits=("V1", "V2"),
            aggregate="participant_visit",
        )

        # Verify results structure
        assert len(res) == 2
        assert "beta_DiD" in res.columns
        assert "p_DiD" in res.columns
        assert "FDR_DiD" in res.columns
        assert "n_units" in res.columns

        # Should have sufficient participants
        assert res["n_units"].min() >= 10

        # 5. Verify plotting works (just test it doesn't crash)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            fig = st.plot_trial_interaction(
                adata,
                feature="ms_EarlyGenes",
                design=design,
                visits=("V1", "V2")
            )
            assert fig is not None
        except ImportError:
            pytest.skip("matplotlib not available")

    def test_did_by_celltype_workflow(self, trial_adata_large):
        """Test DiD stratified by cell type."""
        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
            celltype_col="celltype",
        )

        # Add scores
        adata = st.add_log1p_cpm_layer(trial_adata_large, counts_layer="counts")
        gene_sets = {"TestSet": [f"Gene{i:03d}" for i in range(10)]}
        adata = st.score_gene_sets(adata, gene_sets, layer="log1p_cpm", prefix="ms_")

        # Run stratified DiD
        res = st.did_table_by_celltype(
            adata,
            features=["ms_TestSet"],
            design=design,
            visits=("V1", "V2"),
            celltypes=["CD4_T", "CD8_T", "Mono"],
        )

        # Should have results for multiple cell types
        assert "celltype" in res.columns
        assert len(res["celltype"].unique()) >= 2
        assert len(res) >= 2


class TestEndToEndAbundanceWorkflow:
    """Test cell-type abundance analysis workflow."""

    def test_abundance_did_workflow(self, trial_adata_large):
        """Test abundance DiD from start to finish."""
        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
            celltype_col="celltype",
        )

        # Run abundance DiD
        res = st.abundance_did(
            trial_adata_large,
            design=design,
            visits=("V1", "V2"),
        )

        # Verify structure
        assert "celltype" in res.columns
        assert "beta_DiD" in res.columns
        assert "p_DiD" in res.columns

        # Should have results for each cell type
        expected_celltypes = ["CD4_T", "CD8_T", "Mono", "B"]
        assert set(res["celltype"].unique()).issubset(set(expected_celltypes))


class TestEndToEndComparisonsWorkflow:
    """Test within-arm and between-arm comparisons."""

    def test_within_arm_comparison_workflow(self, trial_adata_large):
        """Test within-arm comparison workflow."""
        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
        )

        adata = st.add_log1p_cpm_layer(trial_adata_large, counts_layer="counts")
        gene_sets = {"TestSet": [f"Gene{i:03d}" for i in range(10)]}
        adata = st.score_gene_sets(adata, gene_sets, layer="log1p_cpm", prefix="ms_")

        # Within-arm comparison
        res = st.within_arm_comparison(
            adata,
            arm="Treated",
            features=["ms_TestSet"],
            design=design,
            visits=("V1", "V2"),
        )

        assert len(res) == 1
        assert "beta_time" in res.columns
        assert "p_time" in res.columns

    def test_between_arm_comparison_workflow(self, trial_adata_large):
        """Test between-arm comparison workflow."""
        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
        )

        adata = st.add_log1p_cpm_layer(trial_adata_large, counts_layer="counts")
        gene_sets = {"TestSet": [f"Gene{i:03d}" for i in range(10)]}
        adata = st.score_gene_sets(adata, gene_sets, layer="log1p_cpm", prefix="ms_")

        # Between-arm comparison
        res = st.between_arm_comparison(
            adata,
            visit="V2",
            features=["ms_TestSet"],
            design=design,
        )

        assert len(res) == 1
        assert "beta_arm" in res.columns
        assert "p_arm" in res.columns


class TestEndToEndGSEAWorkflow:
    """Test GSEA integration workflow."""

    def test_gsea_did_workflow(self, trial_adata_large):
        """Test GSEA on DiD rankings."""
        try:
            import gseapy
        except ImportError:
            pytest.skip("gseapy not installed")

        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
        )

        adata = st.add_log1p_cpm_layer(trial_adata_large, counts_layer="counts")

        # Create simple gene sets
        gene_sets = {
            "SetA": [f"Gene{i:03d}" for i in range(25)],
            "SetB": [f"Gene{i:03d}" for i in range(25, 50)],
            "SetC": [f"Gene{i:03d}" for i in range(50, 75)],
        }

        # Run GSEA
        res = st.run_gsea_did(
            adata,
            gene_sets=gene_sets,
            design=design,
            visits=("V1", "V2"),
            layer="log1p_cpm",
            rank_by="signed_confidence",
        )

        # Should return a DataFrame
        assert isinstance(res, pd.DataFrame)
        # Should have enrichment results
        assert len(res) > 0


class TestDataValidation:
    """Test data validation and error handling."""

    def test_missing_required_columns(self, trial_adata_large):
        """Test error when required columns are missing."""
        design = TrialDesign(
            participant_col="nonexistent_col",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
        )

        adata = st.add_log1p_cpm_layer(trial_adata_large, counts_layer="counts")

        with pytest.raises(KeyError):
            st.did_table(
                adata,
                features=["Gene000"],
                design=design,
                visits=("V1", "V2"),
            )

    def test_insufficient_participants(self):
        """Test handling of insufficient participants."""
        # Create dataset with only 2 participants
        X = np.random.poisson(5, (20, 50)).astype(float)
        obs = pd.DataFrame({
            "participant_id": ["P1", "P1"] * 5 + ["P2", "P2"] * 5,
            "visit": ["V1", "V2"] * 10,
            "arm": ["Treated"] * 10 + ["Control"] * 10,
        })
        var = pd.DataFrame(index=[f"Gene{i}" for i in range(50)])
        adata = AnnData(X=X, obs=obs, var=var)
        adata.layers["counts"] = X.copy()

        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
        )

        adata = st.add_log1p_cpm_layer(adata, counts_layer="counts")

        # Should return results but with NaN for insufficient participants
        res = st.did_table(
            adata,
            features=["Gene0"],
            design=design,
            visits=("V1", "V2"),
        )

        # Results should have NaN due to n_units < 4
        assert res["n_units"].iloc[0] < 4
        assert pd.isna(res["beta_DiD"].iloc[0])
