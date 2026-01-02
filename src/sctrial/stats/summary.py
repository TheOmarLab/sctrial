from __future__ import annotations
import pandas as pd

def summarize_did_results(
    df: pd.DataFrame,
    alpha: float = 0.05,
    p_col: str = "p_DiD",
    fdr_col: str = "FDR_DiD",
    effect_col: str = "beta_DiD",
) -> str:
    """Generate a human-readable summary of DiD analysis results.

    Parameters
    ----------
    df
        DataFrame returned by did_table or abundance_did.
    alpha
        Significance threshold.
    p_col
        Name of the p-value column.
    fdr_col
        Name of the FDR-corrected p-value column.
    effect_col
        Name of the effect size (beta) column.

    Returns
    -------
    Markdown-formatted string summarizing the key findings.
    """
    if df.empty:
        return "No results to summarize."

    n_total = len(df)
    n_sig_p = (df[p_col] < alpha).sum()
    n_sig_fdr = (df[fdr_col] < alpha).sum() if fdr_col in df.columns else 0
    
    top_up = df[df[effect_col] > 0].sort_values(p_col).head(3)
    top_down = df[df[effect_col] < 0].sort_values(p_col).head(3)

    summary = [
        "## Trial-Aware DiD Summary",
        f"Total features tested: {n_total}",
        f"Significant (p < {alpha}): {n_sig_p}",
    ]
    if fdr_col in df.columns:
        summary.append(f"Significant (FDR < {alpha}): {n_sig_fdr}")
    
    summary.append("\n### Top Upregulated Features (by p-value):")
    if top_up.empty:
        summary.append("- None")
    else:
        for _, row in top_up.iterrows():
            feat = row.get("feature", row.get("celltype", "unknown"))
            summary.append(f"- {feat}: beta={row[effect_col]:.3f}, p={row[p_col]:.2e}")

    summary.append("\n### Top Downregulated Features (by p-value):")
    if top_down.empty:
        summary.append("- None")
    else:
        for _, row in top_down.iterrows():
            feat = row.get("feature", row.get("celltype", "unknown"))
            summary.append(f"- {feat}: beta={row[effect_col]:.3f}, p={row[p_col]:.2e}")

    return "\n".join(summary)
