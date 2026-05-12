import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class CorrelationsAnalyzer:
    """Analyze feature correlations with the target variable."""

    def corr_report(self, df: pd.DataFrame, method: str = "pearson", min_n: int = 60, target_column_name = "y_up_1d") -> pd.DataFrame:
        """
        Compute correlation of all numeric features with y_up_Nd.
        Returns a DataFrame with columns: feature, corr, abs_corr, n, p_value, q_value_fdr.
        """
        tmp = df.copy()
        tmp[target_column_name] = pd.to_numeric(tmp[target_column_name], errors="coerce")

        features = [
            c for c in tmp.columns
            if c not in {"date", target_column_name}
            and pd.api.types.is_numeric_dtype(tmp[c])
        ]

        # Fast correlation pass (without p-values)
        corr = tmp[features].corrwith(tmp[target_column_name], method=method)
        res = corr.rename("corr").to_frame()
        res["abs_corr"] = res["corr"].abs()

        # Per-feature valid sample size (n)
        y = tmp[target_column_name]
        n_list = []
        for c in features:
            m = tmp[c].notna() & y.notna()
            n_list.append(int(m.sum()))
        res["n"] = n_list

        # p-values
        pvals = []
        for c in features:
            m = tmp[c].notna() & y.notna()
            if m.sum() < min_n:
                pvals.append(np.nan)
                continue
            x = pd.to_numeric(tmp.loc[m, c], errors="coerce")
            yy = tmp.loc[m, target_column_name]
            if method == "pearson":
                _, p = pearsonr(x, yy)
            else:
                _, p = spearmanr(x, yy)
            pvals.append(p)
        res["p_value"] = pvals
        res["q_value_fdr"] = self.bh_fdr(res["p_value"])

        res = (
            res.reset_index()
               .rename(columns={"index": "feature"})
               .sort_values("abs_corr", ascending=False)
               .reset_index(drop=True)
        )
        return res