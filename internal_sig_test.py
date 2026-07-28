import numpy as np
from scipy.stats import spearmanr
from scipy.stats import wilcoxon, ttest_rel


def bootstrap_compare_spearman(
    y_true,
    pred1,
    pred2,
    n_bootstrap=10000,
    random_state=42,
):
    """
    Compare two dependent Spearman correlations using paired bootstrap.

    Returns
    -------
    result : dict
        rho1
        rho2
        delta
        ci_low
        ci_high
        p_value
    """

    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)

    n = len(y_true)

    rho1 = spearmanr(y_true, pred1).correlation
    rho2 = spearmanr(y_true, pred2).correlation

    delta = rho1 - rho2

    boot_delta = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):

        idx = rng.integers(0, n, n)

        r1 = spearmanr(
            y_true[idx],
            pred1[idx]
        ).correlation

        r2 = spearmanr(
            y_true[idx],
            pred2[idx]
        ).correlation

        boot_delta[i] = r1 - r2

    ci_low = np.percentile(boot_delta, 2.5)
    ci_high = np.percentile(boot_delta, 97.5)

    # two-sided bootstrap p-value
    p = 2 * min(
        np.mean(boot_delta <= 0),
        np.mean(boot_delta >= 0),
    )

    return {
        "rho1": rho1,
        "rho2": rho2,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": min(p, 1.0),
    }

def compare_mae(y_true, pred1, pred2):
    err1 = np.abs(y_true - pred1)
    err2 = np.abs(y_true - pred2)

    mae1 = err1.mean()
    mae2 = err2.mean()
    delta = mae1 - mae2  # negativ = pred1 besser

    # traditioneller nichtparametrischer Test
    stat, p_wilcoxon = wilcoxon(err1, err2)

    # klassischer paired t-test
    stat_t, p_ttest = ttest_rel(err1, err2)

    return {
        "mae1": mae1,
        "mae2": mae2,
        "delta_mae": delta,
        "p_wilcoxon": p_wilcoxon,
        "p_paired_ttest": p_ttest,
    }