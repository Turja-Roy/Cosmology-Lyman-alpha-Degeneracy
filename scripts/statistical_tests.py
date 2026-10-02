import numpy as np
from scipy import stats


def kolmogorov_smirnov_test(data1, data2):
    """KS test for distribution differences."""
    statistic, pvalue = stats.ks_2samp(data1.flatten(), data2.flatten())
    return {'statistic': float(statistic), 'pvalue': float(pvalue)}


def uniformity_test(values, lo=0.0, hi=100.0):
    """One-sample KS against a uniform distribution.

    Where each observed chunk falls inside its own simulated distribution is uniform on
    (lo, hi) if the simulation is right. Testing that is the small-sample question; pooling
    the observed values against every mock instead mixes redshifts and noise levels.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return {'statistic': float('nan'), 'pvalue': float('nan'), 'n': int(v.size)}
    statistic, pvalue = stats.kstest((v - lo) / (hi - lo), 'uniform')
    return {'statistic': float(statistic), 'pvalue': float(pvalue), 'n': int(v.size)}


def mann_whitney_test(data1, data2):
    """Mann-Whitney U test for median differences."""
    statistic, pvalue = stats.mannwhitneyu(data1.flatten(), data2.flatten(), alternative='two-sided')
    return {'statistic': float(statistic), 'pvalue': float(pvalue)}


def levene_test(data1, data2):
    """Levene's test for variance equality."""
    statistic, pvalue = stats.levene(data1.flatten(), data2.flatten())
    return {'statistic': float(statistic), 'pvalue': float(pvalue)}


def cohens_d(data1, data2):
    """Cohen's d effect size."""
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = data1.size, data2.size
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return float((mean1 - mean2) / pooled_std) if pooled_std > 0 else 0.0


def chi_squared_test(observed, expected):
    """Chi-squared goodness of fit test."""
    mask = expected > 0
    statistic, pvalue = stats.chisquare(observed[mask], expected[mask])
    return {'statistic': float(statistic), 'pvalue': float(pvalue)}


def anderson_darling_test(data1, data2):
    """Anderson-Darling test for distributions."""
    combined = np.concatenate([data1.flatten(), data2.flatten()])
    result = stats.anderson_ksamp([data1.flatten(), data2.flatten()])
    return {'statistic': float(result.statistic), 'pvalue': float(result.pvalue) if result.pvalue is not None else np.nan}


def permutation_test(data1, data2, n_permutations=1000, statistic_func=np.mean):
    """Permutation test for any statistic."""
    observed_diff = statistic_func(data1) - statistic_func(data2)
    combined = np.concatenate([data1.flatten(), data2.flatten()])
    n1 = data1.size
    
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = statistic_func(combined[:n1]) - statistic_func(combined[n1:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    
    pvalue = count_extreme / n_permutations
    return {'observed_diff': float(observed_diff), 'pvalue': float(pvalue)}


def comprehensive_comparison(data1, data2, label1='Data1', label2='Data2'):
    """Run full battery of statistical tests."""
    results = {
        'labels': (label1, label2),
        'n_samples': (data1.size, data2.size),
        'means': (float(np.mean(data1)), float(np.mean(data2))),
        'medians': (float(np.median(data1)), float(np.median(data2))),
        'stds': (float(np.std(data1)), float(np.std(data2))),
        'ks_test': kolmogorov_smirnov_test(data1, data2),
        'mw_test': mann_whitney_test(data1, data2),
        'levene_test': levene_test(data1, data2),
        'cohens_d': cohens_d(data1, data2),
        'anderson_darling': anderson_darling_test(data1, data2),
    }
    
    return results


def bonferroni_correction(pvalues, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons."""
    n_tests = len(pvalues)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in pvalues]
    return corrected_alpha, significant


def benjamini_hochberg_correction(pvalues, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    n_tests = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = np.array(pvalues)[sorted_indices]
    
    thresholds = (np.arange(1, n_tests + 1) / n_tests) * alpha
    significant_sorted = sorted_pvalues <= thresholds
    
    if not np.any(significant_sorted):
        return []
    
    max_k = np.where(significant_sorted)[0][-1]
    significant = np.zeros(n_tests, dtype=bool)
    significant[sorted_indices[:max_k + 1]] = True
    
    return significant.tolist()


def format_test_results_table(results_list, correction='bonferroni', alpha=0.05):
    """Format multiple comparison results as text table."""
    n_comparisons = len(results_list)
    
    all_pvalues = []
    for res in results_list:
        all_pvalues.extend([res['ks_test']['pvalue'], res['mw_test']['pvalue'], 
                           res['levene_test']['pvalue']])
    
    if correction == 'bonferroni':
        corrected_alpha, _ = bonferroni_correction(all_pvalues, alpha)
    else:
        significant_flags = benjamini_hochberg_correction(all_pvalues, alpha)
    
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("STATISTICAL COMPARISON RESULTS")
    lines.append("=" * 100)
    
    for i, res in enumerate(results_list):
        lines.append(f"\n[{i+1}] {res['labels'][0]} vs {res['labels'][1]}")
        lines.append("-" * 100)
        lines.append(f"  Samples:      n1={res['n_samples'][0]:,}  n2={res['n_samples'][1]:,}")
        lines.append(f"  Means:        {res['means'][0]:.6f}  vs  {res['means'][1]:.6f}  (diff: {res['means'][1]-res['means'][0]:+.6f})")
        lines.append(f"  Medians:      {res['medians'][0]:.6f}  vs  {res['medians'][1]:.6f}  (diff: {res['medians'][1]-res['medians'][0]:+.6f})")
        lines.append(f"  Std devs:     {res['stds'][0]:.6f}  vs  {res['stds'][1]:.6f}")
        lines.append(f"  Cohen's d:    {res['cohens_d']:+.4f}  (effect size)")
        lines.append(f"\n  Tests:")
        
        ks_sig = '*' if res['ks_test']['pvalue'] < corrected_alpha else ''
        mw_sig = '*' if res['mw_test']['pvalue'] < corrected_alpha else ''
        lev_sig = '*' if res['levene_test']['pvalue'] < corrected_alpha else ''
        
        lines.append(f"    KS test:           p={res['ks_test']['pvalue']:.4e}  {ks_sig}")
        lines.append(f"    Mann-Whitney:      p={res['mw_test']['pvalue']:.4e}  {mw_sig}")
        lines.append(f"    Levene:            p={res['levene_test']['pvalue']:.4e}  {lev_sig}")
        lines.append(f"    Anderson-Darling:  p={res['anderson_darling']['pvalue']:.4e}")
    
    lines.append("\n" + "=" * 100)
    lines.append(f"Significance level: α={corrected_alpha:.4e} ({correction} correction)")
    lines.append("* indicates statistically significant difference")
    lines.append("=" * 100 + "\n")
    
    return "\n".join(lines)


def pairwise_comparison_matrix(data_list, labels, metric='ks'):
    """Create matrix of pairwise test statistics."""
    n = len(data_list)
    matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'ks':
                result = kolmogorov_smirnov_test(data_list[i], data_list[j])
            elif metric == 'mw':
                result = mann_whitney_test(data_list[i], data_list[j])
            elif metric == 'cohens_d':
                d = cohens_d(data_list[i], data_list[j])
                matrix[i, j] = matrix[j, i] = d
                continue
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            matrix[i, j] = matrix[j, i] = result['statistic']
            pvalue_matrix[i, j] = pvalue_matrix[j, i] = result['pvalue']
    
    return {
        'matrix': matrix,
        'pvalue_matrix': pvalue_matrix,
        'labels': labels,
        'metric': metric
    }
