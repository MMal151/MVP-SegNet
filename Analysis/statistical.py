import numpy as np
from scipy.stats import chi2, f_oneway, kruskal, ranksums, ttest_ind, friedmanchisquare, shapiro
from scikit_posthocs import posthoc_nemenyi_friedman, posthoc_conover
from statsmodels.sandbox.stats.multicomp import multipletests

CLASS_NAME = "[DataPreparation/Dataset]"

significance_val = [0.10, 0.05, 0.025, 0.01, 0.005]


# Perform ANOVA
# Input -> metric: Object of Metric class
def do_anova(p1, p2, p3):
    res = f_oneway(p1, p2, p3)
    print(f"ANOVA Results: Statistic: {res}")


def do_kruskal(p1, p2, p3):
    res = kruskal(p1, p2, p3)
    print(f"Kruskal Results: {res}")


def do_wilcoxon(p1, p2, p3):
    res = ranksums(p1, p2, alternative='two-sided')
    print(f"Comparison between group-1 and group-2: {res}")

    res = ranksums(p1, p3, alternative='two-sided')
    print(f"Comparison between group-1 and group-3: {res}")

    res = ranksums(p2, p3, alternative='two-sided')
    print(f"Comparison between group-2 and group-3: {res}")


def do_ttest(p1, p2, p3):
    res = ttest_ind(p1, p2, alternative='two-sided', equal_var=False)
    print(f"Comparison between group-1 and group-2: {res}")

    res = ttest_ind(p1, p3, alternative='two-sided', equal_var=False)
    print(f"Comparison between group-1 and group-3: {res}")

    res = ttest_ind(p2, p3, alternative='two-sided', equal_var=False)
    print(f"Comparison between group-2 and group-3: {res}")


def manage_pvals(p_values):
    significance = np.array([0.1, 0.05, 0.01, 0.99])

    res = multipletests(p_values, significance, method="bonferroni")
    print(f"Corrected: {res}")


# non-parametric test alternative to the one way ANOVA
def do_friedman_test(*groups):
    f_R, p_val = friedmanchisquare(*groups)

    df = len(groups) - 1  # Degree of Freedom
    print(f"F_R = {f_R} ; P-Value = {p_val}")

    for i in significance_val:
        cv = chi2.ppf(i, df)
        if f_R > cv or p_val < i:
            print(f"Rejecting Null-Hypothesis for significance {i} & critical value {cv}")
        else:
            print(f"Accepting Null-Hypothesis for significance {i} & critical value {cv}")


def do_nemenyi(data):
    res = posthoc_nemenyi_friedman(data.T)

    print(f"Nemenyi Results: {res}")


def do_conover(data):
    res = posthoc_conover(data)

    print(f"Conover Results: {res}")


def chk_normality(data):
    res = shapiro(data)

    print(f"shapiro Results: {res}")
