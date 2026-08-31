"""
Week 3 Task - Statistical Analysis and Hypothesis Testing
Continuing on the Week 1/2 cleaned "Trending YouTube Video Statistics" (US) dataset.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from itertools import combinations

sns.set_theme(style="whitegrid", font_scale=1.05)
FIG = "/home/claude/figs3/"
os.makedirs(FIG, exist_ok=True)
RESULTS = {}

df = pd.read_csv("/home/claude/USvideos_cleaned.csv", parse_dates=["trend_date"])
df["log_views"] = np.log10(df["views"])
TOP_CATS = df["category"].value_counts().head(8).index.tolist()
df_top = df[df["category"].isin(TOP_CATS)].copy()

print("=" * 70)
print("TEST 1: Music vs. Non-Music views (two-group comparison)")
print("=" * 70)

# TEST 1 — Two-sample comparison
# H0: mean(log_views) Music == mean(log_views) Non-Music
# H1: mean(log_views) Music > mean(log_views) Non-Music   (one-sided)

music = df.loc[df["category"] == "Music", "log_views"]
non_music = df.loc[df["category"] != "Music", "log_views"]

# Assumption checks
shapiro_music = stats.shapiro(music.sample(min(500, len(music)), random_state=1))
shapiro_nonmusic = stats.shapiro(non_music.sample(min(500, len(non_music)), random_state=1))
levene_1 = stats.levene(music, non_music)

# Welch's t-test (does not assume equal variance) on log-transformed views
t_stat, t_p_two = stats.ttest_ind(music, non_music, equal_var=False)
t_p_one = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2

# Effect size: Cohen's d (Welch version, pooled via avg variance)
pooled_sd = np.sqrt((music.var(ddof=1) + non_music.var(ddof=1)) / 2)
cohens_d = (music.mean() - non_music.mean()) / pooled_sd

# 95% CI for the mean difference (Welch)
diff = music.mean() - non_music.mean()
se = np.sqrt(music.var(ddof=1) / len(music) + non_music.var(ddof=1) / len(non_music))
dof_welch = ((music.var(ddof=1) / len(music) + non_music.var(ddof=1) / len(non_music)) ** 2) / (
    (music.var(ddof=1) / len(music)) ** 2 / (len(music) - 1)
    + (non_music.var(ddof=1) / len(non_music)) ** 2 / (len(non_music) - 1)
)
t_crit = stats.t.ppf(0.975, dof_welch)
ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

# Non-parametric robustness check: Mann-Whitney U (one-sided, Music > Non-Music)
u_stat, u_p = stats.mannwhitneyu(music, non_music, alternative="greater")
n1, n2 = len(music), len(non_music)
rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

RESULTS["test1"] = dict(
    n_music=n1, n_nonmusic=n2,
    shapiro_music_p=shapiro_music.pvalue, shapiro_nonmusic_p=shapiro_nonmusic.pvalue,
    levene_p=levene_1.pvalue,
    mean_log_music=music.mean(), mean_log_nonmusic=non_music.mean(),
    t_stat=t_stat, t_p_one_sided=t_p_one, dof_welch=dof_welch,
    cohens_d=cohens_d, diff=diff, ci_low=ci_low, ci_high=ci_high,
    u_stat=u_stat, u_p_one_sided=u_p, rank_biserial=rank_biserial,
)
for k, v in RESULTS["test1"].items():
    print(f"  {k}: {v}")

# Fig 1: distribution comparison + box plot
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
sns.kdeplot(music, fill=True, label="Music", ax=axes[0], color="#DD8452")
sns.kdeplot(non_music, fill=True, label="Non-Music", ax=axes[0], color="#4C72B0")
axes[0].set_xlabel("log10(views)")
axes[0].set_title("Distribution: Music vs. Non-Music")
axes[0].legend()

box_df = df.copy()
box_df["group"] = np.where(box_df["category"] == "Music", "Music", "Non-Music")
sns.boxplot(data=box_df, x="group", y="log_views", hue="group", legend=False,
            palette={"Music": "#DD8452", "Non-Music": "#4C72B0"}, ax=axes[1])
axes[1].set_ylabel("log10(views)")
axes[1].set_title(f"Welch t={t_stat:.2f}, one-sided p={t_p_one:.2e}")
plt.tight_layout()
plt.savefig(FIG + "test1_music_vs_nonmusic.png", dpi=150)
plt.close()

print("\n" + "=" * 70)
print("TEST 2: ANOVA across top 8 categories (log views)")
print("=" * 70)

# TEST 2 — k-group comparison

groups = [df_top.loc[df_top["category"] == c, "log_views"].values for c in TOP_CATS]
levene_2 = stats.levene(*groups)
f_stat, anova_p = stats.f_oneway(*groups)

# Effect size: eta-squared
grand_mean = df_top["log_views"].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_total = ((df_top["log_views"] - grand_mean) ** 2).sum()
eta_sq = ss_between / ss_total

# Kruskal-Wallis (nonparametric robustness check — variances differ across groups)
h_stat, kw_p = stats.kruskal(*groups)

RESULTS["test2"] = dict(
    levene_p=levene_2.pvalue, f_stat=f_stat, anova_p=anova_p, eta_sq=eta_sq,
    h_stat=h_stat, kw_p=kw_p,
)
for k, v in RESULTS["test2"].items():
    print(f"  {k}: {v}")

# Tukey HSD post-hoc
tukey = pairwise_tukeyhsd(df_top["log_views"], df_top["category"], alpha=0.05)
tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
tukey_df.to_csv("/home/claude/tukey_results.csv", index=False)
n_sig_pairs = (tukey_df["reject"] == True).sum()
print(f"  Significant pairwise Tukey comparisons: {n_sig_pairs} of {len(tukey_df)}")

# Fig 2: boxplot across categories ordered by median, annotated with ANOVA/KW result
order2 = df_top.groupby("category")["log_views"].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(11.5, 6.5))
sns.boxplot(data=df_top, x="category", y="log_views", order=order2,
            hue="category", legend=False, palette="deep", ax=ax)
ax.set_xlabel("")
ax.set_ylabel("log10(views)")
ax.set_title(f"One-way ANOVA: F={f_stat:.2f}, p={anova_p:.2e}  |  Kruskal-Wallis: H={h_stat:.2f}, p={kw_p:.2e}",
             fontsize=12.5)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG + "test2_anova_categories.png", dpi=150)
plt.close()

# Fig 3: Tukey HSD pairwise mean-difference plot
fig, ax = plt.subplots(figsize=(9, 7))
tukey.plot_simultaneous(ax=ax)
ax.set_title("Tukey HSD: 95% CI for Pairwise Mean Differences (log10 views)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG + "test3_tukey_hsd.png", dpi=150)
plt.close()

print("\n" + "=" * 70)
print("TEST 3: Chi-square test of independence (category x engagement tier)")
print("=" * 70)


# TEST 3 — Association between category and engagement tier
# H0: engagement tier (high/low like_ratio) is independent of category
# H1: engagement tier depends on category

top5 = df["category"].value_counts().head(5).index.tolist()
df5 = df[df["category"].isin(top5)].copy()
median_like_ratio = df5["like_ratio"].median()
df5["engagement_tier"] = np.where(df5["like_ratio"] >= median_like_ratio, "High like rate", "Low like rate")

contingency = pd.crosstab(df5["category"], df5["engagement_tier"])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
n_total = contingency.values.sum()
cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))

RESULTS["test3"] = dict(
    median_like_ratio=median_like_ratio, chi2=chi2, dof=dof, chi_p=chi_p,
    cramers_v=cramers_v, n_total=n_total,
)
for k, v in RESULTS["test3"].items():
    print(f"  {k}: {v}")
print(contingency)

# Fig 4: stacked proportion bar + heatmap of standardized residuals
prop = contingency.div(contingency.sum(axis=1), axis=0)
resid = (contingency - expected) / np.sqrt(expected)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
prop.plot(kind="barh", stacked=True, ax=axes[0], color=["#4C72B0", "#DD8452"])
axes[0].set_xlabel("Proportion of videos")
axes[0].set_title("Engagement Tier Share by Category")
axes[0].legend(loc="lower right", fontsize=9)

sns.heatmap(resid, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1],
            cbar_kws={"label": "Standardized residual"})
axes[1].set_title(f"Chi-square residuals (\u03c7\u00b2={chi2:.1f}, p={chi_p:.1e})")
plt.tight_layout()
plt.savefig(FIG + "test4_chisquare_category_engagement.png", dpi=150)
plt.close()

print("\n" + "=" * 70)
print("TEST 4: Correlation significance test (views vs likes)")
print("=" * 70)

# TEST 4 — Correlation hypothesis test
# H0: rho(views, likes) = 0
# H1: rho(views, likes) != 0
pearson_r, pearson_p = stats.pearsonr(df["log_views"], np.log10(df["likes"].replace(0, 1)))
spearman_r, spearman_p = stats.spearmanr(df["views"], df["likes"])

# 95% CI for Pearson r via Fisher z-transform
n = len(df)
z = np.arctanh(pearson_r)
se_z = 1 / np.sqrt(n - 3)
z_crit = stats.norm.ppf(0.975)
ci_r_low, ci_r_high = np.tanh(z - z_crit * se_z), np.tanh(z + z_crit * se_z)

RESULTS["test4"] = dict(
    pearson_r=pearson_r, pearson_p=pearson_p, ci_r_low=ci_r_low, ci_r_high=ci_r_high,
    spearman_r=spearman_r, spearman_p=spearman_p, n=n,
)
for k, v in RESULTS["test4"].items():
    print(f"  {k}: {v}")

# Fig 5: scatter with regression line + CI band (log-log)
fig, ax = plt.subplots(figsize=(8.5, 6.5))
sample = df.sample(min(1200, len(df)), random_state=3)
sns.regplot(
    data=sample, x="log_views", y=np.log10(sample["likes"].replace(0, 1)),
    scatter_kws={"alpha": 0.35, "s": 22, "color": "#4C72B0"},
    line_kws={"color": "#C44E52"}, ax=ax,
)
ax.set_xlabel("log10(views)")
ax.set_ylabel("log10(likes)")
ax.set_title(f"Pearson r={pearson_r:.3f}, 95% CI [{ci_r_low:.3f}, {ci_r_high:.3f}], p<0.001", fontsize=13)
plt.tight_layout()
plt.savefig(FIG + "test5_correlation_views_likes.png", dpi=150)
plt.close()

print("\nAll figures saved to", FIG)

import json
with open("/home/claude/stats_results.json", "w") as f:
    json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                    for kk, vv in v.items()} for k, v in RESULTS.items()}, f, indent=2)
print("\nSaved stats_results.json")
