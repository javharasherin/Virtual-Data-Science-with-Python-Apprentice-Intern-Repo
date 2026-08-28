import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import squarify

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = "deep"
FIG = "/home/claude/figs2/"
import os
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv("/home/claude/USvideos_cleaned.csv", parse_dates=["trend_date"])

TOP_CATS = df["category"].value_counts().head(8).index.tolist()
df_top = df[df["category"].isin(TOP_CATS)]

# 1.Treemap: share of trending slots by category

cat_counts = df["category"].value_counts().reset_index()
cat_counts.columns = ["category", "count"]
total_slots = cat_counts["count"].sum()
labels = [
    (f"{row.category}\n{row.count} slots ({row.count/total_slots*100:.1f}%)"
     if row.count / total_slots >= 0.02 else row.category)
    for row in cat_counts.itertuples()
]
colors = sns.color_palette("Blues", len(cat_counts))[::-1]
fig, ax = plt.subplots(figsize=(12, 7))
squarify.plot(
    sizes=cat_counts["count"], label=labels, color=colors,
    pad=True, text_kwargs={"fontsize": 10.5}, ax=ax,
)
ax.set_title("Share of Trending Slots by Category (Sep 13\u201324, 2017)", fontsize=15, pad=14)
ax.axis("off")
plt.tight_layout()
plt.savefig(FIG + "chart1_treemap_categories.png", dpi=150)
plt.close()


# 2. Violin plot: view distribution by category
fig, ax = plt.subplots(figsize=(11, 6.5))
order = df_top.groupby("category")["views"].median().sort_values(ascending=False).index
sns.violinplot(
    data=df_top, x="category", y="views", order=order,
    hue="category", legend=False, palette=PALETTE, cut=0, ax=ax
)
ax.set_yscale("log")
ax.set_title("The Long Tail of Virality: View Spread Within Each Category", fontsize=15, pad=14)
ax.set_xlabel("")
ax.set_ylabel("Views (log scale)")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG + "chart2_violin_views_by_category.png", dpi=150)
plt.close()

#3 — Bubble chart: views vs likes, size=comments, color=category

sample = df_top.sample(min(900, len(df_top)), random_state=7).copy()
cat_palette = dict(zip(TOP_CATS, sns.color_palette(PALETTE, len(TOP_CATS))))
sizes = 20 + 900 * (sample["comment_total"] / sample["comment_total"].max())

fig, ax = plt.subplots(figsize=(11.5, 7))
for cat in TOP_CATS:
    sub = sample[sample["category"] == cat]
    ax.scatter(
        sub["views"], sub["likes"],
        s=20 + 900 * (sub["comment_total"] / sample["comment_total"].max()),
        color=cat_palette[cat], alpha=0.55, edgecolor="white", linewidth=0.4,
        label=cat,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Views (log scale)")
ax.set_ylabel("Likes (log scale)")
ax.set_title("What Separates Viral Hits: Views, Likes, and Comment Volume Together\n(bubble size = comment count)", fontsize=14.5, pad=14)
leg = ax.legend(title="Category", loc="upper left", fontsize=9.5, ncol=2, framealpha=0.9)
plt.tight_layout()
plt.savefig(FIG + "chart3_bubble_views_likes_comments.png", dpi=150)
plt.close()


#4 — Grouped bar: engagement quality (like vs dislike rate) by category

eng = (
    df_top.groupby("category")[["like_ratio", "dislike_ratio"]]
    .mean().sort_values("like_ratio", ascending=False)
)
fig, ax = plt.subplots(figsize=(11, 6.5))
x = np.arange(len(eng))
w = 0.38
ax.bar(x - w / 2, eng["like_ratio"] * 100, width=w, label="Like rate", color="#4C72B0")
ax.bar(x + w / 2, eng["dislike_ratio"] * 100, width=w, label="Dislike rate", color="#C44E52")
ax.set_xticks(x)
ax.set_xticklabels(eng.index, rotation=30, ha="right")
ax.set_ylabel("Rate (% of views)")
ax.set_title("Engagement Quality, Not Just Volume: Like vs. Dislike Rate by Category", fontsize=15, pad=14)
ax.legend()
plt.tight_layout()
plt.savefig(FIG + "chart4_engagement_quality_bar.png", dpi=150)
plt.close()


# 5 — Stacked area: daily view volume by top 5 categories over the window

top5 = df["category"].value_counts().head(5).index.tolist()
daily = (
    df[df["category"].isin(top5)]
    .groupby(["trend_date", "category"])["views"].sum()
    .unstack(fill_value=0)[top5]
)
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.stackplot(daily.index, daily.T.values, labels=daily.columns, alpha=0.85,
             colors=sns.color_palette(PALETTE, len(top5)))
ax.set_title("The Two-Week Trending Pulse: Daily View Volume by Category", fontsize=15, pad=14)
ax.set_ylabel("Total daily views")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
ax.legend(loc="upper left", ncol=2, fontsize=10)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(FIG + "chart5_stacked_area_daily_views.png", dpi=150)
plt.close()

#  6 — Heatmap: correlation of raw counts vs normalized engagement ratios
cols = ["views", "likes", "dislikes", "comment_total", "like_ratio", "dislike_ratio", "comment_ratio", "engagement_rate"]
corr2 = df[cols].corr()
fig, ax = plt.subplots(figsize=(9, 7.5))
mask = np.triu(np.ones_like(corr2, dtype=bool), k=1)
sns.heatmap(corr2, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Where Correlation Breaks Down: Raw Counts vs. Normalized Rates", fontsize=15, pad=14)
plt.tight_layout()
plt.savefig(FIG + "chart6_correlation_raw_vs_ratio.png", dpi=150)
plt.close()

print("All 6 charts saved to", FIG)


with open("/home/claude/insights2.txt", "w") as f:
    f.write(f"top_category_share={cat_counts.sort_values('count', ascending=False).iloc[0]['category']}: "
            f"{cat_counts['count'].iloc[cat_counts['count'].idxmax()]}\n")
    f.write(f"entertainment_share_pct={100*cat_counts.set_index('category').loc['Entertainment','count']/cat_counts['count'].sum():.1f}\n")
    f.write(f"music_median_views={df_top[df_top.category=='Music']['views'].median():.0f}\n")
    f.write(f"newspolitics_median_views={df_top[df_top.category=='News & Politics']['views'].median():.0f}\n")
    f.write(f"top_like_rate_cat={eng['like_ratio'].idxmax()}: {eng['like_ratio'].max()*100:.2f}%\n")
    f.write(f"top_dislike_rate_cat={eng['dislike_ratio'].idxmax()}: {eng['dislike_ratio'].max()*100:.2f}%\n")
    f.write(f"corr_views_likes_raw={corr2.loc['views','likes']:.2f}\n")
    f.write(f"corr_views_likeratio={corr2.loc['views','like_ratio']:.2f}\n")
    f.write(daily.sum().to_string() + "\n")
print(open("/home/claude/insights2.txt").read())
