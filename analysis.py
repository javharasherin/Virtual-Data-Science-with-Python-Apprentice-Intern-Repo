import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
FIG = "/home/claude/figs/"

# 1. DATA ACQUISITION
df_raw = pd.read_csv("/home/claude/USvideos.csv")
print("RAW SHAPE:", df_raw.shape)
print(df_raw.dtypes)


# 2. DATA CLEANING
df = df_raw.copy()
df.columns = [c.strip().lower() for c in df.columns]
def parse_dd_mm(x):
    day = int(x)
    month = round((x - day) * 100)
    return day, month

parsed = df["date"].apply(parse_dd_mm)
df["trend_day"] = parsed.apply(lambda t: t[0])
df["trend_month"] = parsed.apply(lambda t: t[1])
df["trend_date"] = pd.to_datetime(
    dict(year=2017, month=df["trend_month"], day=df["trend_day"])
)
df = df.drop(columns=["date", "trend_day", "trend_month"])
df["tags"] = df["tags"].replace("[none]", np.nan)
missing_before = {
    "tags (hidden as '[none]')": (df_raw["tags"] == "[none]").sum(),
}

df["category_id"] = df["category_id"].astype("Int64")
CATEGORY_MAP = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals",
    17: "Sports", 18: "Short Movies", 19: "Travel & Events", 20: "Gaming",
    21: "Videoblogging", 22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
    25: "News & Politics", 26: "Howto & Style", 27: "Education",
    28: "Science & Technology", 29: "Nonprofits & Activism",
}
df["category"] = df["category_id"].map(CATEGORY_MAP)

for col in ["views", "likes", "dislikes", "comment_total"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
exact_dupes = df.duplicated().sum()
df = df.drop_duplicates()
repeat_videos = df.duplicated(subset=["video_id"], keep=False).sum()
df_latest = (
    df.sort_values("trend_date")
      .drop_duplicates(subset="video_id", keep="last")
      .reset_index(drop=True)
)
impossible = df[(df["likes"] > df["views"]) | (df["dislikes"] > df["views"])]
df = df[~((df["likes"] > df["views"]) | (df["dislikes"] > df["views"]))]
df["like_ratio"] = df["likes"] / df["views"]
df["dislike_ratio"] = df["dislikes"] / df["views"]
df["comment_ratio"] = df["comment_total"] / df["views"]
df["engagement_rate"] = (df["likes"] + df["dislikes"] + df["comment_total"]) / df["views"]

print("\nCLEANED SHAPE:", df.shape)
print("Exact duplicate rows removed:", exact_dupes)
print("Rows sharing a video_id (multi-day trends, kept):", repeat_videos)
print("Rows dropped for likes/dislikes > views:", len(impossible))
print("Hidden missing tag values recovered:", missing_before["tags (hidden as '[none]')"])
print("Remaining NaNs:\n", df.isnull().sum())

df.to_csv("/home/claude/USvideos_cleaned.csv", index=False)


# 3. EXPLORATORY DATA ANALYSIS
summary_stats = df[["views", "likes", "dislikes", "comment_total"]].describe()
print("\nSUMMARY STATS:\n", summary_stats)

corr = df[["views", "likes", "dislikes", "comment_total"]].corr()
print("\nCORRELATION MATRIX:\n", corr)

cat_avg_views = (
    df.groupby("category")["views"].mean().sort_values(ascending=False)
)
print("\nAVG VIEWS BY CATEGORY:\n", cat_avg_views)

quality_labels = ["Tags missing\n(hidden as '[none]')", "Zero likes", "Likes > Views\n(dropped)", "Exact duplicate rows\n(dropped)"]
quality_counts = [
    missing_before["tags (hidden as '[none]')"],
    int((df_raw["likes"] == 0).sum()),
    len(impossible),
    int(exact_dupes),
]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(quality_labels, quality_counts, color=["#4C72B0", "#DD8452", "#C44E52", "#55A868"])
ax.set_title("Data Quality Issues Found During Cleaning")
ax.set_ylabel("Number of rows affected")
for b in bars:
    ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(FIG + "fig1_data_quality.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["views"], bins=50, log_scale=True, color="#4C72B0", ax=ax)
ax.set_title("Distribution of Video Views (log scale)")
ax.set_xlabel("Views (log scale)")
ax.set_ylabel("Number of videos")
plt.tight_layout()
plt.savefig(FIG + "fig2_views_distribution.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True, ax=ax)
ax.set_title("Correlation Between Engagement Metrics")
plt.tight_layout()
plt.savefig(FIG + "fig3_correlation_heatmap.png", dpi=150)
plt.close()
fig, ax = plt.subplots(figsize=(9, 6))
cat_avg_views.plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("Average Views by Video Category")
ax.set_xlabel("Average views")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
plt.tight_layout()
plt.savefig(FIG + "fig4_category_avg_views.png", dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(7, 6))
sample = df.sample(min(1500, len(df)), random_state=42)
sns.scatterplot(data=sample, x="views", y="likes", alpha=0.4, ax=ax, color="#4C72B0")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Views vs. Likes (log-log)")
plt.tight_layout()
plt.savefig(FIG + "fig5_views_vs_likes.png", dpi=150)
plt.close()

print("\nAll figures saved to", FIG)
with open("/home/claude/insights.txt", "w") as f:
    f.write(f"raw_shape={df_raw.shape}\n")
    f.write(f"cleaned_shape={df.shape}\n")
    f.write(f"unique_videos={df['video_id'].nunique()}\n")
    f.write(f"date_range={df['trend_date'].min().date()} to {df['trend_date'].max().date()}\n")
    f.write(f"hidden_missing_tags={missing_before['tags (hidden as chr none chr)']}\n" if False else "")
    f.write(f"top_category={cat_avg_views.index[0]}\n")
    f.write(f"top_category_avg_views={cat_avg_views.iloc[0]:.0f}\n")
    f.write(f"corr_likes_views={corr.loc['views','likes']:.3f}\n")
    f.write(f"corr_comments_views={corr.loc['views','comment_total']:.3f}\n")
    f.write(f"corr_dislikes_views={corr.loc['views','dislikes']:.3f}\n")
    f.write(f"mean_like_ratio={df['like_ratio'].mean():.4f}\n")
    f.write(f"mean_engagement_rate={df['engagement_rate'].mean():.4f}\n")
