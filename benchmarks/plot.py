import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


df = pd.read_csv("results.csv", sep=";")

numeric_cols = ["M", "N", "K", "block", "time_my", "time_blas", "speedup", "max_err"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)

grouped = (
    df.groupby(["type", "algo", "block"], as_index=False)
      .mean(numeric_only=True)
)

algos  = grouped["algo"].unique()
colors = plt.cm.tab10.colors


fig, ax = plt.subplots(figsize=(11, 6))

for i, algo in enumerate(algos):
    subset = grouped[grouped["algo"] == algo]
    ax.plot(subset["block"], subset["time_my"],
            marker="o", color=colors[i], label=f"My GEMM ({algo})", linewidth=2)
    for _, row in subset.iterrows():
        ax.annotate(f'{row["time_my"]*1000:.1f}ms',
                    xy=(row["block"], row["time_my"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color=colors[i])

blas_ref = grouped.groupby("block", as_index=False)["time_blas"].mean()
blas_mean = blas_ref["time_blas"].mean()
ax.axhline(y=blas_mean, linestyle="--", color="black", linewidth=1.5, label=f"OpenBLAS (~{blas_mean*1000:.2f}ms)")

ax.set_xlabel("Block size")
ax.set_ylabel("Execution time (s) — log scale")
ax.set_yscale("log")
ax.set_xticks(grouped["block"].unique())
ax.set_title("GEMM Execution Time (512×512×512, float)")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig("performance.png", dpi=300)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Speedup vs OpenBLAS (My GEMM / OpenBLAS, higher = slower)", fontsize=13)

for i, algo in enumerate(algos):
    subset = grouped[grouped["algo"] == algo]
    ax = axes[i]

    bars = ax.bar(subset["block"].astype(str), subset["speedup"],
                  color=colors[i], alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, subset["speedup"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=1.0, linestyle="--", color="black", linewidth=1, label="OpenBLAS = 1x")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Speedup")
    ax.set_title(f"algo = {algo}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()
fig.savefig("speedup.png", dpi=300)

print("\nDone.")