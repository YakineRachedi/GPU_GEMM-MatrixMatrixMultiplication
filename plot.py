import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results.csv",
    sep=";",
    header=None,
    names=["type", "size", "block", "algo", "time_my", "time_blas"]
)

df["size"] = df["size"].astype(int)
df["block"] = df["block"].astype(int)
df["time_my"] = df["time_my"].astype(float)
df["time_blas"] = df["time_blas"].astype(float)

types = df["type"].unique()

colors = {
    512: "red",
    1024: "blue",
    2048: "green"
}

fig, axes = plt.subplots(1, len(types), figsize=(14, 6), sharey=True)

if len(types) == 1:
    axes = [axes]

# ── Loop float / double ───────────────────
for ax, t in zip(axes, types):
    df_t = df[df["type"] == t]

    for size in sorted(df_t["size"].unique()):
        df_s = df_t[df_t["size"] == size]

        # ── GEMM BLOCK
        df_block = df_s[df_s["algo"] == "block"].sort_values("block")

        if not df_block.empty:
            ax.plot(
                df_block["block"],
                df_block["time_my"],
                marker="o",
                color=colors.get(size, None),
                linestyle="-",
                label=f"GEMM {size}"
            )

        # ── OpenBLAS (ligne horizontale)
        df_blas = df_s[df_s["block"] == df_s["block"].max()]

        if not df_blas.empty:
            blas_time = df_blas["time_blas"].values[0]

            ax.hlines(
                y=blas_time,
                xmin=min(df_s["block"]),
                xmax=max(df_s["block"]),
                colors=colors.get(size, None),
                linestyles="dashed",
                label=f"OpenBLAS {size}"
            )

    ax.set_title(f"Type: {t}")
    ax.set_xlabel("Block size")
    ax.set_yscale("log")
    ax.grid()

axes[0].set_ylabel("Time (seconds)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.90])

plt.savefig("comparison.png", dpi=300)
plt.show()