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
sizes = sorted(df["size"].unique())

colors = {
    512: "red",
    1024: "blue",
    2048: "green"
}

for t in types:
    for size in sizes:

        df_s = df[(df["type"] == t) & (df["size"] == size)]

        if df_s.empty:
            continue

        plt.figure(figsize=(8, 6))

        # ── GEMM BLOCK
        df_block = df_s[df_s["algo"] == "block"].sort_values("block")

        if not df_block.empty:
            plt.plot(
                df_block["block"],
                df_block["time_my"],
                marker="o",
                linestyle="-",
                color=colors.get(size, "black"),
                label="My GEMM (block)"
            )

        # ── OpenBLAS
        df_blas = df_s[df_s["block"] == df_s["block"].max()]

        if not df_blas.empty:
            blas_time = df_blas["time_blas"].values[0]

            plt.hlines(
                y=blas_time,
                xmin=min(df_block["block"]),
                xmax=max(df_block["block"]),
                colors="black",
                linestyles="dashed",
                label="OpenBLAS"
            )

        plt.title(f"{t.upper()} - Size {size}x{size}")
        plt.xlabel("Block size")
        plt.ylabel("Time (seconds)")
        plt.yscale("log")
        plt.ylim(bottom=1e-3)
        plt.grid()
        plt.legend(loc="lower right")

        filename = f"{t}_{size}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"[OK] Generated {filename}")