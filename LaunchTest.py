import os
import subprocess

def build(dtype):
    print(f"\n[BUILD] TYPE = {dtype}")
    subprocess.run(
        ["make", "test", f"TYPE={dtype}"],
        check=True
    )

def run(algo, block, M, N, K):
    env = os.environ.copy()
    env["GEMM_ALGO"] = algo
    env["GEMM_BLOCK"] = str(block)
    env["GEMM_M"] = str(M)
    env["GEMM_N"] = str(N)
    env["GEMM_K"] = str(K)

    subprocess.run(
        ["./test.exe"],
        env=env,
        check=True
    )


if __name__ == "__main__":

    types  = ["float", "double"]
    sizes  = [512, 1024, 2048]
    blocks = [16, 32, 64, 128]


    for t in types:
        build(t)

        print(f"\n===== TYPE: {t} =====")

        for s in sizes:
            print(f"\n--- Size {s}x{s} ---")

            # ── Naive
            print("Running naive...")
            run("classic", 0, s, s, s)

            # ── Block
            for b in blocks:
                if b > s:
                    continue

                print(f"Running block {b}...")
                run("block", b, s, s, s)

