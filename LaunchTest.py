import os
import subprocess
import time

def run_test(algo, block, M, N, K):
    env = os.environ.copy()
    env["GEMM_ALGO"] = algo
    env["GEMM_BLOCK"] = str(block)
    env["GEMM_M"] = str(M)
    env["GEMM_N"] = str(N)
    env["GEMM_K"] = str(K)

    start = time.time()
    subprocess.run(["make", "test"], env=env, stdout=subprocess.DEVNULL)
    end = time.time()

    return end - start


if __name__ == "__main__":
    sizes = [128, 256, 512]
    blocks = [16, 32, 64, 128]

    for s in sizes:
        for b in blocks:
            t = run_test("block", b, s, s, s)
            print(f"Size={s}, Block={b} → {t:.4f}s")