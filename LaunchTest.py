import os
import subprocess
import time

def run_test(algo, block, M, N, K, dtype):
    env = os.environ.copy()
    env["GEMM_ALGO"] = algo
    env["GEMM_BLOCK"] = str(block)
    env["GEMM_M"] = str(M)
    env["GEMM_N"] = str(N)
    env["GEMM_K"] = str(K)

    start = time.time()
    subprocess.run(
        ["make", "test", f"TYPE={dtype}"],
        env=env,
        stdout=subprocess.DEVNULL
    )
    end = time.time()

    return end - start


if __name__ == "__main__":
    types = ["float", "double"]
    sizes = [512, 1024, 2048]
    blocks = [16, 32, 64, 128]

with open("CPU_outputResults.txt", "w") as out:
    out.write("Type\tSize\tBlockSize\tTime\tSpeedup\n")

    for t in types:
        print(f"\n===== TYPE: {t} =====")
        
        for s in sizes:
            t_naive = run_test("classic", 0, s, s, s, t)
            
            print(f"\n--- Size {s}x{s} (Naive: {t_naive:.4f}s) ---")
            out.write(f"{t}\t{s}\t0\t{t_naive:.6f}\t1.0\n")

            for b in blocks:
                if b > s: continue
                
                t_block = run_test("block", b, s, s, s, t)
                speedup = t_naive / t_block

                print(f"Block {b:3d} → {t_block:.4f}s | speedup x{speedup:.2f}")
                
                out.write(f"{t}\t{s}\t{b}\t{t_block:.6f}\t{speedup:.2f}\n")
                
                out.flush()