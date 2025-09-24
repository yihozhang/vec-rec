import subprocess
import re
import matplotlib.pyplot as plt

# Change these to your actual binary paths
binary1 = "./test"
binary2 = "./test_dd"

runs = 20
pattern = re.compile(r"Vectorized throughput:\s+([0-9.]+)\s+giga-floats per second")

def run_and_capture(binary, runs=20):
    throughputs = []
    for i in range(runs):
        result = subprocess.run([binary], capture_output=True, text=True)
        match = pattern.search(result.stdout)
        if match:
            value = float(match.group(1))
            throughputs.append(value)
        else:
            print(f"Warning: no throughput line found in run {i+1} for {binary}")
    return throughputs

print("Running first binary...")
data1 = run_and_capture(binary1, runs)
print("Running second binary...")
data2 = run_and_capture(binary2, runs)

# Plot distributions
plt.figure(figsize=(8,6))
plt.boxplot([data1, data2], labels=["Binary 1", "Binary 2"])
plt.ylabel("Throughput (GFLOP/s)")
plt.title("Distribution of Vectorized Throughput (20 runs each)")
plt.grid(True, axis="y")

plt.savefig("distribution.png")
