import matplotlib.pyplot as plt
import csv
 
# --- Read data from file ---
data = {}  # { dtype: {'sizes': [], 'time': [], 'speedup': []} }
 
with open('CPU_outputResults.txt', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        dtype     = row['Type'].strip()
        blocksize = int(row['BlockSize'].strip())
        size      = int(row['Size'].strip())
        time      = float(row['Time'].strip())
        speedup   = float(row['Speedup'].strip())
 
        # Keep only float/double at BlockSize=64
        if dtype not in ('float', 'double') or blocksize != 64:
            continue
 
        if dtype not in data:
            data[dtype] = {'sizes': [], 'time': [], 'speedup': []}
        data[dtype]['sizes'].append(size)
        data[dtype]['time'].append(time)
        data[dtype]['speedup'].append(speedup)
 
if not data:
    raise ValueError("There is no data for float/double with BlockSize=64.")
 
# --- Plot ---
colors  = {'float': '#4C9BE8', 'double': '#E8734C'}
markers = {'float': 'o',       'double': 's'}
 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#0F1117')
 
for ax in axes:
    ax.set_facecolor('#181B24')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#3A3F52')
    ax.tick_params(colors='#A0A8C0', labelsize=10)
    ax.xaxis.label.set_color('#C8D0E8')
    ax.yaxis.label.set_color('#C8D0E8')
    ax.title.set_color('#E8EAF4')
    ax.grid(color='#2A2F40', linestyle='--', linewidth=0.7, alpha=0.8)
 
# Plot 1: Time
ax = axes[0]
for dtype, d in data.items():
    ax.plot(d['sizes'], d['time'],
            color=colors[dtype], marker=markers[dtype],
            linewidth=2, markersize=8, label=dtype)
    for x, y in zip(d['sizes'], d['time']):
        ax.annotate(f'{y:.3f}', (x, y),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8.5, color=colors[dtype])
 
ax.set_title('Execution Time  —  BlockSize = 64', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Matrix Size', fontsize=11)
ax.set_ylabel('Time (s)', fontsize=11)
sizes_all = sorted({s for d in data.values() for s in d['sizes']})
ax.set_xticks(sizes_all)
ax.legend(facecolor='#1E2230', edgecolor='#3A3F52', labelcolor='#C8D0E8', fontsize=10)
 
# Plot 2: Speedup
ax = axes[1]
ax.axhline(1.0, color='#888', linestyle=':', linewidth=1.2, label='baseline (speedup=1)')
for dtype, d in data.items():
    ax.plot(d['sizes'], d['speedup'],
            color=colors[dtype], marker=markers[dtype],
            linewidth=2, markersize=8, label=dtype)
    for x, y in zip(d['sizes'], d['speedup']):
        ax.annotate(f'{y:.2f}', (x, y),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8.5, color=colors[dtype])
 
ax.set_title('Speedup  —  BlockSize = 64', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Matrix Size', fontsize=11)
ax.set_ylabel('Speedup', fontsize=11)
ax.set_xticks(sizes_all)
ax.legend(facecolor='#1E2230', edgecolor='#3A3F52', labelcolor='#C8D0E8', fontsize=10)
 
fig.suptitle('float  vs  double  ·  BlockSize = 64', fontsize=15,
             fontweight='bold', color='#E8EAF4', y=0.95)
 
plt.tight_layout()
plt.savefig('plot_blocksize64.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Plot saved → plot_blocksize64.png")