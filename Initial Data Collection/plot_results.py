import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

proto = pd.read_csv('layer_split_prototype_results.csv')
sweep = pd.read_csv('layer_split_sweep_results.csv')
offload = pd.read_csv('offload_results.csv')

print("Prototype:\n", proto.head(), "\n")
print("Sweep:\n", sweep.head(), "\n")
print("Offload:\n", offload.head(), "\n")

# Debug: Show all prompts in each dataset
print("Prototype prompts:")
for i, prompt in enumerate(proto['prompt']):
    print(f"{i+1}: {prompt}")
print("\nOffload prompts:")
for i, prompt in enumerate(offload['prompt']):
    print(f"{i+1}: {prompt}")

# ===== CONSOLIDATED ANALYSIS =====

# 1. Performance Heatmap: CPU layers vs Prompts
plt.figure(figsize=(12, 8))
pivot_data = sweep[sweep['latency_seconds'] > 0].pivot_table(
    values='latency_seconds', 
    index='prompt', 
    columns='cpu_layers', 
    aggfunc='mean'
) * 1000  # Convert to milliseconds

sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Latency (ms)'})
plt.title('Performance Heatmap: Latency by CPU Layers and Prompt')
plt.xlabel('CPU Layers')
plt.ylabel('Prompt')
plt.tight_layout()
plt.show()

# 2. Summary Statistics Table
print("\n=== PERFORMANCE SUMMARY ===")
print("Prototype (50/50 split):")
print(f"  Average latency: {proto['latency_seconds'].mean()*1000:.1f} ms")
print(f"  Std deviation: {proto['latency_seconds'].std()*1000:.1f} ms")
print(f"  Min: {proto['latency_seconds'].min()*1000:.1f} ms")
print(f"  Max: {proto['latency_seconds'].max()*1000:.1f} ms")

print("\nOffload (auto):")
print(f"  Average latency: {offload['latency_seconds'].mean()*1000:.1f} ms")
print(f"  Std deviation: {offload['latency_seconds'].std()*1000:.1f} ms")
print(f"  Min: {offload['latency_seconds'].min()*1000:.1f} ms")
print(f"  Max: {offload['latency_seconds'].max()*1000:.1f} ms")

# 3. Best Configuration Analysis
print("\n=== BEST CONFIGURATIONS ===")
for prompt in sweep['prompt'].unique():
    prompt_data = sweep[(sweep['prompt'] == prompt) & (sweep['latency_seconds'] > 0)]
    if len(prompt_data) > 0:
        best_config = prompt_data.loc[prompt_data['latency_seconds'].idxmin()]
        print(f"{prompt[:50]}...")
        print(f"  Best: {best_config['cpu_layers']} CPU layers ({best_config['latency_seconds']*1000:.1f} ms)")

# 4. Consolidated Performance Comparison
plt.figure(figsize=(14, 8))

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Average latency by CPU layers
avg_by_cpu = sweep[sweep['latency_seconds'] > 0].groupby('cpu_layers')['latency_seconds'].mean() * 1000
ax1.bar(avg_by_cpu.index, avg_by_cpu.values, color='skyblue', alpha=0.7)
ax1.set_xlabel('CPU Layers')
ax1.set_ylabel('Average Latency (ms)')
ax1.set_title('Average Performance by CPU Layer Configuration')
ax1.grid(True, alpha=0.3)

# Subplot 2: Method comparison (box plot)
methods_data = []
methods_labels = []

# Add prototype data
methods_data.extend(proto['latency_seconds'] * 1000)
methods_labels.extend(['Prototype'] * len(proto))

# Add offload data  
methods_data.extend(offload['latency_seconds'] * 1000)
methods_labels.extend(['Offload'] * len(offload))

# Add best sweep data for each prompt
for prompt in sweep['prompt'].unique():
    prompt_data = sweep[(sweep['prompt'] == prompt) & (sweep['latency_seconds'] > 0)]
    if len(prompt_data) > 0:
        best_latency = prompt_data['latency_seconds'].min() * 1000
        methods_data.append(best_latency)
        methods_labels.append('Best Sweep')

ax2.boxplot([methods_data[i:i+len(proto)] for i in range(0, len(methods_data), len(proto))], 
            labels=['Prototype', 'Offload', 'Best Sweep'])
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Method Performance Comparison')
ax2.grid(True, alpha=0.3)

# Subplot 3: Latency distribution
ax3.hist(proto['latency_seconds'] * 1000, alpha=0.7, label='Prototype', bins=10)
ax3.hist(offload['latency_seconds'] * 1000, alpha=0.7, label='Offload', bins=10)
ax3.set_xlabel('Latency (ms)')
ax3.set_ylabel('Frequency')
ax3.set_title('Latency Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Speedup analysis
# Calculate speedup relative to prototype
prototype_avg = proto['latency_seconds'].mean()
speedup_data = []
speedup_labels = []

for cpu_layers in sweep['cpu_layers'].unique():
    layer_data = sweep[(sweep['cpu_layers'] == cpu_layers) & (sweep['latency_seconds'] > 0)]
    if len(layer_data) > 0:
        avg_latency = layer_data['latency_seconds'].mean()
        speedup = prototype_avg / avg_latency
        speedup_data.append(speedup)
        speedup_labels.append(f'{cpu_layers} CPU')

# Add offload speedup
offload_avg = offload['latency_seconds'].mean()
offload_speedup = prototype_avg / offload_avg
speedup_data.append(offload_speedup)
speedup_labels.append('Offload')

ax4.bar(speedup_labels, speedup_data, color='lightgreen', alpha=0.7)
ax4.axhline(y=1, color='red', linestyle='--', label='Prototype baseline')
ax4.set_ylabel('Speedup (higher = faster)')
ax4.set_title('Speedup Relative to Prototype')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 5. Memory vs Performance Trade-off Analysis
print("\n=== MEMORY vs PERFORMANCE TRADE-OFF ===")
print("CPU Layers | Avg Latency (ms) | Relative Memory Usage")
print("-" * 50)
for cpu_layers in sorted(sweep['cpu_layers'].unique()):
    layer_data = sweep[(sweep['cpu_layers'] == cpu_layers) & (sweep['latency_seconds'] > 0)]
    if len(layer_data) > 0:
        avg_latency = layer_data['latency_seconds'].mean() * 1000
        # Estimate memory usage (more CPU layers = less GPU memory)
        memory_usage = (40 - cpu_layers) / 40 * 100  # Assuming 40 total layers
        print(f"{cpu_layers:10d} | {avg_latency:14.1f} | {memory_usage:18.1f}%")

plt.figure(figsize=(8,5))
plt.bar(proto['prompt'], proto['latency_seconds'] * 1000)  # Convert to milliseconds
plt.xticks(rotation=45, ha='right')
plt.ylabel('Latency (ms)')
plt.title('Prototype - Average latency per prompt')
plt.tight_layout()
plt.show()

#Sweep: Latency vs split-point curves:

plt.figure(figsize=(8,5))
for prompt in sweep['prompt'].unique():
    dfp = sweep[sweep['prompt']==prompt]
    # Filter out timeout values (-1)
    dfp_valid = dfp[dfp['latency_seconds'] > 0]
    if len(dfp_valid) > 0:
        plt.plot(dfp_valid['cpu_layers'], dfp_valid['latency_seconds'] * 1000, label=prompt, marker='o')

plt.xlabel('Number of CPU layers')
plt.ylabel('Latency (ms)')
plt.legend()
plt.tight_layout()
plt.show()

#Offload vs Prototype on the same prompt set

common = set(proto['prompt']).intersection(set(offload['prompt']))
print(f"\nCommon prompts found: {len(common)}")
print("Common prompts:", common)

if len(common) > 0:
    p = proto[proto['prompt'].isin(common)].set_index('prompt')
    o = offload[offload['prompt'].isin(common)].set_index('prompt')
    df_cmp = pd.DataFrame({
        'Prototype': p['latency_seconds'] * 1000,  # Convert to milliseconds
        'Offload': o['latency_seconds'] * 1000     # Convert to milliseconds
    })

    df_cmp.plot.bar(figsize=(8,5))
    plt.ylabel('Latency (ms)')
    plt.title('Prototype vs Offload latencies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No common prompts found between prototype and offload datasets.")
    print("Creating separate plots for each dataset...")
    
    # Plot prototype latencies
    plt.figure(figsize=(8,5))
    plt.bar(range(len(proto)), proto['latency_seconds'] * 1000)
    plt.xticks(range(len(proto)), proto['prompt'], rotation=45, ha='right')
    plt.ylabel('Latency (ms)')
    plt.title('Prototype Latencies')
    plt.tight_layout()
    plt.show()
    
    # Plot offload latencies
    plt.figure(figsize=(8,5))
    plt.bar(range(len(offload)), offload['latency_seconds'] * 1000)
    plt.xticks(range(len(offload)), offload['prompt'], rotation=45, ha='right')
    plt.ylabel('Latency (ms)')
    plt.title('Offload Latencies')
    plt.tight_layout()
    plt.show()