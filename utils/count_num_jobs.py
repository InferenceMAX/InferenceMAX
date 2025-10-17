import yaml
from collections import defaultdict

with open('.github/configs/search-space.yml', 'r') as f:
    data = yaml.safe_load(f)

gpu_totals = defaultdict(int)
overall_total = 0

for model in data.values():
    for precision in model.values():
        for gpu, runner_data in precision.items():
            for seq_len in runner_data.values():
                for entry in seq_len:
                    # Count TP values
                    tp_list = entry['tp'] if isinstance(entry['tp'], list) else [entry['tp']]
                    tp_count = len(tp_list)
                    
                    # Count CONC values
                    conc = entry['conc']
                    start, end = conc['start'], conc['end']
                    step = conc.get('step', 2)
                    
                    conc_count = 0
                    current = start
                    while current <= end:
                        conc_count += 1
                        current *= step
                    
                    combo_count = tp_count * conc_count
                    gpu_totals[gpu] += combo_count
                    overall_total += combo_count

print("Breakdown by GPU:")
for gpu in sorted(gpu_totals.keys()):
    print(f"  {gpu}: {gpu_totals[gpu]}")
print(f"\nTotal combinations: {overall_total}")