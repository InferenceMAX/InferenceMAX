import sys
from pathlib import Path


def display_graphs(exp_name, repo, run_id, artifact_name):
    """Display performance graphs in GitHub Actions summary with artifact links."""
    print("\n## Performance Graphs\n")
    
    # Find all generated graphs
    current_dir = Path('.')
    
    # Look for tput_vs_intvty graphs
    intvty_graphs = sorted(current_dir.glob(f'tput_vs_intvty_*_{exp_name}.png'))
    e2el_graphs = sorted(current_dir.glob(f'tput_vs_e2el_*_{exp_name}.png'))
    
    # Construct artifact download URL
    artifact_url = f"https://github.com/{repo}/actions/runs/{run_id}/artifacts"
    
    # Display interactivity graphs
    if intvty_graphs:
        print("### Throughput vs Interactivity\n")
        for graph in intvty_graphs:
            # Extract model name from filename
            model_name = graph.name.replace(f'tput_vs_intvty_', '').replace(f'_{exp_name}.png', '')
            print(f"#### {model_name.upper()}\n")
            print(f"📊 [{graph.name}]({artifact_url}) (download `{artifact_name}` artifact)\n")
    
    # Display end-to-end latency graphs
    if e2el_graphs:
        print("### Throughput vs End-to-End Latency\n")
        for graph in e2el_graphs:
            # Extract model name from filename
            model_name = graph.name.replace(f'tput_vs_e2el_', '').replace(f'_{exp_name}.png', '')
            print(f"#### {model_name.upper()}\n")
            print(f"📊 [{graph.name}]({artifact_url}) (download `{artifact_name}` artifact)\n")
    
    if not intvty_graphs and not e2el_graphs:
        print("*No performance graphs were generated.*\n")


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python3 display_graphs.py <exp_name> <repo> <run_id> <artifact_name>")
        sys.exit(1)
    
    exp_name = sys.argv[1]
    repo = sys.argv[2]
    run_id = sys.argv[3]
    artifact_name = sys.argv[4]
    display_graphs(exp_name, repo, run_id, artifact_name)
