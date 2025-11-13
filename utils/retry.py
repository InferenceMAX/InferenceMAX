import argparse
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description="Retry a command multiple times")
parser.add_argument("-a", "--attempts", type=int, default=3, help="Max attempts (default: 3)")
parser.add_argument("-d", "--delay", type=float, default=1, help="Delay in seconds (default: 1)")
parser.add_argument("-c", "--command", required=True, help="Command to execute")
args = parser.parse_args()

for attempt in range(1, args.attempts + 1):
    if attempt > 1:
        print(f"[Retry] Attempt {attempt}/{args.attempts}", file=sys.stderr)
    
    result = subprocess.run(args.command, shell=True)
    
    if result.returncode == 0:
        sys.exit(0)
    
    if attempt < args.attempts:
        print(f"[Retry] Command '{args.command}' failed with exit code {result.returncode}, retrying in {args.delay}s...", file=sys.stderr)
        time.sleep(args.delay)

print(f"[Retry] All {args.attempts} attempts failed", file=sys.stderr)
sys.exit(result.returncode)