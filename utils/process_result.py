import sys
import json
import os
from pathlib import Path


def process_benchmark_result(bmk_result, env_vars):
    """
    Process benchmark results and generate aggregated metrics.
    
    Args:
        bmk_result: Dictionary containing benchmark results
        env_vars: Dictionary containing environment variables
        
    Returns:
        Dictionary containing processed metrics
    """
    hw = env_vars.get('RUNNER_TYPE')
    tp_size = int(env_vars.get('TP'))
    ep_size = int(env_vars.get('EP_SIZE'))
    prefill_gpus_str = env_vars.get('PREFILL_GPUS', '')
    decode_gpus_str = env_vars.get('DECODE_GPUS', '')
    
    # If empty string (aggregated runs), assign to tp_size (total gpus), otherwise convert to int
    prefill_gpus = tp_size if not prefill_gpus_str else int(prefill_gpus_str)
    decode_gpus = tp_size if not decode_gpus_str else int(decode_gpus_str)
    dp_attention = env_vars.get('DP_ATTENTION')
    framework = env_vars.get('FRAMEWORK')
    precision = env_vars.get('PRECISION')
    mtp_mode = env_vars.get('MTP_MODE')
    
    data = {
        'hw': hw,
        'tp': tp_size,
        'ep': ep_size,
        'dp_attention': dp_attention,  # true or false
        'conc': int(bmk_result['max_concurrency']),
        'model': bmk_result['model_id'],
        'framework': framework,
        'precision': precision,
        'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp_size,
        'output_tput_per_gpu': float(bmk_result['output_throughput']) / decode_gpus,
        'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / prefill_gpus
    }
    
    if mtp_mode:  # MTP
        data['mtp'] = mtp_mode
    
    for key, value in bmk_result.items():
        if key.endswith('ms'):
            data[key.replace('_ms', '')] = float(value) / 1000.0
        if 'tpot' in key:
            data[key.replace('_ms', '').replace('tpot', 'intvty')] = 1000.0 / float(value)
    
    return data


def main():
    """Main function to process benchmark results from environment variables."""
    result_filename = os.environ.get('RESULT_FILENAME')
    
    with open(f'{result_filename}.json') as f:
        bmk_result = json.load(f)
    
    data = process_benchmark_result(bmk_result, os.environ)
    
    print(json.dumps(data, indent=2))
    
    with open(f'agg_{result_filename}.json', 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
