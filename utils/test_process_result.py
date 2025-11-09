import pytest
import json
import os
import sys
from pathlib import Path

# Import the function to test
sys.path.insert(0, str(Path(__file__).parent))
from process_result import process_benchmark_result


@pytest.fixture
def sample_benchmark_result():
    """Sample benchmark result JSON data."""
    return {
        'max_concurrency': 8,
        'model_id': 'meta-llama/Llama-3-70b',
        'total_token_throughput': 10000.0,
        'output_throughput': 3000.0,
        'ttft_ms': 150.5,
        'tpot_ms': 25.0,
        'e2e_latency_ms': 500.0,
        'decode_tpot_ms': 30.0,
        'prefill_tpot_ms': 20.0
    }


@pytest.fixture
def basic_env_vars():
    """Basic environment variables for testing."""
    return {
        'RUNNER_TYPE': 'h200',
        'TP': '8',
        'EP_SIZE': '1',
        'PREFILL_GPUS': '',
        'DECODE_GPUS': '',
        'DP_ATTENTION': 'false',
        'RESULT_FILENAME': 'test_result',
        'FRAMEWORK': 'vllm',
        'PRECISION': 'fp8',
        'MTP_MODE': ''
    }


def test_basic_processing(sample_benchmark_result, basic_env_vars):
    """Test basic processing of benchmark results."""
    result = process_benchmark_result(sample_benchmark_result, basic_env_vars)
    
    assert result['hw'] == 'h200'
    assert result['tp'] == 8
    assert result['ep'] == 1
    assert result['dp_attention'] == 'false'
    assert result['conc'] == 8
    assert result['model'] == 'meta-llama/Llama-3-70b'
    assert result['framework'] == 'vllm'
    assert result['precision'] == 'fp8'
    assert result['tput_per_gpu'] == 10000.0 / 8
    assert result['output_tput_per_gpu'] == 3000.0 / 8
    assert result['input_tput_per_gpu'] == (10000.0 - 3000.0) / 8


def test_ms_to_seconds_conversion(basic_env_vars):
    """Test conversion of millisecond values to seconds."""
    benchmark_result = {
        'max_concurrency': 4,
        'model_id': 'test/model',
        'total_token_throughput': 5000.0,
        'output_throughput': 1500.0,
        'ttft_ms': 200.0,
        'e2e_latency_ms': 1000.0,
        'decode_latency_ms': 500.0
    }
    
    result = process_benchmark_result(benchmark_result, basic_env_vars)
    
    # Check ms values were converted to seconds
    assert result['ttft'] == 200.0 / 1000.0
    assert result['e2e_latency'] == 1000.0 / 1000.0
    assert result['decode_latency'] == 500.0 / 1000.0


def test_tpot_to_intvty_conversion(basic_env_vars):
    """Test conversion of tpot (time per output token) to intvty (interactivity/throughput)."""
    benchmark_result = {
        'max_concurrency': 2,
        'model_id': 'test/model',
        'total_token_throughput': 2000.0,
        'output_throughput': 500.0,
        'tpot_ms': 25.0,
        'decode_tpot_ms': 20.0,
        'prefill_tpot_ms': 30.0
    }
    
    result = process_benchmark_result(benchmark_result, basic_env_vars)
    
    # Check tpot values were converted to intvty
    # The logic: if 'tpot' in key, convert ms value and then intvty = 1000.0 / tpot_ms
    # So: tpot_ms: 25.0 -> tpot: 0.025 (ms to s), intvty: 1000.0/25.0 = 40.0
    assert result['tpot'] == 25.0 / 1000.0  # Converted from ms to s
    assert result['intvty'] == 1000.0 / 25.0  # intvty = 1000.0 / tpot_ms
    
    assert result['decode_tpot'] == 20.0 / 1000.0
    assert result['decode_intvty'] == 1000.0 / 20.0
    
    assert result['prefill_tpot'] == 30.0 / 1000.0
    assert result['prefill_intvty'] == 1000.0 / 30.0
    
    # Check that the intvty calculation is correct
    assert 'decode_intvty' in result
    assert 'prefill_intvty' in result


def test_mtp_mode_included(sample_benchmark_result, basic_env_vars):
    """Test that MTP mode is included when set."""
    env_vars = basic_env_vars.copy()
    env_vars['MTP_MODE'] = 'disaggregated'
    
    result = process_benchmark_result(sample_benchmark_result, env_vars)
    
    assert 'mtp' in result
    assert result['mtp'] == 'disaggregated'


def test_mtp_mode_not_included(sample_benchmark_result, basic_env_vars):
    """Test that MTP mode is not included when not set."""
    result = process_benchmark_result(sample_benchmark_result, basic_env_vars)
    
    assert 'mtp' not in result


def test_prefill_decode_gpus_explicit(sample_benchmark_result, basic_env_vars):
    """Test explicit prefill and decode GPU counts."""
    env_vars = basic_env_vars.copy()
    env_vars['PREFILL_GPUS'] = '4'
    env_vars['DECODE_GPUS'] = '4'
    
    result = process_benchmark_result(sample_benchmark_result, env_vars)
    
    # With explicit GPU counts
    assert result['output_tput_per_gpu'] == 3000.0 / 4
    assert result['input_tput_per_gpu'] == (10000.0 - 3000.0) / 4


def test_prefill_decode_gpus_defaults_to_tp(sample_benchmark_result, basic_env_vars):
    """Test that prefill/decode GPUs default to TP size when not specified."""
    # Default env vars have empty strings for PREFILL_GPUS and DECODE_GPUS
    result = process_benchmark_result(sample_benchmark_result, basic_env_vars)
    
    # Should use TP size (8) when PREFILL_GPUS and DECODE_GPUS are empty
    assert result['output_tput_per_gpu'] == 3000.0 / 8
    assert result['input_tput_per_gpu'] == (10000.0 - 3000.0) / 8


def test_different_tp_sizes(sample_benchmark_result, basic_env_vars):
    """Test processing with different TP sizes."""
    test_cases = [
        ('1', 1),
        ('2', 2),
        ('4', 4),
        ('8', 8),
        ('16', 16)
    ]
    
    for tp_str, tp_int in test_cases:
        env_vars = basic_env_vars.copy()
        env_vars['TP'] = tp_str
        
        result = process_benchmark_result(sample_benchmark_result, env_vars)
        
        assert result['tp'] == tp_int
        assert result['tput_per_gpu'] == 10000.0 / tp_int


def test_different_ep_sizes(sample_benchmark_result, basic_env_vars):
    """Test processing with different EP sizes."""
    test_cases = [1, 2, 4, 8]
    
    for ep_size in test_cases:
        env_vars = basic_env_vars.copy()
        env_vars['EP_SIZE'] = str(ep_size)
        
        result = process_benchmark_result(sample_benchmark_result, env_vars)
        
        assert result['ep'] == ep_size


def test_output_file_content_structure(sample_benchmark_result, basic_env_vars):
    """Test that output has the expected structure."""
    result = process_benchmark_result(sample_benchmark_result, basic_env_vars)
    
    # Check required fields exist
    required_fields = [
        'hw', 'tp', 'ep', 'dp_attention', 'conc', 'model',
        'framework', 'precision', 'tput_per_gpu', 
        'output_tput_per_gpu', 'input_tput_per_gpu'
    ]
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"


def test_complex_benchmark_result(basic_env_vars):
    """Test processing with a more complex benchmark result."""
    complex_result = {
        'max_concurrency': 16,
        'model_id': 'meta-llama/Llama-3-405b',
        'total_token_throughput': 50000.0,
        'output_throughput': 15000.0,
        'ttft_ms': 100.0,
        'tpot_ms': 15.0,
        'e2e_latency_ms': 2000.0,
        'decode_tpot_ms': 12.0,
        'prefill_tpot_ms': 18.0,
        'p50_latency_ms': 1500.0,
        'p90_latency_ms': 2500.0,
        'p99_latency_ms': 3000.0
    }
    
    result = process_benchmark_result(complex_result, basic_env_vars)
    
    # Check all ms values were converted
    assert result['ttft'] == 100.0 / 1000.0
    assert result['tpot'] == 15.0 / 1000.0
    assert result['e2e_latency'] == 2000.0 / 1000.0
    assert result['p50_latency'] == 1500.0 / 1000.0
    assert result['p90_latency'] == 2500.0 / 1000.0
    assert result['p99_latency'] == 3000.0 / 1000.0
    
    # Check tpot to intvty conversions
    assert 'intvty' in result
    assert 'decode_intvty' in result
    assert 'prefill_intvty' in result


def test_dp_attention_values(sample_benchmark_result, basic_env_vars):
    """Test different DP_ATTENTION values."""
    test_values = ['true', 'false', 'True', 'False']
    
    for dp_attn_value in test_values:
        env_vars = basic_env_vars.copy()
        env_vars['DP_ATTENTION'] = dp_attn_value
        
        result = process_benchmark_result(sample_benchmark_result, env_vars)
        
        assert result['dp_attention'] == dp_attn_value


def test_different_frameworks(sample_benchmark_result, basic_env_vars):
    """Test different framework values."""
    frameworks = ['vllm', 'trt', 'sglang', 'tensorrt-llm']
    
    for framework in frameworks:
        env_vars = basic_env_vars.copy()
        env_vars['FRAMEWORK'] = framework
        
        result = process_benchmark_result(sample_benchmark_result, env_vars)
        
        assert result['framework'] == framework


def test_different_precisions(sample_benchmark_result, basic_env_vars):
    """Test different precision values."""
    precisions = ['fp8', 'fp16', 'fp32', 'int8', 'int4']
    
    for precision in precisions:
        env_vars = basic_env_vars.copy()
        env_vars['PRECISION'] = precision
        
        result = process_benchmark_result(sample_benchmark_result, env_vars)
        
        assert result['precision'] == precision


def test_throughput_calculations(basic_env_vars):
    """Test throughput calculations with various values."""
    benchmark_result = {
        'max_concurrency': 10,
        'model_id': 'test/model',
        'total_token_throughput': 24000.0,
        'output_throughput': 8000.0
    }
    
    env_vars = basic_env_vars.copy()
    env_vars['TP'] = '4'
    env_vars['PREFILL_GPUS'] = '2'
    env_vars['DECODE_GPUS'] = '2'
    
    result = process_benchmark_result(benchmark_result, env_vars)
    
    # tput_per_gpu = total_token_throughput / tp_size
    assert result['tput_per_gpu'] == 24000.0 / 4
    
    # output_tput_per_gpu = output_throughput / decode_gpus
    assert result['output_tput_per_gpu'] == 8000.0 / 2
    
    # input_tput_per_gpu = (total_token_throughput - output_throughput) / prefill_gpus
    assert result['input_tput_per_gpu'] == (24000.0 - 8000.0) / 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
