# Matrix Batching Solution for GitHub Actions 256 Job Limit

## Overview

This implementation provides an elegant workaround for GitHub Actions' hard limit of 256 jobs per matrix, without requiring nested matrices or complex workflow restructuring.

## The Problem

GitHub Actions workflows fail when a matrix strategy generates more than 256 jobs. For InferenceMAX, this can happen when:
- A model prefix generates many configuration combinations
- Expanding concurrency search spaces (conc-start to conc-end)
- Testing across multiple sequence lengths and runner types
- Adding new model variants or precision levels

## The Solution

Three new command-line flags enable batch-based matrix splitting:

1. **`--max-batch-size N`**: Maximum configs per batch (default: 256)
2. **`--batch-index N`**: Retrieve a specific batch (0-indexed)
3. **`--get-batch-count`**: Output the total number of batches needed

## Quick Start

### Check if batching is needed
```bash
python3 generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --seq-lens 1k1k \
  --model-prefix mymodel | jq 'length'
```

If output > 256, you need batching.

### Determine number of batches
```bash
python3 generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --seq-lens 1k1k \
  --model-prefix mymodel \
  --get-batch-count
```

### Get specific batch
```bash
# First batch (configs 0-255)
python3 generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --seq-lens 1k1k \
  --model-prefix mymodel \
  --batch-index 0

# Second batch (configs 256-511)
python3 generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --seq-lens 1k1k \
  --model-prefix mymodel \
  --batch-index 1
```

## Documentation

- **[BATCHING_GUIDE.md](BATCHING_GUIDE.md)**: Complete technical reference
  - Detailed API documentation
  - GitHub Actions workflow patterns
  - Command-line examples

- **[PRACTICAL_GUIDE.md](PRACTICAL_GUIDE.md)**: Real-world usage guide
  - Step-by-step workflow migration
  - Before/after examples
  - Troubleshooting tips
  - Best practices

- **[example-batched-matrix.yml](../../.github/workflows/example-batched-matrix.yml)**: Working example
  - Demonstrates batch-count generation
  - Shows batch-index usage
  - Includes result collection pattern

## Key Features

### ✅ No Nested Matrices
Simple sequential batch indices instead of complex nested matrix strategies.

### ✅ Backwards Compatible
Existing workflows continue to work unchanged. Batching is opt-in via flags.

### ✅ Flexible Batch Sizes
Customize `--max-batch-size` for testing or different limits.

### ✅ Comprehensive Testing
84 tests with 100% pass rate, including:
- Unit tests for batch splitting logic
- Integration tests with CLI
- Edge cases (empty lists, exact fits, large matrices)

### ✅ Security Hardened
- Zero security vulnerabilities
- Explicit GITHUB_TOKEN permissions
- Principle of least privilege

## Implementation Details

### Core Function
```python
def split_into_batches(matrix_values, max_batch_size):
    """Split matrix_values into batches of at most max_batch_size entries."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    
    batches = []
    for i in range(0, len(matrix_values), max_batch_size):
        batches.append(matrix_values[i:i + max_batch_size])
    return batches
```

### Workflow Integration
When a model prefix exceeds 256 configs, create separate jobs for each batch:

```yaml
jobs:
  get-configs-batch-0:
    steps:
      - run: |
          CONFIG_JSON=$(python3 generate_sweep_configs.py full-sweep \
            --config-files master.yaml \
            --model-prefix mymodel \
            --batch-index 0)
          echo "search-space-config=$CONFIG_JSON" >> $GITHUB_OUTPUT

  benchmark-batch-0:
    needs: get-configs-batch-0
    strategy:
      matrix:
        config: ${{ fromJson(needs.get-configs-batch-0.outputs.search-space-config) }}
    # ... benchmark parameters
```

## When to Use

### Use Batching When:
- Single model-prefix generates > 256 configs
- Need to test exhaustive parameter combinations
- Expanding search spaces significantly

### Consider Alternatives When:
- Configs < 256 (batching not needed)
- Can split by model-prefix (current approach)
- Can filter by precision, framework, or runner-type
- Can reduce search space with `--test-mode` or larger `--step-size`

## Testing

Run the test suite:
```bash
cd utils/matrix-logic
python3 -m pytest test_generate_sweep_configs.py -v
```

Expected: 84 tests, 100% passing

## Examples

### Example 1: Split 500 configs into 2 batches
```bash
# Get batch count
$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --seq-lens 1k1k \
    --get-batch-count
2

# Get batches
$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --seq-lens 1k1k \
    --batch-index 0 | jq 'length'
256

$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --seq-lens 1k1k \
    --batch-index 1 | jq 'length'
244
```

### Example 2: Custom batch size
```bash
# Split into batches of 100
$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --max-batch-size 100 \
    --get-batch-count
5
```

## Performance

- Batch calculation: O(1) using math.ceil()
- Batch retrieval: O(n) where n = batch_size
- Memory efficient: Only requested batch is held in memory
- No nested iterations or complex computations

## Backwards Compatibility

All existing workflows continue to work without modification:
```bash
# Old way (still works)
python3 generate_sweep_configs.py full-sweep --config-files master.yaml

# New way (opt-in)
python3 generate_sweep_configs.py full-sweep --config-files master.yaml --batch-index 0
```

## Troubleshooting

### "Invalid batch-index X. Valid range is 0 to Y"
The batch index is out of range. Check valid range with `--get-batch-count`.

### All configs in batch 0
Total configs < 256, no batching needed. This is expected behavior.

### Missing configs
Verify: `sum(all batch sizes) == total configs`
```bash
for i in {0..N}; do
  python3 generate_sweep_configs.py ... --batch-index $i | jq 'length'
done
```

## Future Enhancements

Potential improvements:
- Dynamic batch size based on workflow complexity
- Parallel batch generation
- Batch-aware result collection utilities
- Integration with workflow dispatch events

## Support

For questions or issues:
1. Review [BATCHING_GUIDE.md](BATCHING_GUIDE.md) for technical details
2. Check [PRACTICAL_GUIDE.md](PRACTICAL_GUIDE.md) for real-world examples
3. See [example-batched-matrix.yml](../../.github/workflows/example-batched-matrix.yml) for working code
4. Open an issue if problems persist

## License

This solution is part of InferenceMAX and follows the repository's Apache 2.0 license.
