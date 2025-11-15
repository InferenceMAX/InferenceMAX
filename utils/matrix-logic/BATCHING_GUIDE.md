# Matrix Batching Guide

## Overview

GitHub Actions has a hard limit of 256 jobs per matrix. To work around this limitation, the `generate_sweep_configs.py` script now supports splitting large configuration sets into multiple batches.

## Key Features

### 1. Automatic Batching
- Configurations are automatically split into batches when the total exceeds the batch size
- Default batch size is 256 (GitHub Actions limit)
- Each batch contains at most `max-batch-size` configurations

### 2. Get Batch Count
Use `--get-batch-count` to determine how many batches are needed:

```bash
python3 generate_sweep_configs.py full-sweep \
  --config-files config1.yaml config2.yaml \
  --seq-lens 1k1k \
  --get-batch-count
```

This outputs a single integer representing the number of batches needed.

### 3. Retrieve Specific Batch
Use `--batch-index` to get a specific batch (0-indexed):

```bash
python3 generate_sweep_configs.py full-sweep \
  --config-files config1.yaml config2.yaml \
  --seq-lens 1k1k \
  --batch-index 0
```

### 4. Custom Batch Size
Override the default batch size with `--max-batch-size`:

```bash
python3 generate_sweep_configs.py full-sweep \
  --config-files config1.yaml config2.yaml \
  --seq-lens 1k1k \
  --max-batch-size 100 \
  --batch-index 0
```

## Usage in GitHub Actions Workflows

### Example: Dynamic Batch Generation

Here's how to use batching in a workflow:

```yaml
jobs:
  get-batch-count:
    runs-on: ubuntu-latest
    outputs:
      batch-count: ${{ steps.count.outputs.batch-count }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - id: count
        run: |
          pip install pydantic
          BATCH_COUNT=$(python3 utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files .github/configs/master.yaml \
            --seq-lens 1k1k \
            --get-batch-count)
          echo "batch-count=$BATCH_COUNT" >> $GITHUB_OUTPUT

  get-batch-configs:
    needs: get-batch-count
    runs-on: ubuntu-latest
    strategy:
      matrix:
        batch-index: ${{ range(0, fromJson(needs.get-batch-count.outputs.batch-count)) }}
    outputs:
      configs-batch-${{ matrix.batch-index }}: ${{ steps.get-configs.outputs.configs }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - id: get-configs
        run: |
          pip install pydantic
          CONFIG_JSON=$(python3 utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files .github/configs/master.yaml \
            --seq-lens 1k1k \
            --batch-index ${{ matrix.batch-index }})
          echo "configs=$CONFIG_JSON" >> $GITHUB_OUTPUT

  benchmark:
    needs: get-batch-configs
    uses: ./.github/workflows/benchmark-tmpl.yml
    strategy:
      fail-fast: false
      matrix:
        config: ${{ fromJson(needs.get-batch-configs.outputs.configs-batch-0) }}
    with:
      # your benchmark parameters
      ...
```

### Simplified Approach (Current Pattern)

For the current use case where configs are split by model prefix, no changes are needed to workflows. The batching feature is available when a single model prefix generates more than 256 configurations:

```yaml
jobs:
  get-model-configs:
    runs-on: ubuntu-latest
    outputs:
      search-space-config: ${{ steps.get-configs.outputs.search-space-config }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - id: get-configs
        run: |
          pip install pydantic
          # If this generates >256 configs, use --batch-index
          CONFIG_JSON=$(python3 utils/matrix-logic/generate_sweep_configs.py \
            full-sweep \
            --config-files .github/configs/nvidia-master.yaml \
            --seq-lens 1k1k \
            --model-prefix mymodel \
            --batch-index 0)
          echo "search-space-config=$CONFIG_JSON" >> $GITHUB_OUTPUT

  benchmark:
    needs: get-model-configs
    uses: ./.github/workflows/benchmark-tmpl.yml
    strategy:
      fail-fast: false
      matrix:
        config: ${{ fromJson(needs.get-model-configs.outputs.search-space-config) }}
    # ... rest of config
```

## Examples

### Example 1: Split 500 configs into batches of 256

```bash
# Get number of batches
$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --get-batch-count
2

# Get first batch (256 configs)
$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --batch-index 0 | jq 'length'
256

# Get second batch (244 configs)
$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --batch-index 1 | jq 'length'
244
```

### Example 2: Use custom batch size

```bash
# Split into batches of 100
$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --max-batch-size 100 \
    --get-batch-count
5

# Get third batch
$ python3 generate_sweep_configs.py full-sweep \
    --config-files master.yaml \
    --max-batch-size 100 \
    --batch-index 2 | jq 'length'
100
```

## Notes

- All batches except possibly the last will contain exactly `max-batch-size` configurations
- The last batch contains the remainder (can be less than `max-batch-size`)
- Batch indices are 0-based
- Requesting an invalid batch index will result in an error
- The `--get-batch-count` and `--batch-index` flags work with all subcommands: `full-sweep`, `test-config`, `runner-model-sweep`, `runner-sweep`, and `custom`
