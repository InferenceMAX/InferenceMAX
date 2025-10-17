import yaml
import os
import json

def flatten_search_space(config_path, runner, model, isl, osl, precision):
    with open(config_path, 'r') as f:
        search_space = yaml.safe_load(f)

    seq_len_map = {
        '1024': '1k',
        '8192': '8k',
    }

    model_map = {
        'gpt-oss': 'gptoss',
        'llama-3.3-70b-instruct': 'llama',
        'deepseek-r1-0528': 'dsr1',
    }

    model_key = None
    model_lower = model.lower()
    for key, value in model_map.items():
        if key.lower() in model_lower:
            model_key = value
            break

    assert model_key, f"model '{model}' not recognized"

    assert seq_len_map.get(isl) and seq_len_map.get(osl), f"either isl or osl not recognized"
    seq_len = f"{seq_len_map[isl]}{seq_len_map[osl]}"

    assert search_space.get(model_key, {}).get(precision), f"precision '{precision}' not recognized"
    assert search_space.get(model_key, {}).get(precision).get(runner), f"runner '{runner}' not recognized"

    entries = search_space.get(model_key, {}).get(
        precision).get(runner, {}).get(seq_len, [])

    flattened_search_space = []
    for entry in entries:
        assert entry.get('tp'), f"entry malformed, expecting field 'tp'"
        tp = entry.get('tp')
        assert isinstance(tp, int) or (isinstance(tp, list) and all(isinstance(x, int) for x in tp)), \
            f"entry malformed, expecting field 'tp' to be either an int or list of ints"
        
        tp_list = entry['tp'] if isinstance(entry['tp'], list) else [entry['tp']]

        conc_config = entry.get('conc')
        
        assert conc_config, f"entry malformed, missing field 'conc'"
        assert isinstance(conc_config, dict), f"entry malformed, 'conc' must be a dict"
        assert 'start' in conc_config, f"entry malformed, 'conc' missing required field 'start'"
        assert 'end' in conc_config, f"entry malformed, 'conc' missing required field 'end'"
        assert isinstance(conc_config['start'], int), f"entry malformed, 'conc.start' must be an int"
        assert isinstance(conc_config['end'], int), f"entry malformed, 'conc.end' must be an int"
        assert conc_config['start'] <= conc_config['end'], f"entry malformed, 'conc.start' must be <= 'conc.end'"
        
        start = conc_config['start']
        end = conc_config['end']
        step_factor = conc_config.get('step', 2)
        
        if 'step' in conc_config:
            assert isinstance(step_factor, int), f"entry malformed, 'conc.step' must be an int"
            assert step_factor > 1, f"entry malformed, 'conc.step' must be > 1"
        
        conc_list = []
        current = start
        while current <= end:
            conc_list.append(current)
            current *= step_factor

        for tp_value in tp_list:
            for conc_value in conc_list:
                new_entry = entry.copy()
                new_entry['tp'] = tp_value
                new_entry['conc'] = conc_value
                flattened_search_space.append(new_entry)

    return flattened_search_space


def main():
    config_path = '.github/configs/search-space.yml'
    runner = os.environ['RUNNER']
    model = os.environ['MODEL']
    isl = os.environ['ISL']
    osl = os.environ['OSL']
    precision = os.environ['PRECISION']
    
    flattened_search_space = flatten_search_space(
        config_path, runner, model, isl, osl, precision
    )
    
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"flattened-matrix={json.dumps(flattened_search_space)}\n")


if __name__ == '__main__':
    main()