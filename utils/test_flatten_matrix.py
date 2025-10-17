import pytest
import yaml
from flatten_matrix import flatten_search_space


@pytest.fixture
def minimal_config():
    """Minimal valid config for testing"""
    return {
        'gptoss': {
            'fp4': {
                'h100': {
                    '1k1k': [
                        {'tp': [2, 4], 'conc': {'start': 4, 'end': 8}}
                    ]
                }
            }
        },
        'llama': {
            'fp8': {
                'b200': {
                    '1k8k': [
                        {'tp': 2, 'conc': {'start': 4, 'end': 64, 'step': 4}}
                    ]
                }
            }
        },
        'dsr1': {
            'fp4': {
                'b200-trt': {
                    '8k1k': [
                        {'tp': 4, 'conc': {'start': 4, 'end': 32}},
                        {'tp': 4, 'ep': 4, 'dp_attention': 'true', 'conc': {'start': 64, 'end': 256}}
                    ]
                }
            }
        }
    }


@pytest.fixture
def config_file(minimal_config, tmp_path):
    # temp config file
    config_path = tmp_path / "search-space.yml"
    with open(config_path, 'w') as f:
        yaml.dump(minimal_config, f)
    return config_path


class TestValidCases:
    """Test valid input scenarios"""
    
    def test_single_tp_value(self, config_file):
        """Test with single TP value"""
        result = flatten_search_space(
            config_file, 'b200', 'llama-3.3-70b-instruct', '1024', '8192', 'fp8'
        )
        
        # Should generate: tp=2, conc=[4, 16, 64] with step=4
        assert len(result) == 3
        assert all(entry['tp'] == 2 for entry in result)
        assert [entry['conc'] for entry in result] == [4, 16, 64]
    
    def test_list_of_tp_values(self, config_file):
        """Test with list of TP values"""
        result = flatten_search_space(
            config_file, 'h100', 'gpt-oss', '1024', '1024', 'fp4'
        )
        
        # Should generate: tp=[2,4], conc=[4,8] = 2*2 = 4 combinations
        assert len(result) == 4
        tp_values = [entry['tp'] for entry in result]
        assert tp_values.count(2) == 2
        assert tp_values.count(4) == 2
    
    def test_optional_fields_preserved(self, config_file):
        """Test that optional fields like ep and dp_attention are preserved"""
        result = flatten_search_space(
            config_file, 'b200-trt', 'deepseek-r1-0528', '8192', '1024', 'fp4'
        )
        
        # Second entry should have ep and dp_attention
        entries_with_ep = [e for e in result if 'ep' in e]
        assert len(entries_with_ep) > 0
        assert all(e['ep'] == 4 for e in entries_with_ep)
        
        entries_with_dp = [e for e in result if 'dp_attention' in e]
        assert len(entries_with_dp) > 0
        assert all(e['dp_attention'] == 'true' for e in entries_with_dp)
    
    def test_default_step_factor(self, config_file):
        """Test that default step factor of 2 is used when not specified"""
        result = flatten_search_space(
            config_file, 'h100', 'gpt-oss', '1024', '1024', 'fp4'
        )
        
        # conc: start=4, end=8, default step=2 -> [4, 8]
        conc_values = sorted(set(entry['conc'] for entry in result))
        assert conc_values == [4, 8]
    
    def test_custom_step_factor(self, config_file):
        """Test custom step factor"""
        result = flatten_search_space(
            config_file, 'b200', 'llama-3.3-70b-instruct', '1024', '8192', 'fp8'
        )
        
        # conc: start=4, end=64, step=4 -> [4, 16, 64]
        conc_values = sorted(set(entry['conc'] for entry in result))
        assert conc_values == [4, 16, 64]


class TestModelMapping:
    """Test model name mapping"""
    
    def test_gptoss_mapping(self, config_file):
        """Test gpt-oss maps to gptoss"""
        result = flatten_search_space(
            config_file, 'h100', 'gpt-oss', '1024', '1024', 'fp4'
        )
        assert len(result) > 0
    
    def test_llama_mapping(self, config_file):
        """Test llama mapping with case insensitivity"""
        result = flatten_search_space(
            config_file, 'b200', 'LLAMA-3.3-70B-INSTRUCT', '1024', '8192', 'fp8'
        )
        assert len(result) > 0
    
    def test_dsr1_mapping(self, config_file):
        """Test deepseek-r1 maps to dsr1"""
        result = flatten_search_space(
            config_file, 'b200-trt', 'deepseek-r1-0528', '8192', '1024', 'fp4'
        )
        assert len(result) > 0


class TestInvalidInputs:
    """Test error handling for invalid inputs"""
    
    def test_unrecognized_model(self, config_file):
        """Test error for unrecognized model"""
        with pytest.raises(AssertionError, match="model .* not recognized"):
            flatten_search_space(
                config_file, 'h100', 'unknown-model', '1024', '1024', 'fp4'
            )
    
    def test_invalid_isl(self, config_file):
        """Test error for invalid ISL"""
        with pytest.raises(AssertionError, match="either isl or osl not recognized"):
            flatten_search_space(
                config_file, 'h100', 'gpt-oss', '2048', '1024', 'fp4'
            )
    
    def test_invalid_osl(self, config_file):
        """Test error for invalid OSL"""
        with pytest.raises(AssertionError, match="either isl or osl not recognized"):
            flatten_search_space(
                config_file, 'h100', 'gpt-oss', '1024', '4096', 'fp4'
            )
    
    def test_invalid_precision(self, config_file):
        """Test error for invalid precision"""
        with pytest.raises(AssertionError, match="precision .* not recognized"):
            flatten_search_space(
                config_file, 'h100', 'gpt-oss', '1024', '1024', 'fp16'
            )
    
    def test_invalid_runner(self, config_file):
        """Test error for invalid runner"""
        with pytest.raises(AssertionError, match="runner .* not recognized"):
            flatten_search_space(
                config_file, 'a100', 'gpt-oss', '1024', '1024', 'fp4'
            )


class TestMalformedEntries:
    """Test validation of malformed config entries"""
    
    def test_missing_tp_field(self, tmp_path):
        """Test error when tp field is missing"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'conc': {'start': 4, 'end': 8}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="entry malformed, expecting field 'tp'"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_tp_wrong_type(self, tmp_path):
        """Test error when tp is wrong type"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 'invalid', 'conc': {'start': 4, 'end': 8}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="expecting field 'tp' to be either an int or list of ints"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_tp_list_with_non_ints(self, tmp_path):
        """Test error when tp list contains non-integers"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': [2, 'four', 8], 'conc': {'start': 4, 'end': 8}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="expecting field 'tp' to be either an int or list of ints"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_missing_conc_field(self, tmp_path):
        """Test error when conc field is missing"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="entry malformed, missing field 'conc'"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_conc_not_dict(self, tmp_path):
        """Test error when conc is not a dict"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2, 'conc': [4, 8, 16]}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="'conc' must be a dict"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_conc_missing_start(self, tmp_path):
        """Test error when conc.start is missing"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2, 'conc': {'end': 8}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="'conc' missing required field 'start'"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_conc_missing_end(self, tmp_path):
        """Test error when conc.end is missing"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2, 'conc': {'start': 4}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="'conc' missing required field 'end'"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_conc_start_not_int(self, tmp_path):
        """Test error when conc.start is not an int"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2, 'conc': {'start': '4', 'end': 8}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="'conc.start' must be an int"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_conc_end_not_int(self, tmp_path):
        """Test error when conc.end is not an int"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2, 'conc': {'start': 4, 'end': '8'}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="'conc.end' must be an int"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_conc_start_greater_than_end(self, tmp_path):
        """Test error when conc.start > conc.end"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2, 'conc': {'start': 16, 'end': 4}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="'conc.start' must be <= 'conc.end'"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_step_not_int(self, tmp_path):
        """Test error when step is not an int"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2, 'conc': {'start': 4, 'end': 16, 'step': '2'}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="'conc.step' must be an int"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
    
    def test_step_not_greater_than_one(self, tmp_path):
        """Test error when step <= 1"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 2, 'conc': {'start': 4, 'end': 16, 'step': 1}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(AssertionError, match="'conc.step' must be > 1"):
            flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')


class TestConcurrencyGeneration:
    """Test concurrency value generation logic"""
    
    def test_geometric_progression(self, tmp_path):
        """Test that concurrency values follow geometric progression"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 1, 'conc': {'start': 4, 'end': 64, 'step': 2}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        result = flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
        
        conc_values = [entry['conc'] for entry in result]
        assert conc_values == [4, 8, 16, 32, 64]
    
    def test_single_conc_value(self, tmp_path):
        """Test when start equals end"""
        config = {
            'gptoss': {
                'fp4': {
                    'h100': {
                        '1k1k': [
                            {'tp': 1, 'conc': {'start': 64, 'end': 64}}
                        ]
                    }
                }
            }
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        result = flatten_search_space(config_path, 'h100', 'gpt-oss', '1024', '1024', 'fp4')
        
        assert len(result) == 1
        assert result[0]['conc'] == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])