import pytest
from nkululeko.optim import OptimizationRunner

class DummyUtil:
    def __init__(self, *a, **kw): pass
    def error(self, msg): raise Exception(msg)
    def debug(self, msg): pass
    def high_is_good(self): return True

class DummyConfig(dict):
    def __getitem__(self, key):
        return super().get(key, {})
    def __contains__(self, key):
        return dict.__contains__(self, key)
    def add_section(self, key):
        self[key] = {}

@pytest.fixture
def runner():
    runner = OptimizationRunner(DummyConfig())
    runner.util = DummyUtil()
    return runner

def test_generate_param_combinations_single_param(runner):
    param_specs = {'lr': [0.001, 0.01, 0.1]}
    combos = runner.generate_param_combinations(param_specs)
    assert combos == [{'lr': 0.001}, {'lr': 0.01}, {'lr': 0.1}]
    assert len(combos) == 3

def test_generate_param_combinations_two_params(runner):
    param_specs = {'lr': [0.001, 0.01], 'bs': [16, 32]}
    combos = runner.generate_param_combinations(param_specs)
    expected = [
        {'lr': 0.001, 'bs': 16},
        {'lr': 0.001, 'bs': 32},
        {'lr': 0.01, 'bs': 16},
        {'lr': 0.01, 'bs': 32},
    ]
    assert combos == expected
    assert len(combos) == 4

def test_generate_param_combinations_empty(runner):
    param_specs = {}
    combos = runner.generate_param_combinations(param_specs)
    assert combos == [{}]

def test_generate_param_combinations_multiple_types(runner):
    param_specs = {'nlayers': [1, 2], 'activation': ['relu', 'tanh']}
    combos = runner.generate_param_combinations(param_specs)
    expected = [
        {'nlayers': 1, 'activation': 'relu'},
        {'nlayers': 1, 'activation': 'tanh'},
        {'nlayers': 2, 'activation': 'relu'},
        {'nlayers': 2, 'activation': 'tanh'},
    ]
    assert combos == expected
    assert len(combos) == 4