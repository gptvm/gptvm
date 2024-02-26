import pytest

def lazy_import(name, as_=None):
    # Doesn't handle error_msg well yet
    import importlib
    mod = importlib.import_module(name)
    if as_ is not None:
        name = as_
    # yuck...
    globals()[name] = mod

def gptvm_lazy_import():
    lazy_import('libgptvm')
    lazy_import('gptvm')

def foo(a, b):
    return a - b

def func(a, b):
    return a * b

def test_injector():
    gptvm_lazy_import()
    libgptvm.register_injector(foo.__code__, func)
    assert(foo(2, 1) == 2)