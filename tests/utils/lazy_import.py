# This file is part of the GPTVM Toolkit.
# if test case need to import gptvm/libgptvm, copy blow codes
# to your test case file and call gptvm_lazy_import() before
# using gptvm/libgptvm

def lazy_import(name, as_=None):
    # Doesn't handle error_msg well yet
    import importlib
    mod = importlib.import_module(name)
    if as_ is not None:
        name = as_
    globals()[name] = mod

def gptvm_lazy_import():
    lazy_import('libgptvm')
    lazy_import('gptvm')