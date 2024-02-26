#

## Test structure

├── conftest.py                  ***pytest config，define the common methods***<br>
├── __init__.py<br>
├── model                        ***place to put the test for a certain model***<br>
│   └── test_mnist.py            ***mnist：handwriting digits recognition***<br>
├── python                       ***test cases for gptvm python related functins***<br>
│   └── test_injector.py         ***test for function injector***<br>
├── runtime                      ***runtime related unit tests***<br>
│   └── test_resource.py<br>
├── test_build.py                ***building test(skip by default because conftest will call it implictly)***<br>
└── utils                        ***toolkits***<br>
│   └── model_util.py            ***model download util***<br>
│   └── lazy_import.py           ***module import util***<br>
├── README.md

## Run test

- setup test env

  ```bash
  pip install pytest
  ```

- run

  ```bash
  cd ${gptvm_src}/test
  python3 -m pytest .
  ```

## How to write test case

- all the test should follow pytest's rules, each test case should name with 'test' started

- if you need to import gptvm/libgptvm module, copy the following code to your source code

  ```python
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
  ```

  and call gptvm_lazy_import() in it.