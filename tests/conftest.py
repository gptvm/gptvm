import pytest
import os, sys
import importlib

from . import test_build


# first we build gptvm to setup the environment globally
@pytest.fixture(scope="session", autouse=True)
def setup_gptvm_runtime():
    # get the build dir
    bpath = test_build.get_build_dir()
    # check if build directory {}/bin/gptvm exists,
    # if not create it and build gptvm
    if not os.path.exists(bpath) or not os.path.exists(
        f"{bpath}/gptvm/python/libgptvm.so"
    ):
        test_build.test_build()
    py_path = []
    for x in ["../python", "gptvm/python"]:
        py_path.append(f"{bpath}/{x}")
    # append current dir to the python path
    py_path.append(os.getcwd())
    os.environ["PYTHONPATH"] = f"{':'.join(py_path)}:{os.environ.get('PYTHONPATH','')}"
    os.environ["PATH"] = f"{bpath}/bin:{os.environ.get('PATH','')}"
    # add the build dir to the environment
    os.environ["BUILD_DIR_ABSOLUTE_PATH"] = bpath
    sys.path.extend(py_path)
