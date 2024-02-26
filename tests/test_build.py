import pytest
import os
import math

BUILD_DIR="build"
# get the parent dir of the current file
# and append the build dir to it
BUILD_DIR_ABSOLUTE_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, BUILD_DIR)
# Path: tests/test_build.py
@pytest.mark.skip(reason="build done in conftest.py")
def test_build():
    print(f"\nBuild GPTVM {BUILD_DIR_ABSOLUTE_PATH}")
    # check if build directory exists, if not create it
    if not os.path.exists(BUILD_DIR_ABSOLUTE_PATH):
        os.mkdir(BUILD_DIR_ABSOLUTE_PATH)
    # cd into build dir
    pushd = os.getcwd()
    os.chdir(BUILD_DIR_ABSOLUTE_PATH)
    # run cmake
    os.system(f"cmake ..")
    assert(os.path.exists(f"{BUILD_DIR_ABSOLUTE_PATH}/CMakeCache.txt"))
    # run make, with 2/3 of the available cores
    os.system(f"make -j{math.ceil((os.cpu_count()) * 2 / 3)}")
    assert(os.path.exists(f"{BUILD_DIR_ABSOLUTE_PATH}/bin/gptvm"))
    # go back to the original dir
    os.chdir(pushd)

# return the build dir path
def get_build_dir():
    return BUILD_DIR_ABSOLUTE_PATH