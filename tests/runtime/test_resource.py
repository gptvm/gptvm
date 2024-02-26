import pytest
import os


@pytest.mark.skip(reason="Skip in python version")
def test_resource():
    print(f"\nTest resource")
    EXAMPLES_DIR = os.path.join(os.environ["BUILD_DIR_ABSOLUTE_PATH"], "examples")
    RES_BIN = "resource_test"
    assert os.path.exists(f"{EXAMPLES_DIR}/{RES_BIN}")
    # execute the resource test
    # check if the test passed by checking the return code
    assert os.system(f"{EXAMPLES_DIR}/{RES_BIN}") == 0
