# How to use the test case

1. install the requirement in file requirements-onnx.txt and requirements.txt

   ```bash
   pip install -r requirements-onnx.txt
   pip install -r requirements.txt
   ```

2. setup PYTHONPATH to path of libobiert.so (for example build/python)

   ```bash
   export PYTHONPATH=~/gptvm/build/python:~/gptvm/python:$PYTHONPATH
   ```

3. copy your bert onnx model file to onnx directory

   ```bash
   cp <path to>/model.onnx onnx/
   ```

   or you can convert a huggingface model to onnx model like below:

   ```bash
   python -m transformers.onnx --model=bert-base-uncased --feature=masked-lm --atol=2e-4 onnx/
   ```

4. run the test

   ```bash
   python3 test.py
   ```


***Note: maybe you need proxy to down load the model's data***
