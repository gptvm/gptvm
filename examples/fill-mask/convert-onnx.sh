python -m transformers.onnx --model=bert-base-uncased --feature=masked-lm --atol=2e-4 onnx/
python -m onnxruntime.tools.make_dynamic_shape_fixed --input_name input_ids --input_shape 1,1,512,512  --input_name token_type_ids --input_shape 1,1,512,512 --input_name attention_mask --input_shape 1,1,512,512  model.onnx model.fixed.onnx
