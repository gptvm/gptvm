    
module {
  func.func @main_graph(%arg0: tensor<1x1x4096xf32>, %arg1: tensor<1x1x1x128xf32>, %arg2: tensor<1x1xi64>, %arg3: tensor<2048x128xf32>) ->(tensor<1x32x1x128xf32>){
    %1133 = onnx.Constant dense<1> : tensor<1xi64>
    %1134 = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
    %1135 = onnx.Constant dense<64> : tensor<1xi64>
    %1136 = onnx.Constant dense<3> : tensor<1xi64>
    %1137 = onnx.Constant dense<1> : tensor<1xi64>
    %1138 = onnx.Constant dense<64> : tensor<1xi64>
    %1139 = onnx.Constant dense<0> : tensor<1xi64>
    %1140 = onnx.Constant dense<3> : tensor<1xi64>
    %1142 = onnx.Constant dense<1> : tensor<1xi64>
    %1147 = onnx.Constant dense<1> : tensor<1xi64>
    %1149 = onnx.Constant dense<0> : tensor<1xi64>
    %1150 = onnx.Constant dense<0> : tensor<1xi64>
    %1154 = onnx.Constant dense<[1, 1, 32, 128]> : tensor<4xi64>
    %1232 = onnx.Constant dense_resource<__elided__> : tensor<4096x4096xf32>
    %1488 = onnx.Constant dense<64> : tensor<1xi64>
    %1476 = "onnx.MatMul"(%arg0, %1232) : (tensor<1x1x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x1x4096xf32>
    %1479 = "onnx.Reshape"(%1476, %1154) {allowzero = 0 : si64, onnx_node_name = "/Reshape_1"} : (tensor<1x1x4096xf32>, tensor<4xi64>) -> tensor<1x1x32x128xf32>
    %1480 = "onnx.Transpose"(%1479) {perm = [0, 2, 1, 3]} : (tensor<1x1x32x128xf32>) -> tensor<1x32x1x128xf32>
    %1497 = "onnx.Slice"(%1480, %1139, %1138, %1140, %1137) : (tensor<1x32x1x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x32x1x64xf32>
    %1498 = "onnx.Slice"(%1480, %1135, %1134, %1136, %1133) : (tensor<1x32x1x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x32x1x64xf32>
    %1499 = "onnx.Neg"(%1498) {onnx_node_name = "/Neg"} : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x64xf32>
    %1500 = "onnx.Concat"(%1499, %1497) {axis = 3 : si64, onnx_node_name = "/Concat_1"} : (tensor<1x32x1x64xf32>, tensor<1x32x1x64xf32>) -> tensor<1x32x1x128xf32>
    %1501 = "onnx.Mul"(%1500, %arg1) {onnx_node_name = "/Mul_4"} : (tensor<1x32x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x32x1x128xf32>
    %1489 = "onnx.Slice"(%arg3, %1149, %1488, %1150, %1147) : (tensor<2048x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<64x128xf32>
    %1492 = "onnx.Gather"(%1489, %arg2) {axis = 0 : si64, onnx_node_name = "/Gather_2"} : (tensor<64x128xf32>, tensor<1x1xi64>) -> tensor<1x1x128xf32>
    %1493 = "onnx.Unsqueeze"(%1492, %1142) {onnx_node_name = "/Unsqueeze_5"} : (tensor<1x1x128xf32>, tensor<1xi64>) -> tensor<1x1x1x128xf32>
    %1496 = "onnx.Mul"(%1480, %1493) {onnx_node_name = "/Mul_3"} : (tensor<1x32x1x128xf32>, tensor<1x1x1x128xf32>) -> tensor<1x32x1x128xf32>
    %1502 = "onnx.Add"(%1496, %1501) {onnx_node_name = "/Add_2"} : (tensor<1x32x1x128xf32>, tensor<1x32x1x128xf32>) -> tensor<1x32x1x128xf32>
    return %1502 : tensor<1x32x1x128xf32>
   }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}