// Run gptvm-onnx -domain-init
func.func @test_domain_op(%arg0: tensor<2x3xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32> {
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 6>} : (tensor<2x3xf32>) -> tensor<6xf32>
  %1 = "tosa.add"(%0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %1 : tensor<6xf32>
}

func.func @test_domain_op1(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> (tensor<6xf32> , tensor<6xf32>){
  %0 = "tosa.sub"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %1 = "tosa.add"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %0, %1 : tensor<6xf32>, tensor<6xf32>
}

func.func @test_domain_op2(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> (tensor<6xf32> , tensor<6xf32>){
  %0 = "tosa.sub"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %1 = "tosa.add"(%0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %0, %1 : tensor<6xf32>, tensor<6xf32>
}

func.func @test_domain_op3(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32> {
  %0 = "tosa.sub"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %1 = "tosa.add"(%0, %0) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %2 = "tosa.sub"(%arg1, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  %3 = "tosa.add"(%1, %2) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %3 : tensor<6xf32>
}