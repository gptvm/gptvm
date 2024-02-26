from pipeline import pipeline

classifier = pipeline("onnx", model="bert-base-uncased")

print(classifier("Paris is the [MASK] of France."))
print(classifier("Rose is my favorite [MASK]."))
print(classifier("Dog is the [MASK] of human."))
