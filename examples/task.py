import gptvm
import os
import struct

def graph(objects):
    # task = gptvm.GVTask.create("Hello")
    # return task.launch(objects)
    return objects[0]

if __name__ == "__main__":
    input = b'Hello'
    object = gptvm.GVObject.create(input)

    root_task = gptvm.GVTask.create(graph)
    result = root_task.launch([object])
    print (str(result.get()))
