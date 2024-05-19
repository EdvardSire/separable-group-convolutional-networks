import torch
from vggmodel import ShapeClassifier

model = torch.load("VGGshape-normalgrayscale.pt",
                   map_location=torch.device("cpu")
                   )

sample_input = torch.randn(1,1,200,200)


torch.onnx.export(model,
                  sample_input,
                  "VGGshape-normalgrayscale-rollingshutter.onnx",
                  input_names=["input"],
                  output_names=["output"]
                  )
