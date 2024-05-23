import torch
from vggmodel import ShapeClassifier

model = torch.load("./model_epoch_num_2.pt",
                   map_location=torch.device("cpu")
                   )

sample_input = torch.randn(1,1,200,200)


torch.onnx.export(model,
                  sample_input,
                  "VGGshape-datamay.onnx",
                  input_names=["input"],
                  output_names=["output"]
                  )
