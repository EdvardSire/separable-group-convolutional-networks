import torch

# model = torch.load("./cp_8k_steps_ckgresnet+nonseparable+h_s-32_32_64+k-7+om-10.0+f_om-10.0+do-0.0+wd-0. 0001+n_el-8+grp-SE2+smplng-uniform+implm-SIREN+ds-SUAS.pt")
model = torch.load("cp_8k_steps_ckgresnet+nonseparable+h_s-32_32_64+k-7+om-10.0+f_om-10.0+do-0.0+wd-0.0001+n_el-8+grp-SE2+smplng-uniform+implm-SIREN+ds-SUAS.pt",
                   map_location=torch.device("cpu")
                   )

sample_input = torch.randn(1,1,200,200)


torch.onnx.export(model,
                  sample_input,
                  "gcnn-gray.onnx",
                  input_names=["input"],
                  output_names=["output"]
                  )
