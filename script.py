import torch 
x = torch.rand(5, 3) 
print(x) 
print(torch.cuda.is_available())

print(torch.cuda.is_available())        # Should print: True
print(torch.cuda.get_device_name(0))    # e.g. "NVIDIA RTX 4090"
print(torch.cuda.device_count())        # Number of GPUs
