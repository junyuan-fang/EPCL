import torch
# Check if CUDA is available
is_cuda_available = torch.cuda.is_available()

print("CUDA Available:", is_cuda_available)

# If CUDA is available, print the number of CUDA devices and the name of the first device
if is_cuda_available:
    print("Number of CUDA Devices:", torch.cuda.device_count())
    print("Name of CUDA Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
