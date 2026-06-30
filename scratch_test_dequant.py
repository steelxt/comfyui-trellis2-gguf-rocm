import sys
import torch
import numpy as np

# Add ComfyUI-GGUF to path
sys.path.append("/app/ComfyUI/custom_nodes/ComfyUI-GGUF")
from dequant import dequantize_tensor, gguf

# Let's mock a simple GGUF tensor
# Q8_0 uses block size 32. Each block has 1 float16 scale + 32 int8 weights = 2 + 32 = 34 bytes.
# We will create mock bytes for 2 blocks (68 bytes)
np.random.seed(42)
mock_data = np.random.randint(0, 256, size=(68,), dtype=np.uint8)

# Create raw tensor from numpy (original)
tensor_orig = torch.from_numpy(mock_data)

# Create cloned tensor (patched)
tensor_clone = torch.from_numpy(mock_data).clone()

# Let's define the gguf quantization type
qtype = gguf.GGMLQuantizationType.Q8_0
oshape = (2, 32) # 2 blocks, 32 weights each

# Assign metadata needed by dequantize_tensor
class MockTensor:
    def __init__(self, data, qtype, shape):
        self.data = data
        self.tensor_type = qtype
        self.tensor_shape = shape
        self.shape = data.shape

# Dequantize both
mock_orig = MockTensor(tensor_orig, qtype, oshape)
mock_clone = MockTensor(tensor_clone, qtype, oshape)

dequant_orig = dequantize_tensor(mock_orig, dtype=torch.float32)
dequant_clone = dequantize_tensor(mock_clone, dtype=torch.float32)

# Check if they are exactly equal
equal = torch.equal(dequant_orig, dequant_clone)
print(f"Are original and cloned dequantizations equal? {equal}")
if equal:
    print("Dequantized values are identical.")
else:
    print("WARNING: Dequantized values differ!")
    print("Original:", dequant_orig[0, :5])
    print("Cloned:", dequant_clone[0, :5])
