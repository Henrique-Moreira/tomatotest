import tensorflow as tf
import os
import sys

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("\nCUDA Environment Variables:")
print("CUDA_PATH:", os.environ.get('CUDA_PATH', 'Not set'))
print("PATH entries containing CUDA or NVIDIA:")
for path in os.environ.get('PATH', '').split(';'):
    if 'cuda' in path.lower() or 'nvidia' in path.lower():
        print(path)

print("\nGPU Devices:")
print(tf.config.list_physical_devices('GPU'))

# Test GPU availability
if tf.test.is_built_with_cuda():
    print("\nTensorFlow was built with CUDA")
else:
    print("\nTensorFlow was NOT built with CUDA")