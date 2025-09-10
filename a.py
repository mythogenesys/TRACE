import gymnasium
import tensorflow as tf
import numpy as np

print("\n--- Verification Report ---")
print("✅ Gymnasium imported successfully.")
print("✅ NumPy imported successfully.")
print("✅ TensorFlow imported successfully.")

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"✅ Success! TensorFlow can see your M1 GPU: {gpu_devices}")
else:
    print("❌ Warning: TensorFlow cannot see your M1 GPU. CPU will be used.")
print("---------------------------\n")