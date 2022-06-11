import tensorflow as tf
import sys
from pathlib import Path
import os

converted = tf.lite.TFLiteConverter.from_saved_model(sys.argv[1]).convert()
path_args = Path(sys.argv[1]).parts
unique_id = path_args[7] + "_" + path_args[9]
with open(unique_id + ".tflite", "wb") as f:
    f.write(converted)

print("writing to", unique_id)

os.system(f"xxd -i {unique_id}.tflite > {unique_id}.h")
os.system(f"rm -f {unique_id}.tflite")
