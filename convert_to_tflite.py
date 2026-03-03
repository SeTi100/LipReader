import tensorflow as tf
from src.mobilenet_lstm_model import GrayscaleToRGB

# Load your model
model = tf.keras.models.load_model(
    "model/best_mobilenet_lstm.h5",
    custom_objects={"GrayscaleToRGB": GrayscaleToRGB},
)

# ================================
# FIX 1: Allow SELECT TF OPs
# ================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,       # enable built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS          # enable TensorFlow ops (for LSTM)
]

# ================================
# FIX 2: Disable TensorList lowering
# ================================
converter._experimental_lower_tensor_list_ops = False

# ================================
# FLOAT32 MODEL
# ================================
tflite_model = converter.convert()
with open("best_mobilenet_lstm.tflite", "wb") as f:
    f.write(tflite_model)

# ================================
# INT8 QUANTIZED MODEL
# (Quantization + Select_TF_Ops works fine)
# ================================
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter.convert()
with open("best_mobilenet_lstm_quantized.tflite", "wb") as f:
    f.write(tflite_quant)

print("🎉 DONE — TFLite conversion succeeded with SELECT_TF_OPS!")
