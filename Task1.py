import tensorflow as tf

inputs_tensor = tf.constant([7, 15], dtype=tf.float32)
inputs_weight = tf.constant([0.6, 0.4], dtype=tf.float32)

weighted_inputs = tf.multiply(inputs_tensor, inputs_weight)
yield=tf.reduce_sum(weighted_inputs)

weather_severity_probability = tf.sigmoid(yield)
tf.print("Вероятность суровости зимы:", weather_severity_probability)