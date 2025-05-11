import tensorflow as tf
from keras.utils import plot_model
import pydot
import graphviz
import os

# 1. Install necessary libraries
#    - tensorflow, pydot, graphviz
#    - pip install tensorflow pydot graphviz

# 2.  Ensure graphviz is in your system's PATH.
#     -   Download and install Graphviz from https://graphviz.org/download/
#     -   Add the Graphviz 'bin' directory to your PATH environment variable.
#     -   For example, on Windows, add 'C:\Program Files\Graphviz\bin' to PATH.
#     -   After installation, restart your IDE or terminal for the PATH changes to take effect.

# 3. Create a simplified model
def create_simplified_model():
    # Input layers
    input_nbhd = tf.keras.layers.Input(shape=(3, 3, 2), name='nbhd_input')
    input_flow = tf.keras.layers.Input(shape=(3, 3, 4), name='flow_input')
    input_lstm = tf.keras.layers.Input(shape=(10, 20), name='lstm_input')

    # CNN (simplified - one conv layer for each input)
    conv_nbhd = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='nbhd_conv')(input_nbhd)
    conv_flow = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='flow_conv')(input_flow)
    
    # Flatten CNN outputs
    flat_nbhd = tf.keras.layers.Flatten()(conv_nbhd)
    flat_flow = tf.keras.layers.Flatten()(conv_flow)
    
    # Concatenate CNN output
    concat_cnn = tf.keras.layers.Concatenate()([flat_nbhd, flat_flow])
    cnn_dense = tf.keras.layers.Dense(128, activation='relu')(concat_cnn)

    # LSTM
    lstm_out = tf.keras.layers.LSTM(units=128, return_sequences=True)(input_lstm)

    # Attention (simplified - using a basic dense layer as a placeholder)
    attention_out = tf.keras.layers.Dense(128, activation='sigmoid', name='attention_layer')(lstm_out)  # Simplified attention

    # Combine LSTM and Attention
    concat_lstm_attn = tf.keras.layers.Concatenate()([lstm_out, attention_out])
    
    # Final output
    output = tf.keras.layers.Dense(1, activation='tanh')(concat_lstm_attn)

    model = tf.keras.Model(inputs=[input_nbhd, input_flow, input_lstm], outputs=output)
    return model

model = create_simplified_model()

# 4. Visualize the model and save to a file
# Save the model visualization to a file.
try:
    plot_model(model, to_file='model_visualization.png', show_shapes=True, show_layer_names=True)
    print("Model visualization saved to 'model_visualization.png'")
except Exception as e:
    print(f"Error occurred while plotting the model: {e}")
    print("Please make sure graphviz is installed and in your PATH.")

# 5.  Display the image (optional)
# If you are running this in a local environment with a graphical interface,
# you can try to display the image.
try:
    from IPython.display import Image, display
    display(Image(filename='model_visualization.png'))
except ImportError:
    print("IPython is not available, so the image cannot be displayed inline.")
    print("You can still view 'model_visualization.png' directly.")
except FileNotFoundError:
    print("The model visualization was not generated. Check for previous errors.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

