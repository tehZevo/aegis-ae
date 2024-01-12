from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam

def dense_stack(input_size, output_size, hidden_sizes, activation, output_activation):
  model = Sequential()
  model.add(InputLayer([input_size]))
  for size in hidden_sizes:
    model.add(Dense(size, activation=activation))
  model.add(Dense(output_size, activation=output_activation))

  return model

def create_ae(input_size, latent_size, encoder_hidden, decoder_hidden, lr, activation, latent_activation):
  encoder = dense_stack(input_size, latent_size, encoder_hidden, activation, latent_activation)
  decoder = dense_stack(latent_size, input_size, decoder_hidden, activation, None)

  model = Sequential([encoder, decoder])
  model.compile(loss="mse", optimizer=Adam(lr))
  model.build(input_shape=[None, input_size]) #dunno why this is required...

  return model, encoder, decoder
