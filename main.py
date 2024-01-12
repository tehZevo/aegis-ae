import os
import gc

from tensorflow.keras.models import load_model
import numpy as np

from protopost import ProtoPost
from nd_to_json import nd_to_json, json_to_nd

from ae import create_ae

PORT = os.getenv("PORT", 80)
INPUT_SIZE = int(os.getenv("INPUT_SIZE", 128))
LATENT_SIZE = int(os.getenv("LATENT_SIZE", 32))
ENCODER_HIDDEN = os.getenv("ENCODER_HIDDEN", "64").split()
DECODER_HIDDEN = os.getenv("DECODER_HIDDEN", "64").split()
ACTIVATION = os.getenv("ACTIVATION", "swish")
LATENT_ACTIVATION = os.getenv("LATENT_ACTIVATION", "tanh")
LR = float(os.getenv("LR", 1e-3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 10000))
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.keras")
AUTOSAVE = int(os.getenv("AUTOSAVE", 1000)) #save every N calls to train

#TODO: support conv AE
#TODO: make and use a model builder library

#try to load model, else build and save a new one
try:
  model = load_model(MODEL_PATH)
  print(f"Loaded model from {MODEL_PATH}")
  encoder = model.layers[-2]
  decoder = model.layers[-1]
except IOError as e:
  print(f"Model not found at {MODEL_PATH}, creating...")
  model, encoder, decoder = create_ae(
    input_size=INPUT_SIZE,
    latent_size=LATENT_SIZE,
    encoder_hidden=ENCODER_HIDDEN,
    decoder_hidden=DECODER_HIDDEN,
    lr=LR,
    activation=ACTIVATION,
    latent_activation=LATENT_ACTIVATION
  )
  model.save(MODEL_PATH)

print("Encoder summary:")
encoder.summary()

print("Decoder summary:")
decoder.summary()

print("Full model summary:")
model.summary()

memory = []

def add_to_buffer(x):
    memory.append(x)

    while len(memory) > BUFFER_SIZE:
        memory.pop(0)

#adds to the buffer if sample loss is greater than batch loss
def add_to_buffer_smart(x, batch_loss):
    if len(memory) < BUFFER_SIZE:
        memory.append(x)
        return

    x2 = np.expand_dims(x, 0)
    loss = model.test_on_batch(x2, x2)
    if loss > batch_loss:
        memory.pop(0)
        memory.append(x)

def get_batch(size=32):
    batch = []
    for _ in range(size):
        i = np.random.randint(len(memory))
        batch.append(memory[i])
    batch = np.array(batch)
    return batch

#x should be batch/stacked
def calc_test_loss(x):
    loss = model.test_on_batch(x, x)
    return loss

train_steps = 0

def train():
    global train_steps
    if len(memory) == 0:
        return 0

    batch = get_batch(BATCH_SIZE)
    loss = model.train_on_batch(batch, batch)

    if AUTOSAVE >= 1:
        train_steps += 1
        if train_steps >= AUTOSAVE:
            train_steps = 0
            save_model()

    return loss

#encodes the input, adds input to buffer, and trains for 1 batch
def encode_and_train(data):
    x = json_to_nd(data)
    #encode
    encoded = encoder.predict_on_batch(np.expand_dims(x, 0))[0]
    encoded = nd_to_json(encoded)
    #add to buffer
    add_to_buffer(x)
    #train
    train_loss = train()

    return encoded

def just_encode(data):
    x = json_to_nd(data)
    #encode
    x = encoder.predict_on_batch(np.expand_dims(x, 0))[0]
    x = nd_to_json(x)

    return x

def just_decode(data):
    x = json_to_nd(data)
    #decode
    x = decoder.predict_on_batch(np.expand_dims(x, 0))[0]
    x = nd_to_json(x)

    return x

def reconstruct(data):
    x = json_to_nd(data)
    #encode-decode
    x = model.predict_on_batch(np.expand_dims(x, 0))[0]
    x = nd_to_json(x)

    return x

#calculates and returns loss, adds to buffer, then trains for 1 batch
def surprise_and_train(data):
    x = json_to_nd(data)
    #calc loss
    loss = calc_test_loss(np.expand_dims(x, 0))
    #add to buffer
    add_to_buffer(x)
    #train
    train_loss = train()

    return loss

#calculates and returns loss, adds to buffer, then trains for 1 batch
def just_surprise(data):
    x = json_to_nd(data)
    #calc loss
    loss = calc_test_loss(np.expand_dims(x, 0))

    return loss

def save_model():
    model.save(MODEL_PATH)
    print(f"Saving model to {MODEL_PATH}...")

routes = {
    "": encode_and_train,
    "add": lambda x: add_to_buffer(json_to_nd(x)),
    "encode": just_encode,
    "decode": just_decode,
    "reconstruct": reconstruct,
    "train": lambda data: train(), #trains on a batch and returns loss
    "surprise-and-train": surprise_and_train,
    "surprise": just_surprise,
    "save": lambda data: save_model()
}

ProtoPost(routes).start(PORT)
