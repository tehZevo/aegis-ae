import os
import gc

from tensorflow.keras.models import load_model
import numpy as np

from protopost import ProtoPost
from nd_to_json import nd_to_json, json_to_nd

PORT = os.getenv("PORT", 80)
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.h5")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 10000))
AUTOSAVE = os.getenv("AUTOSAVE", None) #save every N calls to train
AUTOSAVE = None if AUTOSAVE is None else int(AUTOSAVE)

model = load_model(MODEL_PATH)
encoder = model.layers[-2]
decoder = model.layers[-1]

encoder.summary()
decoder.summary()
model.summary()

memory = []

def add_to_buffer(x):
    while len(memory) >= BUFFER_SIZE - 1:
        memory.pop(np.random.randint(len(memory)))

    memory.append(x)

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

    if AUTOSAVE is not None:
        train_steps += 1
        if train_steps >= AUTOSAVE:
            train_steps = 0
            save_model()

    #hopefully avoid memory leak in keras
    gc.collect()

    return loss

#encodes the input, adds input to buffer, and trains for 1 batch
def encode_and_train(data):
    x = json_to_nd(data)
    #train
    train_loss = train()
    #add to buffer
    #add_to_buffer_smart(x, train_loss)
    add_to_buffer(x)
    #encode
    x = encoder.predict(np.expand_dims(x, 0))[0]
    x = nd_to_json(x)

    return x

def just_encode(data):
    x = json_to_nd(data)
    #TODO: add to buffer?
    # add_to_buffer(x)
    #encode
    x = encoder.predict(np.expand_dims(x, 0))[0]
    x = nd_to_json(x)

    return x

def just_decode(data):
    x = json_to_nd(data)
    #decode
    x = decoder.predict(np.expand_dims(x, 0))[0]
    x = nd_to_json(x)

    return x

def reconstruct(data):
    x = json_to_nd(data)
    #encode-decode
    x = model.predict(np.expand_dims(x, 0))[0]
    x = nd_to_json(x)

    return x

#calculates and returns loss, adds to buffer, then trains for 1 batch
def surprise_and_train(data):
    x = json_to_nd(data)
    #train
    train_loss = train()
    #calc loss
    loss = calc_test_loss(np.expand_dims(x, 0))
    #add to buffer
    #add_to_buffer_smart(x, train_loss)
    add_to_buffer(x)

    return loss

#calculates and returns loss, adds to buffer, then trains for 1 batch
def just_surprise(data):
    x = json_to_nd(data)
    #calc loss
    loss = calc_test_loss(np.expand_dims(x, 0))

    return loss

def safe_model():
  model.save(MODEL_PATH)
  print("Saving model to {}...".format(MODEL_PATH))

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
