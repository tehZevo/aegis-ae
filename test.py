import random

import numpy as np
from protopost import protopost_client as ppcl
from nd_to_json import nd_to_json

INPUT_SIZE = 128
HOST = "http://127.0.0.1:8080"

AE_SURPRISE = lambda obs: ppcl(HOST + "/surprise-and-train", nd_to_json(obs))

#by using a seed, repeated runs should have low rnd reward
np.random.seed(777)
samples = np.random.normal(0, 1, size=[10, INPUT_SIZE])

#ae loss should decrease
for step in range(100):
  sample = random.choice(samples)
  ae_surprise = AE_SURPRISE(sample)
  print(f"Step {step + 1} reward: {ae_surprise}")
