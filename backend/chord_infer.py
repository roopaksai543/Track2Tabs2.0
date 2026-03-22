import json
import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=np.float32)
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


class ChordModel:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.labels = data["labels"]
        self.W = np.array(data["coef"], dtype=np.float32)
        self.b = np.array(data["intercept"], dtype=np.float32)

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        if x.shape[0] != self.W.shape[1]:
            raise ValueError(
                f"Feature length mismatch: got {x.shape[0]}, expected {self.W.shape[1]}"
            )

        z = self.W @ x + self.b
        p = softmax(z)
        k = int(np.argmax(p))

        return self.labels[k], float(p[k]), p