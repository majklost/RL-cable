import random
import numpy as np

class CollisionData:
    def __init__(self,normal,other_shape, other_body, my_body_idx,other_body_idx):
        self.stamp = random.randint(0, 1000000)
        self.normal = np.array([normal[0], normal[1]])
        self.other_shape = other_shape
        self.other_body = other_body
        self.my_body_idx = my_body_idx
        self.other_body_idx = other_body_idx
        self.read_level = 0

    def __str__(self):
        return f"CollisionData: {self.other_body}, {self.stamp}"