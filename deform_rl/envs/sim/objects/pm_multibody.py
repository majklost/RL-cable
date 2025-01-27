from typing import List, Tuple
from abc import abstractmethod

from ..collision_data import CollisionData
from .base_multibody import BaseMultiBodyObject
from .pm_singlebody import PMSingleBodyObject


class PMMultiBodyObject(BaseMultiBodyObject):
    """abstract class for multi body object, preimplemented for linear
    objects - e.g. Cable"""

    def __init__(self, track_colisions=False, ignore_neighbour_collision=True):
        super().__init__()
        self.bodies = []  # type: List[PMSingleBodyObject]
        self._density = 1
        self._friction = 0.5
        self._color = (0, 0, 0, 0)
        self.self_collision_idxs = set()
        self.outer_collision_idxs = set()
        self.ignore_neighbour_collision = ignore_neighbour_collision
        self.track_colisions = track_colisions

    # def __deepcopy__(self, memodict={}):
    #     raise NotImplementedError("Deepcopy not implemented")

    def set_collision_type(self, collision_type):
        for b in self.bodies:
            b.set_collision_type(collision_type)

    def __getitem__(self, item) -> PMSingleBodyObject:
        return self.bodies[item]

    def __len__(self):
        return len(self.bodies)

    def __iter__(self):
        return iter(self.bodies)

    def __setitem__(self, key, value):
        self.bodies[key] = value

    def append(self, body: PMSingleBodyObject):
        body.color = self.color
        body.density = self.density
        body.friction = self.friction
        self.bodies.append(body)

    def set_ID(self, base_ID: tuple[int], moveable=bool):
        """
        Default setId that expects linear indexing
        Must be overloaded if the object has different indexing (e.g) square indexing
        :param base_ID:
        :param moveable:
        :return:
        """
        if moveable:
            for i, b in enumerate(self.bodies):
                b.set_ID((base_ID[0], i), moveable=True)
        else:
            for i, b in enumerate(self.bodies):
                b.set_ID((base_ID[0], i), moveable=False)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color
        for b in self.bodies:
            b.color = color

    # def get_force_template(self):
    #     return [b.get_force_template() for b in self.bodies]

    def add_to_space(self, space):
        for b in self.bodies:
            b.add_to_space(space)

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, density):
        self._density = density
        for b in self.bodies:
            b.density = density

    @property
    def friction(self):
        return self._friction

    @friction.setter
    def friction(self, friction):
        self._friction = friction
        for b in self.bodies:
            b.friction = friction

    def link_body(self, body, index: Tuple[int, ...]):
        if len(index) == 1:
            self.bodies[index[0]].link_body(body, index[1:])
        else:
            raise ValueError(
                "Linking of multidimensional bodies not implemented")

    def get_body(self, index: Tuple[int, ...]):
        return self.bodies[index[0]]

    def save_manual_forces(self):
        for b in self.bodies:
            b.save_manual_forces()

    @property
    def position(self):
        raise NotImplementedError("This is abstract class")

    @property
    def orientation(self):
        raise NotImplementedError("This is abstract class")

    @property
    def velocity(self):
        raise NotImplementedError("This is abstract class")

    @property
    def angular_velocity(self):
        raise NotImplementedError("This is abstract class")

    @staticmethod
    def are_neighbours(idx1: Tuple[int, ...], idx2: Tuple[int, ...]):
        """
        Check if two bodies are neighbours
        :param idx1:
        :param idx2:
        :return:
        """
        if len(idx1) != len(idx2):
            raise ValueError("Indexes must have the same length")
        for i1, i2 in zip(idx1, idx2):
            if abs(i1 - i2) > 1:
                return False
        return True

    def collision_start(self, cd: CollisionData):
        cd.read_level += 1
        self_col = (cd.other_body == self)
        if self_col:
            if self.ignore_neighbour_collision and self.are_neighbours(cd.my_body_idx, cd.other_body_idx):
                return
            self._handle_self_collision(cd)
        else:
            print("Add: ", cd.other_body_idx)
            self.outer_collision_idxs.add(cd.my_body_idx)
            self.bodies[cd.my_body_idx[cd.read_level]].collision_start(cd)

    def _handle_self_collision(self, cd: CollisionData):
        self.self_collision_idxs.add((cd.my_body_idx, cd.other_body_idx))
        self.bodies[cd.my_body_idx[cd.read_level]].collision_start(cd)

    def collision_end(self, cd: CollisionData):
        cd.read_level += 1
        self_col = (cd.other_body == self)
        if self_col:
            if self.ignore_neighbour_collision and self.are_neighbours(cd.my_body_idx, cd.other_body_idx):
                return
            if (cd.my_body_idx, cd.other_body_idx) in self.self_collision_idxs:
                self.self_collision_idxs.remove(
                    (cd.my_body_idx, cd.other_body_idx))
                self.bodies[cd.my_body_idx[cd.read_level]].collision_end(cd)
        else:
            # print("Remove: ", cd.other_body_idx)
            if cd.my_body_idx in self.outer_collision_idxs:
                self.outer_collision_idxs.remove(cd.my_body_idx)
                self.bodies[cd.my_body_idx[cd.read_level]].collision_end(cd)

    def collision_clear(self):
        for idx in self.outer_collision_idxs:
            self.get_body(idx[1:]).collision_clear()
        self.outer_collision_idxs.clear()
        for idx in self.self_collision_idxs:
            self.get_body([idx[0][1]]).collision_clear()
        self.self_collision_idxs.clear()
