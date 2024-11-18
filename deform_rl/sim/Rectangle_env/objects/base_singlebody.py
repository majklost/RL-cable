from abc import ABC, abstractmethod

class BaseSingleBodyObject(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def position(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def orientation(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def velocity(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def angular_velocity(self):
        raise NotImplementedError


    @abstractmethod
    def apply_force(self, force):
        pass

    @abstractmethod
    def collision_start(self, collsion_data):
        pass
    @abstractmethod
    def collision_end(self, collsion_data):
        pass