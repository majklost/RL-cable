from abc import ABC, abstractmethod
from typing import Tuple

class BaseMultiBodyObject(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def position(self):
        """
        Position of the object
        Be careful each multibody object can interpret position differently
        :return:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def orientation(self):
        """
        Orientation of the object  - angle or quaternion
        Be careful each multibody object can interpret orientation differently

        :return:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def velocity(self):
        """
        Velocity of the object
        Be careful each multibody object can interpret velocity differently

        :return:
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def angular_velocity(self):
        """
        Angular velocity of the object radians per second and direction
        :return:
        """
        raise NotImplementedError



    @abstractmethod
    def link_body(self, body,index: Tuple[int,...]):
        """
        Link body to the object
        :param body: body to link
        :param index: index of the body, custom for each object
        :return:
        """
        pass

    @abstractmethod
    def get_body(self, index: Tuple[int,...]):
        """
        Get body of the object
        :param index: index of the body, custom for each object
        :return:
        """
        pass
    @abstractmethod
    def collision_start(self, collsion_data):
        pass
    @abstractmethod
    def collision_end(self, collsion_data):
        pass