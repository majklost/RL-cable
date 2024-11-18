from abc import ABC, abstractmethod
from typing import List

from deform_rl.sim.Rectangle_env.objects.base_singlebody import BaseSingleBodyObject
from deform_rl.sim.Rectangle_env.objects.base_multibody import BaseMultiBodyObject
class Simulator(ABC):
    def __init__(self,
                 movable_objects: List[BaseMultiBodyObject| BaseSingleBodyObject],
                 fixed_objects: List[BaseMultiBodyObject| BaseSingleBodyObject]):

        self.movable_objects = movable_objects
        self.fixed_objects = fixed_objects

    @abstractmethod
    def step(self):
        """
        Make a step in the simulation
        :return: None
        """
        pass

    @abstractmethod
    def apply_forces(self, forces: List, indexes: List[int]=None):
        pass


    @abstractmethod
    def export(self) -> 'BaseSimulatorExport':
        """
        Export the simulator so that it can be used in another environment
        Export must be independent of the current version of the simulator
        :return: BaseSimulatorExport object
        """
        pass

class BaseSimulatorExport:
    pass
