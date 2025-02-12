from abc import ABC, abstractmethod

from simulator import Simulator

class VisualizationBase(ABC):
    @abstractmethod
    def render(self, simulator: Simulator) -> None:
        """
        Render the current state of the simulation.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources used by the visualization.
        """
        pass