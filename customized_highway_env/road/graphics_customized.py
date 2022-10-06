from highway_env.road.graphics import RoadGraphics
from customized_highway_env.vehicle.graphics_customized import VehicleGraphics_original

class RoadGraphics_original(RoadGraphics):

    @staticmethod
    def display_traffic(road, surface, simulation_frequency: int = 15, offscreen: bool = False) \
            -> None:

        if road.record_history:
            for v in road.vehicles:
                VehicleGraphics_original.display_history(v, surface, simulation=simulation_frequency, offscreen=offscreen)
        for v in road.vehicles:
            VehicleGraphics_original.display(v, surface, offscreen=offscreen)