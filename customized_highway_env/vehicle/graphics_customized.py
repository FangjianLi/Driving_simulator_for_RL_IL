from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from customized_highway_env.vehicle.controller_customized import MDPVehicle_original



class VehicleGraphics_original(VehicleGraphics):

    @classmethod
    def get_color(cls, vehicle: Vehicle, transparent: bool = False):
        color = cls.DEFAULT_COLOR
        if getattr(vehicle, "color", None):
            color = vehicle.color
        elif vehicle.crashed:
            color = cls.RED
        elif isinstance(vehicle, LinearVehicle):
            color = cls.YELLOW
        elif isinstance(vehicle, IDMVehicle):
            color = cls.BLUE
        elif isinstance(vehicle, MDPVehicle):
            color = cls.EGO_COLOR
        elif isinstance(vehicle, MDPVehicle_original):
            if vehicle.color_schme:
                color = (200, 200, 0)
            else:
                color = (50, 200, 0)
        try:
            if not vehicle.color_schme:
                color = (50, 200, 0)
        except:
            pass
        if transparent:
            color = (color[0], color[1], color[2], 30)
        return color