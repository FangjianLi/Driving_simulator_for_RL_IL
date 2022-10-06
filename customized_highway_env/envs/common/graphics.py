import pygame
import os
import numpy as np
from highway_env.road.graphics import WorldSurface
from highway_env.envs.common.graphics import EnvViewer
from customized_highway_env.road.graphics_customized import RoadGraphics_original
from highway_env.envs.common.action import ActionType, DiscreteMetaAction, ContinuousAction
from customized_highway_env.vehicle.kinematics_customized import Vehicle_original
from customized_highway_env.vehicle.controller_customized import MDPVehicle_original
import time

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}


class EnvViewer_color(EnvViewer):
    def display(self) -> None:

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics_original.display(self.env.road, self.sim_surface)

        RoadGraphics_original.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        RoadGraphics_original.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()


class EnvViewer_vertical(object):
    """A viewer to render a highway driving environment."""

    SAVE_IMAGES = False

    def __init__(self, env, ratio=1, config=None) -> None:
        self.env = env
        self.ratio = ratio
        self.config = config or env.config
        self.offscreen = self.config["offscreen_rendering"]

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.config["screen_width"], self.config["screen_height"])

        self.font_1 = pygame.font.Font(None, int(50 * self.ratio))

        # self.screen = pygame.display.set_mode([self.config["screen_width"], self.config["screen_height"]])
        self.screen = pygame.display.set_mode(
            [(self.config["screen_height"] * 3 + 300) * self.ratio, self.config["screen_width"] * 3 * self.ratio])
        self.screen.fill("azure2")

        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = self.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)

        self.clock = pygame.time.Clock()

        self.enabled = True
        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None

        ## setup the joystick
        # we should change the checking condition a little bit

        # if isinstance(self.env.action_type, ContinuousAction):
        if not isinstance(self.displayed_vehicle, MDPVehicle_original):
            try:
                pygame.joystick.init()
                self.joystick = pygame.joystick.Joystick(0)
            except:
                print("Please check the joystick")

    @property
    def displayed_vehicle(self):

        return self.env.vehicle

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler_vertical.handle_event(self, event, self.env)

    def display(self) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics_original.display(self.env.road, self.sim_surface)

        RoadGraphics_original.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        RoadGraphics_original.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        rotated_surface = pygame.transform.rotozoom(self.sim_surface, 90, 3 * self.ratio)

        self.screen.blit(rotated_surface, (0, 0))

        surface_discriptive = pygame.Surface((300 * self.ratio, 900 * self.ratio))
        surface_discriptive.fill("burlywood1")
        title_1 = self.font_1.render("Highway Driving", False, "Blue")
        surface_discriptive.blit(title_1, (15 * self.ratio, 5 * self.ratio))

        time_info = self.font_1.render(f"Time: {time.time() - self.env.time_real:.1f} s", True, "Blue")
        surface_discriptive.blit(time_info, (15 * self.ratio, 125 * self.ratio))

        speed_info = self.font_1.render(f"Speed: {self.displayed_vehicle.speed:.1f} m/s", True, "darkgreen")
        surface_discriptive.blit(speed_info, (15 * self.ratio, 225 * self.ratio))

        # if isinstance(self.env.action_type, DiscreteMetaAction):

        if isinstance(self.displayed_vehicle, MDPVehicle_original):

            mode_1 = self.font_1.render("Meta-action", False, "brown4")
            surface_discriptive.blit(mode_1, (15 * self.ratio, 50 * self.ratio))

            command_1_info = self.font_1.render(f"Command: ", True, "darkgreen")
            command_2_info = self.font_1.render(f"{ACTIONS_ALL[self.env.action_taken]}", True, "darkgreen")
            surface_discriptive.blit(command_1_info, (15 * self.ratio, 300 * self.ratio))
            surface_discriptive.blit(command_2_info, (70 * self.ratio, 350 * self.ratio))

            target_speed_1_info = self.font_1.render(f"Target Speed: ", True, "aquamarine4")
            target_speed_2_info = self.font_1.render(f"{self.displayed_vehicle.target_speed} m/s", True, "aquamarine4")
            surface_discriptive.blit(target_speed_1_info, (15 * self.ratio, 450 * self.ratio))
            surface_discriptive.blit(target_speed_2_info, (50 * self.ratio, 500 * self.ratio))

            target_lane_1_info = self.font_1.render(f"Target Lane: ", True, "aquamarine4")
            target_lane_2_info = self.font_1.render(f"{self.displayed_vehicle.target_lane_index[2]}", True,
                                                    "aquamarine4")
            surface_discriptive.blit(target_lane_1_info, (15 * self.ratio, 550 * self.ratio))
            surface_discriptive.blit(target_lane_2_info, (125 * self.ratio, 600 * self.ratio))

        else:

            mode_1 = self.font_1.render("Continuous-act.", False, "brown4")
            surface_discriptive.blit(mode_1, (15 * self.ratio, 50 * self.ratio))

            steering_1_info = self.font_1.render(f"Steering: ", True, "aquamarine4")
            surface_discriptive.blit(steering_1_info, (15 * self.ratio, 300 * self.ratio))

            steering_bar = (15 * self.env.action_taken[1] * 250 + 0.1) * self.ratio
            pygame.draw.rect(surface_discriptive, "burlywood2",
                             (15 * self.ratio, 348 * self.ratio, 250 * self.ratio, 29 * self.ratio), 0)

            pygame.draw.rect(surface_discriptive, "dodgerblue4", (
            (min((15 + 250) / 2 + steering_bar) * self.ratio, (15 + 250) / 2) * self.ratio, 350 * self.ratio,
            abs(steering_bar) * self.ratio, 25 * self.ratio), 0)
            pygame.draw.rect(surface_discriptive, "coral4",
                             (((15 + 250) / 2 - 2.5) * self.ratio, 348 * self.ratio, 5 * self.ratio, 29 * self.ratio),
                             0)

            accel_1_info = self.font_1.render(f"Accelerator: ", True, "aquamarine4")
            surface_discriptive.blit(accel_1_info, (15 * self.ratio, 400 * self.ratio))

            accel_bar = (max(0, self.env.action_taken[0]) * 250 + 0.1) * self.ratio

            pygame.draw.rect(surface_discriptive, "burlywood2", (15 * self.ratio, 448 * self.ratio, 250 * self.ratio, 29 * self.ratio), 0)
            pygame.draw.rect(surface_discriptive, "dodgerblue4", (15 * self.ratio, 450 * self.ratio, accel_bar * self.ratio, 25 * self.ratio), 0)

            brake_1_info = self.font_1.render(f"Brake: ", True, "aquamarine4")
            surface_discriptive.blit(brake_1_info, (15 * self.ratio, 500 * self.ratio))

            brake_bar = (min(-1 * min(0, self.env.action_taken[0]), 1) * 250 + 0.1) * self.ratio
            pygame.draw.rect(surface_discriptive, "burlywood2", (15 * self.ratio, 548 * self.ratio, 250 * self.ratio, 29 * self.ratio), 0)
            pygame.draw.rect(surface_discriptive, "dodgerblue4", (15 * self.ratio, 550 * self.ratio, brake_bar * self.ratio, 25 * self.ratio), 0)

        self.screen.blit(surface_discriptive, (600 * self.ratio, 0))

        if self.env.config["real_time_rendering"]:
            self.clock.tick(self.env.config["simulation_frequency"])
        pygame.display.flip()

    def window_position(self):
        if self.displayed_vehicle:
            return self.displayed_vehicle.position
        else:
            return np.array([0, 0])

    def close(self) -> None:
        pygame.quit()


class EnvViewer_vertical_carla(EnvViewer_vertical):
    def __init__(self, env, ratio=1):
        super().__init__(env, ratio)
        self.screen = pygame.display.set_mode([(1800 + 200 + 300) * self.ratio, 900 * self.ratio])
        self.screen.fill("azure2")

    def display(self) -> None:

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics_original.display(self.env.road, self.sim_surface)

        RoadGraphics_original.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        RoadGraphics_original.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        rotated_surface = pygame.transform.rotozoom(self.sim_surface, 90, 1 * self.ratio)

        self.screen.blit(rotated_surface, (1800 * self.ratio, 300 * self.ratio))

        surface_discriptive = pygame.Surface((300 * self.ratio, 900 * self.ratio))
        surface_discriptive.fill("burlywood1")
        title_1 = self.font_1.render("Highway Driving", False, "Blue")
        surface_discriptive.blit(title_1, (15 * self.ratio, 5 * self.ratio))

        time_info = self.font_1.render(f"Time: {time.time() - self.env.time_real:.1f} s", True, "Blue")
        surface_discriptive.blit(time_info, (15 * self.ratio, 125 * self.ratio))

        speed_info = self.font_1.render(f"Speed: {self.displayed_vehicle.speed:.1f} m/s", True, "darkgreen")
        surface_discriptive.blit(speed_info, (15 * self.ratio, 225 * self.ratio))

        if isinstance(self.displayed_vehicle, MDPVehicle_original):

            mode_1 = self.font_1.render("Meta-action", False, "brown4")
            surface_discriptive.blit(mode_1, (15 * self.ratio, 50 * self.ratio))

            command_1_info = self.font_1.render(f"Command: ", True, "darkgreen")
            command_2_info = self.font_1.render(f"{ACTIONS_ALL[self.env.action_taken]}", True, "darkgreen")
            surface_discriptive.blit(command_1_info, (15 * self.ratio, 300 * self.ratio))
            surface_discriptive.blit(command_2_info, (70 * self.ratio, 350 * self.ratio))

            target_speed_1_info = self.font_1.render(f"Target Speed: ", True, "aquamarine4")
            target_speed_2_info = self.font_1.render(f"{self.displayed_vehicle.target_speed} m/s", True, "aquamarine4")
            surface_discriptive.blit(target_speed_1_info, (15 * self.ratio, 450 * self.ratio))
            surface_discriptive.blit(target_speed_2_info, (50 * self.ratio, 500 * self.ratio))

            target_lane_1_info = self.font_1.render(f"Target Lane: ", True, "aquamarine4")
            target_lane_2_info = self.font_1.render(f"{self.displayed_vehicle.target_lane_index[2]}", True,
                                                    "aquamarine4")
            surface_discriptive.blit(target_lane_1_info, (15 * self.ratio, 550 * self.ratio))
            surface_discriptive.blit(target_lane_2_info, (125 * self.ratio, 600 * self.ratio))

        else:

            mode_1 = self.font_1.render("Continuous-act.", False, "brown4")
            surface_discriptive.blit(mode_1, (15 * self.ratio, 50 * self.ratio))

            steering_1_info = self.font_1.render(f"Steering: ", True, "aquamarine4")
            surface_discriptive.blit(steering_1_info, (15 * self.ratio, 300 * self.ratio))

            steering_bar = (15 * self.env.action_taken[1] * 250 + 0.1) * self.ratio
            pygame.draw.rect(surface_discriptive, "burlywood2", (15 * self.ratio, 348 * self.ratio, 250 * self.ratio, 29 * self.ratio), 0)

            pygame.draw.rect(surface_discriptive, "dodgerblue4",
                             ((min((15 + 250) / 2 + steering_bar, (15 + 250) / 2)) * self.ratio, 350 * self.ratio, abs(steering_bar), 25) * self.ratio, 0)
            pygame.draw.rect(surface_discriptive, "coral4", ((15 + 250) / 2 - 2.5, 348, 5, 29) * self.ratio, 0)

            accel_1_info = self.font_1.render(f"Accelerator: ", True, "aquamarine4")
            surface_discriptive.blit(accel_1_info, (15 * self.ratio, 400 * self.ratio))

            accel_bar = (max(0, self.env.action_taken[0]) * 250 + 0.1) * self.ratio

            pygame.draw.rect(surface_discriptive, "burlywood2", (15 * self.ratio, 448 * self.ratio, 250 * self.ratio, 29 * self.ratio), 0)
            pygame.draw.rect(surface_discriptive, "dodgerblue4", (15 * self.ratio, 450 * self.ratio, accel_bar * self.ratio, 25 * self.ratio), 0)

            brake_1_info = self.font_1.render(f"Brake: ", True, "aquamarine4")
            surface_discriptive.blit(brake_1_info, (15 * self.ratio, 500 * self.ratio))

            brake_bar = (min(-1 * min(0, self.env.action_taken[0]), 1) * 250 + 0.1) * self.ratio
            pygame.draw.rect(surface_discriptive, "burlywood2", (15 * self.ratio, 548 * self.ratio, 250 * self.ratio, 29 * self.ratio), 0)
            pygame.draw.rect(surface_discriptive, "dodgerblue4", (15 * self.ratio, 550 * self.ratio, brake_bar * self.ratio, 25 * self.ratio), 0)

        if self.env.carla_visualizer.image_array_surface:
            self.screen.blit(self.env.carla_visualizer.image_array_surface, (0, 0))
        self.screen.blit(surface_discriptive, (2000 * self.ratio, 0))

        if self.env.config["real_time_rendering"]:
            self.clock.tick(self.env.config["simulation_frequency"])
        pygame.display.flip()


class EnvViewer_vertical_tp(EnvViewer_vertical):
    @property
    def displayed_vehicle(self):
        return self.env.neighbor_vehicle


class EnvViewer_vertical_tp_carla(EnvViewer_vertical_carla):
    @property
    def displayed_vehicle(self):
        return self.env.neighbor_vehicle


class EventHandler_vertical(object):
    @classmethod
    def handle_event(cls, env_viewer, event, env):
        action_type = env.action_type

        # if isinstance(action_type, DiscreteMetaAction):

        if isinstance(env_viewer.displayed_vehicle, MDPVehicle_original):
            cls.handle_discrete_action_event(env, event)
        else:
            cls.handle_continuous_action_event(env_viewer, env)

    @classmethod
    def handle_discrete_action_event(cls, env, event):

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                env.action_to_take = "FASTER"
            if event.key == pygame.K_DOWN:
                env.action_to_take = "SLOWER"
            if event.key == pygame.K_RIGHT:
                env.action_to_take = "LANE_RIGHT"
            if event.key == pygame.K_LEFT:
                env.action_to_take = "LANE_LEFT"

    @classmethod
    def handle_continuous_action_event(cls, env_viewer, env):
        steering_value = env_viewer.joystick.get_axis(0)
        if steering_value > -0.001 and steering_value < 0.001:
            steering_value = 0
        else:
            steering_value *= 0.4

        accelerator_value = (-1 * env_viewer.joystick.get_axis(1) + 1) / 2
        brake_value = (-1 * env_viewer.joystick.get_axis(2) + 1) / 2 * 6
        acceleration_value = accelerator_value - brake_value
        env.action_to_take = [acceleration_value, steering_value]

        # print(env_viewer.joystick.get_axis(1))
