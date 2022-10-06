import carla
import numpy as np
import pygame
from pygame.locals import *
import sys
import time

spawn_points_x = np.concatenate(
    (0 * np.ones(4), 10 * np.ones(4), 20 * np.ones(4), 30 * np.ones(4), 40 * np.ones(4), 50 * np.ones(4))) - 950
spawn_points_y = [-12, -8, -4, 0] * 6




class CARLA_viusalizer():
    def __init__(self, image_x=1028, image_y=720, num_env_cars=20, driver_view=True, animation=True, two_player=False):
        self.image_x = image_x
        self.image_y = image_y
        self.num_env_cars = num_env_cars
        if animation:
            self.set_up_pygame()
        self.driver_view = driver_view
        self.two_player = two_player
        self.actor_list = []
        self.car_list = []  # lets create a list of environmental car
        self.image_array_surface = None
        self.set_up_carla()
        time.sleep(1)

    def set_up_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.image_x, self.image_y))
        pygame.display.set_caption('The Carla viewer')
        self.clock = pygame.time.Clock()

    def set_up_carla(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.load_world('zz2')

        # we may also want to check change the weather
        weather = carla.WeatherParameters(
            cloudiness=0.8,
            precipitation=0.0,
            sun_altitude_angle=50.0)  # create an weather object

        self.world.set_weather(weather)
        self.blue_print_library = self.world.get_blueprint_library()
        self.spawn_subject_car()
        if self.two_player:
            self.spawn_target_car()
        self.spawn_environmental_cars()
        self.spectator = self.world.get_spectator()
        car_s_transform = self.controlled_car.get_transform()
        spectator_transform = carla.Transform(car_s_transform.location + carla.Location(z=50),
                                              carla.Rotation(pitch=-90))
        self.spectator.set_transform(spectator_transform)
        self.spawn_camera()
        time.sleep(2)



    def spawn_subject_car(self):
        tesla_bp = self.blue_print_library.find('vehicle.tesla.model3')
        tesla_bp.set_attribute('color', '200, 200, 0')
        car_spawn_point = carla.Transform()
        car_spawn_point.location.x = spawn_points_x[0]
        car_spawn_point.location.y = spawn_points_y[0]
        car_spawn_point.location.z = 0.02
        self.controlled_car = self.world.spawn_actor(tesla_bp, car_spawn_point)
        self.controlled_car.set_simulate_physics(False)
        self.car_list.append(self.controlled_car)
        self.actor_list.append(self.controlled_car)

    def spawn_target_car(self):
        tesla_bp_target = self.blue_print_library.find('vehicle.tesla.model3')
        tesla_bp_target.set_attribute('color', '50, 200, 0')
        car_spawn_point_target = carla.Transform()
        car_spawn_point_target.location.x = spawn_points_x[1]
        car_spawn_point_target.location.y = spawn_points_y[1]
        car_spawn_point_target.location.z = 0.02
        self.target_car = self.world.spawn_actor(tesla_bp_target, car_spawn_point_target)
        self.target_car.set_simulate_physics(False)
        self.car_list.append(self.target_car)
        self.actor_list.append(self.target_car)

    def spawn_environmental_cars(self):
        tesla_bp_env = self.blue_print_library.find('vehicle.tesla.model3')
        tesla_bp_env.set_attribute('color', '100, 200, 255')

        for index in range(self.num_env_cars):
            car_spawn_point_env = carla.Transform()
            if not self.two_player:
                car_spawn_point_env.location.x = spawn_points_x[index + 1]
                car_spawn_point_env.location.y = spawn_points_y[index + 1]
            else:
                car_spawn_point_env.location.x = spawn_points_x[index + 2]
                car_spawn_point_env.location.y = spawn_points_y[index + 2]
            car_spawn_point_env.location.z += 0.02
            car_env = self.world.spawn_actor(tesla_bp_env, car_spawn_point_env)
            car_env.set_simulate_physics(False)
            self.car_list.append(car_env)
            self.actor_list.append(car_env)
            print("spawn success")

    def spawn_camera(self):
        camera_rgb_bp = self.blue_print_library.find('sensor.camera.rgb')
        camera_rgb_bp.set_attribute('image_size_x', str(self.image_x))
        camera_rgb_bp.set_attribute('image_size_y', str(self.image_y))
        camera_rgb_bp.set_attribute('sensor_tick', str(0))

        if not self.two_player:
            car_to_attach = self.controlled_car
        else:
            car_to_attach = self.target_car

        if self.driver_view:
            relative_transform = carla.Transform(carla.Location(x=-0.2, y=-0.4, z=1.2), carla.Rotation(pitch=8.0))
            self.camera_rgb = self.world.spawn_actor(camera_rgb_bp, relative_transform, car_to_attach)
        else:
            relative_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
            self.camera_rgb = self.world.spawn_actor(camera_rgb_bp, relative_transform, car_to_attach,
                                                     carla.AttachmentType.SpringArm)
        self.actor_list.append(self.camera_rgb)

        self.camera_rgb.listen(lambda image: self.show_image(image))

    def show_image(self, image):
        image_0 = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_0 = np.reshape(image_0, (image.height, image.width, 4))
        image_array = image_0[:, :, :3].copy()
        image_array_changed = image_array[:, :, ::-1]
        self.image_array_surface = pygame.surfarray.make_surface(image_array_changed.swapaxes(0, 1))

    def visualization_demo(self):

        transform_env_list = []
        for car_env in self.car_list:
            car_env.enable_constant_velocity(carla.Vector3D(x=1, y=0.0, z=0.0))
            transform_env_list.append(car_env.get_transform())

        start_time = time.time()

        while time.time() - start_time < 10:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit('The user ends this session')
            self.clock.tick(20)

            for car_env, transform_env in zip(self.car_list, transform_env_list):
                transform_env.location.x += 0.5
                car_env.set_transform(transform_env)

            car_s_transform = self.controlled_car.get_transform()
            spectator_transform = carla.Transform(car_s_transform.location + carla.Location(z=50),
                                                  carla.Rotation(pitch=-90))
            self.spectator.set_transform(spectator_transform)

            self.screen.blit(self.image_array_surface, (0, 0))
            pygame.display.update()

        self.controlled_car.enable_constant_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
        for car_env in self.car_list:
            car_env.enable_constant_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))