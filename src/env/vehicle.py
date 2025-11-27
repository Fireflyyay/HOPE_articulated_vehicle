from typing import Callable, List
from enum import Enum
import copy

import numpy as np
from shapely.geometry import Point, LinearRing
from shapely.affinity import affine_transform

from configs import *



class Status(Enum):
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5


class State:
    def __init__(self, raw_state: list):
        self.loc: Point = Point(raw_state[:2])
        self.heading: float = raw_state[2]
        if len(raw_state) == 3:
            self.speed: float = 0
            self.steering: float = 0
        else:
            self.speed: float = raw_state[3]
            self.steering: float = raw_state[4]
        self.rear_heading: float = raw_state[5] if len(raw_state) > 5 else self.heading

    def create_box(self) -> LinearRing:
        cos_theta = np.cos(self.heading)
        sin_theta = np.sin(self.heading)
        mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
        return affine_transform(VehicleBox, mat)

    def get_pos(self,):
        return (self.loc.x, self.loc.y, self.heading)


class KSModel(object):
    """Update the state of a vehicle by the Kinematic Single-Track Model.

    Kinematic Single-Track Model use the vehicle's current speed, heading, location, 
    acceleration, and velocity of steering angle as input. Then it returns the estimation of 
    speed, heading, steering angle and location after a small time step.

    Use the center of vehicle's rear wheels as the origin of local coordinate system.

    Assume the vehicle is front-wheel-only drive.
    """
    def __init__(
        self, 
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list
    ):
        self.wheel_base = wheel_base
        self.step_len = step_len
        self.n_step = n_step
        self.speed_range = speed_range
        self.angle_range = angle_range
        self.mini_iter = 20


    def step(self, state: State, action: list, step_time:int=NUM_STEP) -> State:
        """Update the state of a vehicle with the Kinematic Single-Track Model.

        Args:
            state (list): [x, y, car_angle, speed, steering]
            action (list): [steer, speed].
            step (float, optional): the step length for each simulation.
            n_step (int): number of step of updating the physical state. This value is decide by
                (physics simulation step length : rendering step length).

        """
        new_state = copy.deepcopy(state)
        x, y = new_state.loc.x, new_state.loc.y
        steer, speed = action
        new_state.steering = steer
        new_state.speed = speed
        new_state.speed = np.clip(new_state.speed, *self.speed_range)
        new_state.steering = np.clip(new_state.steering, *self.angle_range)

        for _ in range(step_time):
            for _ in range(self.mini_iter):
                x += new_state.speed * np.cos(new_state.heading) * self.step_len/self.mini_iter
                y += new_state.speed * np.sin(new_state.heading) * self.step_len/self.mini_iter
                new_state.heading += \
                    new_state.speed * np.tan(new_state.steering) / self.wheel_base * self.step_len/self.mini_iter 

        new_state.loc = Point(x, y)
        return new_state


class ArticulatedKSModel(KSModel):
    """Kinematic model for a single-tractor + single-trailer articulated vehicle.

    This extends the single-track model by adding a trailer heading (phi) and
    the trailer axle position computed from the hitch (assumed at the rear axle).
    The trailer dynamics use a simple nonholonomic trailer model:
        phi_dot = (v / L_t) * sin(theta - phi)
    where L_t is the trailer length (distance from hitch to trailer axle).

    The constructor accepts additional parameters `trailer_length` and `hitch_offset`.
    """
    def __init__(
        self,
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list,
        trailer_length: float = 3.0,
        hitch_offset: float = 0.0,
    ):
        super().__init__(wheel_base, step_len, n_step, speed_range, angle_range)
        self.trailer_length = trailer_length
        # hitch_offset: distance from rear-axle reference to actual hitch along vehicle body
        self.hitch_offset = hitch_offset

    def step(self, state: State, action: list, step_time:int=NUM_STEP) -> State:
        new_state = copy.deepcopy(state)
        x, y = new_state.loc.x, new_state.loc.y
        steer, speed = action
        new_state.steering = steer
        new_state.speed = speed
        new_state.speed = np.clip(new_state.speed, *self.speed_range)
        new_state.steering = np.clip(new_state.steering, *self.angle_range)

        # trailer heading (phi) default
        phi = getattr(new_state, 'trailer_heading', new_state.heading)

        for _ in range(step_time):
            for _ in range(self.mini_iter):
                # tractor (rear-axle reference) kinematics (same as KSModel)
                x += new_state.speed * np.cos(new_state.heading) * self.step_len/self.mini_iter
                y += new_state.speed * np.sin(new_state.heading) * self.step_len/self.mini_iter
                new_state.heading += (
                    new_state.speed * np.tan(new_state.steering) / self.wheel_base * self.step_len/self.mini_iter
                )

                # trailer kinematics (single trailer attached at hitch on rear axle)
                # phi_dot = (v / L_t) * sin(theta - phi)
                if self.trailer_length > 1e-6:
                    phi += (new_state.speed / self.trailer_length) * np.sin(new_state.heading - phi) * self.step_len/self.mini_iter

        new_state.loc = Point(x, y)
        new_state.trailer_heading = phi

        # compute trailer axle position assuming hitch at rear-axle minus hitch_offset
        # hitch position (hx, hy) = rear axle position shifted backward along heading by hitch_offset
        hx = new_state.loc.x - self.hitch_offset * np.cos(new_state.heading)
        hy = new_state.loc.y - self.hitch_offset * np.sin(new_state.heading)
        # trailer axle at distance trailer_length behind hitch, along trailer heading
        tx = hx - self.trailer_length * np.cos(phi)
        ty = hy - self.trailer_length * np.sin(phi)
        new_state.trailer_loc = Point(tx, ty)

        return new_state


class Vehicle:
    """_summary_
    """
    def __init__(
        self,
        wheel_base: float = WHEEL_BASE,
        step_len: float = STEP_LENGTH,
        n_step: int = NUM_STEP,
        speed_range: list = VALID_SPEED, 
        angle_range: list = VALID_STEER,
        articulated: bool = False,
        trailer_length: float = 3.0,
        hitch_offset: float = 0.0,
    ) -> None:

        self.initial_state: list = None
        self.state: State = None
        self.box: LinearRing = None
        self.trajectory: List[State] = []
        if articulated:
            self.kinetic_model = ArticulatedKSModel(wheel_base, step_len, n_step, speed_range, angle_range, trailer_length, hitch_offset)
        else:
            self.kinetic_model: Callable = KSModel(wheel_base, step_len, n_step, speed_range, angle_range)

        self.color = COLOR_POOL[0]
        self.v_max = None
        self.v_min = None


    def reset(self, initial_state: State):
        """
        Args:
            init_pos (list): [x0, y0, theta0]
        """
        self.initial_state = initial_state
        self.state = self.initial_state
        # self.color = random.sample(COLOR_POOL, 1)[0]
        self.v_max = self.initial_state.speed
        self.v_min = self.initial_state.speed
        self.box = self.state.create_box()
        self.trajectory.clear()
        self.trajectory.append(self.state)
        self.tmp_trajectory = self.trajectory.copy()

    def step(self, action: np.ndarray, step_time: int=NUM_STEP):
        """
        Args:
            action (list): [steer, speed]
        """
        prev_info = copy.deepcopy((self.state, self.box, self.v_max, self.v_min))
        self.state = self.kinetic_model.step(self.state, action, step_time)
        self.box = self.state.create_box()
        self.trajectory.append(self.state)
        self.tmp_trajectory.append(self.state)
        self.v_max = self.state.speed if self.state.speed > self.v_max else self.v_max
        self.v_min = self.state.speed if self.state.speed < self.v_min else self.v_min
        return prev_info

    def retreat(self, prev_info):
        '''
        Retreat the vehicle state from previous one.

        Args:
            prev_info (tuple): (state, box, v_max, v_min)
        '''
        self.state, self.box, self.v_max, self.v_min = prev_info
        self.trajectory.pop(-1)