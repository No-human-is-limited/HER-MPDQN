import random

import numpy as np
from typing import Tuple
from typing import Optional
from collections import namedtuple

import gym
from gym import spaces
from gym.utils import seeding

gym.logger.set_level(40)  # noqa

from gym_hybrid.agents import BaseAgent
from gym_hybrid.agents import MovingAgent
from gym_hybrid.agents import SlidingAgent
from gym_hybrid.agents import Goods

# Action Id
ACCELERATE = 0
TURN = 1
CATCH = 2

Target = namedtuple('Target', ['x', 'y', 'radius'])


class Action:
    """"
    Action class to store and standardize the action for the environment.
    """

    def __init__(self, id_: int, parameters: list):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        """"
        Property method to return the parameter related to the action selected.

        Returns:
            The parameter related to this action_id
        """
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]


class BaseEnv(gym.Env):
    """"
    Gym environment parent class.
    """

    def __init__(
            self,
            seed: Optional[int] = None,
            max_turn: float = np.pi / 2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 100,
            penalty: float = 0.001,
            break_value: float = 0.1,
            use_goal_switch: bool = True,
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
        """
        # Agent Parameters
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.1
        self.goods_radius = 0.04
        self.agent_radius = 0.05
        self.penalty = penalty

        # Initialization
        self.goal_shape = 2
        self.seed(seed)
        self.target = None
        self.goods = Goods()
        self.viewer = None
        self.current_step = None
        self.agent = BaseAgent(break_value=break_value, delta_t=delta_t)
        parameters_min = np.array([-1, -1])
        parameters_max = np.array([+1, +1])

        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(parameters_min, parameters_max)))
        self.observation_space = spaces.Box(-np.ones(12), np.ones(12))
        # new param
        self.use_goal_switch = use_goal_switch

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        random.seed(seed)
        return [seed]

    def reset(self) -> list:
        self.current_step = 0
        limit = self.field_size - self.target_radius
        low_t = [-limit, -limit, self.target_radius]
        high_t = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low_t, high_t))

        low_g = [-limit, -limit]
        high_g = [limit, limit]
        self.goods.reset(*self.np_random.uniform(low_g, high_g))

        low_a = [-self.field_size, -self.field_size, 0]
        high_a = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low_a, high_a))
        self.agent.catch_goods = False
        while self.goal_distance(np.array((self.goods.x, self.goods.y)), np.array((self.target.x, self.target.y))) \
                <= self.target_radius - self.goods_radius:
            self.goods.reset(*self.np_random.uniform(low_g, high_g))
        return self.get_state()

    def step(self, raw_action: Tuple[int, list]) -> Tuple[list, float, bool, dict]:
        action = Action(*raw_action)
        self.current_step += 1
        if action.id == TURN:
            rotation = self.max_turn * max(min(action.parameter, 1), -1)
            self.agent.turn(rotation)
        elif action.id == ACCELERATE:
            acceleration = self.max_acceleration * max(min(action.parameter, 1), -1)
            self.agent.accelerate(acceleration)
        elif action.id == CATCH:
            if 0 < self.get_distance(self.agent.x, self.agent.y, self.goods.x, self.goods.y) \
                    <= self.agent_radius + self.goods_radius:
                self.agent.catch_goods = True
        if self.agent.catch_goods:
            self.goods.x = self.agent.x
            self.goods.y = self.agent.y

        if self.distance <= self.target_radius - self.goods_radius:
            reward = 0
            done = True
        elif self.current_step == self.max_step:
            reward = -1
            done = True
        else:
            reward = -1
            done = False

        return self.get_state(), reward, done, {}

    def get_state(self):
        state = [
            self.agent.x,
            self.agent.y,
            self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta),
            self.target.x,
            self.target.y,
            self.goods.x,
            self.goods.y,
            self.distance,
            1 if self.agent.catch_goods else 0,
            self.current_step / self.max_step
        ]
        # The environment returns the corresponding goal according to the current state
        if self.agent.catch_goods:
            achieved_goal = np.array((state[7], state[8]))
            desired_goal = np.array((state[5], state[6]))
        else:
            achieved_goal = np.array((state[0], state[1]))
            desired_goal = np.array((state[7], state[8]))
        if not self.use_goal_switch:
            achieved_goal = np.array((state[7], state[8]))
            desired_goal = np.array((state[5], state[6]))
        obs = {
            'observation': np.array(state).copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),
        }
        return obs

    def compute_reward(self, achieved_goal_next, goal, info):
        # Compute distance between goal and the achieved goal next.
        reward = -1
        d = self.goal_distance(achieved_goal_next, goal)
        if d <= self.target_radius - self.goods_radius:
            reward = 0
        return reward

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_b - goal_a, axis=-1)

    @property
    def distance(self) -> float:
        return self.get_distance(self.goods.x, self.goods.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400
        unit_x = screen_width / 2
        unit_y = screen_height / 2
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(unit_x * self.agent_radius)
            self.agent_trans = rendering.Transform(
                translation=(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y)))  # noqa
            agent.add_attr(self.agent_trans)
            agent.set_color(0.1, 0.3, 0.9)
            self.viewer.add_geom(agent)

            t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
            arrow = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrow_trans = rendering.Transform(rotation=self.agent.theta)  # noqa
            arrow.add_attr(self.arrow_trans)
            arrow.add_attr(self.agent_trans)
            arrow.set_color(0, 0, 0)
            self.viewer.add_geom(arrow)

            t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
            arrow2 = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrow_trans2 = rendering.Transform(rotation=self.agent.theta)
            arrow2.add_attr(self.arrow_trans2)
            arrow2.add_attr(self.agent_trans)
            arrow2.set_color(0, 0, 0)
            self.viewer.add_geom(arrow2)

            t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
            arrow3 = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrow_trans3 = rendering.Transform(rotation=self.agent.theta)
            arrow3.add_attr(self.arrow_trans3)
            arrow3.add_attr(self.agent_trans)
            arrow3.set_color(0, 0, 0)
            self.viewer.add_geom(arrow3)

            t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
            arrow4 = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrow_trans4 = rendering.Transform(rotation=self.agent.theta)
            arrow4.add_attr(self.arrow_trans4)
            arrow4.add_attr(self.agent_trans)
            arrow4.set_color(0, 0, 0)
            self.viewer.add_geom(arrow4)

            target = rendering.make_circle(unit_x * self.target_radius)
            target_trans = rendering.Transform(translation=(unit_x * (1 + self.target.x), unit_y * (1 + self.target.y)))
            target.add_attr(target_trans)
            target.set_color(1, 0.5, 0.5)
            self.viewer.add_geom(target)

            goods = rendering.make_circle(unit_x * self.goods_radius)
            self.goods_trans = rendering.Transform(
                translation=(unit_x * (1 + self.goods.x), unit_y * (1 + self.goods.y)))
            goods.add_attr(self.goods_trans)
            goods.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(goods)

        self.arrow_trans.set_rotation(self.agent.theta)
        self.arrow_trans2.set_rotation(self.agent.theta + np.pi / 2)
        self.arrow_trans3.set_rotation(self.agent.theta + np.pi)
        self.arrow_trans4.set_rotation(self.agent.theta - np.pi / 2)
        self.agent_trans.set_translation(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y))
        self.goods_trans.set_translation(unit_x * (1 + self.goods.x), unit_y * (1 + self.goods.y))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class MovingEnv(BaseEnv):
    def __init__(
            self,
            seed: int = None,
            max_turn: float = np.pi / 2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 200,
            penalty: float = 0.001,
            break_value: float = 0.1,
    ):
        super(MovingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
        )

        self.agent = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )


class SlidingEnv(BaseEnv):
    def __init__(
            self,
            seed: int = None,
            max_turn: float = np.pi / 2,
            max_acceleration: float = 0.5,
            delta_t: float = 0.005,
            max_step: int = 200,
            penalty: float = 0.001,
            break_value: float = 0.1
    ):
        super(SlidingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value
        )

        self.agent = SlidingAgent(
            break_value=break_value,
            delta_t=delta_t
        )
