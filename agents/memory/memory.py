"""
Source: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py
"""
import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        # print(self.start)
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data[:] = 0  # unnecessary, not freeing any memory, could be slow


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.random_integers(low=0, high=self.nb_entries - 1, size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)

        if next_actions is not None:
            return states_batch, actions_batch, rewards_batch, next_states_batch, next_actions, terminals_batch
        else:
            return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, training=True):
        if not training:
            return

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.next_actions.clear()
        self.terminals.clear()

    @property
    def nb_entries(self):
        return len(self.states)


class Memory_HER(object):
    def __init__(self, limit, observation_shape, action_shape, goal_shape, next_actions=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.terminals = RingBuffer(limit, shape=(1,))
        self.goal = RingBuffer(limit, shape=(goal_shape,))
        self.achieve_goal = RingBuffer(limit, shape=(goal_shape,))
        self.achieve_goal_new = RingBuffer(limit, shape=(goal_shape,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        # batch_idxs = np.arange(batch_size)
        # random_machine.shuffle(batch_idxs)
        # print(batch_idxs)
        batch_idxs = random_machine.random_integers(low=0, high=self.nb_entries - 1, size=batch_size)
        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)
        goal_batch = self.goal.get_batch(batch_idxs)
        achieve_goal_batch = self.achieve_goal.get_batch(batch_idxs)
        achieve_goal_new_batch = self.achieve_goal_new.get_batch(batch_idxs)

        if next_actions is not None:
            return states_batch, actions_batch, rewards_batch, next_states_batch, next_actions, terminals_batch, \
                   goal_batch, achieve_goal_batch
        else:
            return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch, goal_batch, \
                   achieve_goal_batch, achieve_goal_new_batch

    def append(self, state, action, reward, next_state, goal, achieve_goal, achieve_goal_new, next_action=None,
               terminal=False, training=True):
        if not training:
            return
        # print(reward)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        self.goal.append(goal)
        self.achieve_goal.append(achieve_goal)
        self.achieve_goal_new.append(achieve_goal_new)
        # print(self.rewards.data)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.next_actions.clear()
        self.terminals.clear()
        self.goal.clear()
        self.achieve_goal.clear()
        self.achieve_goal_new.clear()

    @property
    def nb_entries(self):
        return len(self.states)


class Memory_True(object):
    def __init__(self, limit, observation_shape, action_shape, goal_shape, next_actions=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.terminals = RingBuffer(limit, shape=(1,))
        self.goal = RingBuffer(limit, shape=(goal_shape,))
        self.achieve_goal = RingBuffer(limit, shape=(goal_shape,))
        self.achieve_goal_new = RingBuffer(limit, shape=(goal_shape,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        # batch_idxs = np.arange(batch_size)
        # random_machine.shuffle(batch_idxs)
        # print(batch_idxs)
        batch_idxs = random_machine.random_integers(low=0, high=self.nb_entries - 1, size=batch_size)
        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)
        goal_batch = self.goal.get_batch(batch_idxs)
        achieve_goal_batch = self.achieve_goal.get_batch(batch_idxs)
        achieve_goal_new_batch = self.achieve_goal_new.get_batch(batch_idxs)

        if next_actions is not None:
            return states_batch, actions_batch, rewards_batch, next_states_batch, next_actions, terminals_batch, \
                   goal_batch, achieve_goal_batch
        else:
            return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch, goal_batch, \
                   achieve_goal_batch, achieve_goal_new_batch

    def append(self, state, action, reward, next_state, goal, achieve_goal, achieve_goal_new, next_action=None,
               terminal=False, training=True):
        if not training:
            return
        # print(reward)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        self.goal.append(goal)
        self.achieve_goal.append(achieve_goal)
        self.achieve_goal_new.append(achieve_goal_new)
        # print(self.rewards.data)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.next_actions.clear()
        self.terminals.clear()
        self.goal.clear()
        self.achieve_goal.clear()
        self.achieve_goal_new.clear()

    @property
    def nb_entries(self):
        return len(self.states)


class MemoryV2(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False, time_steps=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.time_steps = RingBuffer(limit, shape=(1,)) if time_steps else None
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.choice(self.nb_entries, size=batch_size)
        # batch_idxs = random_machine.choice(self.nb_entries, weights=[i/self.nb_entries for i in range(self.nb_entries)], size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)
        time_steps = self.time_steps.get_batch(batch_idxs) if self.time_steps is not None else None

        ret = [states_batch, actions_batch, rewards_batch, next_states_batch]
        if next_actions is not None:
            ret.append(next_actions)
        ret.append(terminals_batch)
        if time_steps is not None:
            ret.append(time_steps)
        return tuple(ret)

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, time_steps=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions is not None:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        if self.time_steps is not None:
            self.time_steps.append(time_steps)

    @property
    def nb_entries(self):
        return len(self.states)

