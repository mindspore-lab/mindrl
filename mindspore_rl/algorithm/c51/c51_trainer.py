"""C51 Trainer"""

import mindspore as ms
from mindspore import Parameter, Tensor, set_seed
from mindspore.ops import operations as P

from mindspore_rl.agent import trainer
from mindspore_rl.agent.trainer import Trainer
from mindspore_rl.utils.n_step_buffer import NStepBuffer

set_seed(5)


class CategoricalDQNTrainer(Trainer):
    """DQN Trainer"""

    def __init__(self, msrl, params):
        super().__init__(msrl)
        self.zero = Tensor(0, ms.float32)
        self.squeeze = P.Squeeze()
        self.less = P.Less()
        self.zero_value = Tensor(0, ms.int64)
        self.fill_value = Tensor(1000, ms.int64)
        self.inited = Parameter(Tensor((False,), ms.bool_), name="init_flag")
        self.mod = P.Mod()
        self.false = Tensor((False,), ms.bool_)
        self.true = Tensor((True,), ms.bool_)
        self.num_evaluate_episode = params["num_evaluate_episode"]
        self.print = P.Print()
        self.td_step = params["td_step"]
        self.data_shapes = params["data_shape"]
        self.data_types = params["data_type"]
        self.n_step_buffer = NStepBuffer(
            self.data_shapes, self.data_types, self.td_step
        )

    def trainable_variables(self):
        """Trainable variables for saving."""
        trainable_variables = {"policy_net": self.msrl.learner.policy_network}
        return trainable_variables

    @ms.jit
    def init_training(self):
        """Initialize training"""
        state = self.msrl.collect_environment.reset()
        done = self.false
        i = self.zero_value
        if self.td_step == 1:
            while self.less(i, self.fill_value):
                done, _, new_state, action, my_reward = self.msrl.agent_act(
                    trainer.INIT, state
                )
                self.msrl.replay_buffer_insert(
                    [state, action, my_reward, new_state, done]
                )
                state = new_state
                if done:
                    state = self.msrl.collect_environment.reset()
                    done = self.false
                i += 1
        else:
            while self.less(i, self.fill_value):
                done, _, new_state, action, my_reward = self.msrl.agent_act(
                    trainer.INIT, state
                )
                exp = [state, action, my_reward, new_state, done]
                self.n_step_buffer.push(exp)
                check, exp_u = self.n_step_buffer.get_data()
                if check:
                    self.msrl.replay_buffer_insert(
                        [
                            exp_u[0][0],
                            exp_u[1][0],
                            exp_u[2].squeeze(1).reshape(self.td_step),
                            exp_u[3][self.td_step - 1],
                            exp_u[4].squeeze(1).reshape(self.td_step),
                        ]
                    )
                state = new_state
                if done:
                    state = self.msrl.collect_environment.reset()
                    done = self.false
                i += 1
            self.n_step_buffer.clear()
        return done

    @ms.jit
    def train_one_episode(self):
        """Train one episode"""
        if not self.inited:
            self.init_training()
            self.inited = self.true
        state = self.msrl.collect_environment.reset()
        done = self.false
        total_reward = self.zero
        steps = self.zero
        loss = self.zero
        i = self.zero_value
        if self.td_step == 1:
            while not done:
                done, r, new_state, action, my_reward = self.msrl.agent_act(
                    trainer.COLLECT, state
                )
                self.msrl.replay_buffer_insert(
                    [state, action, my_reward, new_state, done]
                )
                state = new_state
                r = self.squeeze(r)
                loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
                total_reward += r
                steps += 1
                self.msrl.learner.update()
        else:
            while not done:
                done, r, new_state, action, my_reward = self.msrl.agent_act(
                    trainer.COLLECT, state
                )
                exp = [state, action, my_reward, new_state, done]
                self.n_step_buffer.push(exp)
                check, exp_u = self.n_step_buffer.get_data()
                if check:
                    self.msrl.replay_buffer_insert(
                        [
                            exp_u[0][0],
                            exp_u[1][0],
                            exp_u[2].squeeze(1).reshape(self.td_step),
                            exp_u[3][self.td_step - 1],
                            exp_u[4].squeeze(1).reshape(self.td_step),
                        ]
                    )
                state = new_state
                i += 1
                r = self.squeeze(r)
                loss = self.msrl.agent_learn(self.msrl.replay_buffer_sample())
                total_reward += r
                steps += 1
                self.msrl.learner.update()
            self.n_step_buffer.clear()
        return loss, total_reward, steps

    @ms.jit
    def evaluate(self):
        """Policy evaluate"""
        total_reward = self.zero
        eval_iter = self.zero
        while self.less(eval_iter, self.num_evaluate_episode):
            episode_reward = self.zero
            state = self.msrl.eval_environment.reset()
            done = self.false
            while not done:
                done, r, state = self.msrl.agent_act(trainer.EVAL, state)
                r = self.squeeze(r)
                episode_reward += r
            total_reward += episode_reward
            eval_iter += 1
        avg_reward = total_reward / self.num_evaluate_episode
        return avg_reward
