#!/usr/bin/env python3
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel, RewardSimSSL
from collections import deque

import utils
import hydra
import wandb

import data_augs as rad


class Workspace(object):
    def __init__(self, cfg):
        timestamp = utils.timeStamped()
        run_name = timestamp + '_' + cfg.env + '_ssl_' + str(cfg.add_ssl) + '_freq_' + str(cfg.ssl_update_freq)
        self.work_dir = os.path.join(os.getcwd(), run_name)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        # Add wandb logging
        self.wlogger = None
        if cfg.enable_wandb:
            wandb_id = wandb.util.generate_id()
            wandb.init(id=wandb_id, name=run_name,
                       project=cfg.wandb_project, entity=cfg.wandb_entity, group=cfg.wandb_group)
            wandb.config.update(cfg)
            self.wlogger = wandb

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        self.env = utils.FrameStack(self.env, k=cfg.frame_stack)

        cfg.agent.params.obs_shape = (3*cfg.frame_stack, cfg.image_size, cfg.image_size)
        cfg.agent.params.action_shape = self.env.action_space.shape
        # cfg.agent.params.action_range = [
        #     float(self.env.action_space.low.min()),
        #     float(self.env.action_space.high.max())
        # ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            cfg.batch_size,
            cfg.image_size,
            cfg.pre_transform_image_size,
            self.device,
        )

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # TODO: add image-based reward model, should consider SSL as well
        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.env.observation_space.shape,
            cfg.pre_transform_image_size,
            cfg.image_size,
            data_augs='crop',
            cfg=cfg,
            add_ssl=cfg.add_ssl,
            ssl_update_freq=cfg.ssl_update_freq,
            capacity=cfg.reward_buffer_capacity,
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal)

        self.reward_update_steps = 0

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0

        for episode in range(self.cfg.num_eval_episodes):
            frames = []
            obs = self.env.reset()
            # self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0

            frame = self.env.render(mode='rgb_array', width=256, height=256)

            frames.append(frame)

            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    obs = utils.center_crop_image(obs, self.cfg.image_size)
                    action = self.agent.select_action(obs)
                obs, reward, done, extra = self.env.step(action)

                frame = self.env.render(mode='rgb_array', width=256, height=256)
                frames.append(frame)

                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

            # save frames to video
            frames = np.stack(frames)
            if frames[0].shape[0] == 3:
                frame_size = (frames[0].shape[2], frames[0].shape[1])
            else:
                frame_size = (frames[0].shape[1], frames[0].shape[0])

            video_dir = os.path.join(self.work_dir, "eval/{}".format(self.step))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            video_name = os.path.join(self.work_dir, "eval/{}/{}.mp4".format(self.step, episode))
            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"avc1"), 30, frame_size)
            for frame in frames:
                if frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
                video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            video.release()

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                            self.step)
            self.logger.log('train/true_episode_success', success_rate,
                            self.step)
        if self.wlogger:
            log_dict = {
                'eval/episode_reward': average_episode_reward,
                'eval/true_episode_reward': average_true_episode_reward,
                'eval/video': wandb.Video(data_or_path=video_name, fps=30, format="gif")
            }
            self.wlogger.log(log_dict, step=self.step)
        self.logger.dump(self.step)

    def learn_reward(self, first_flag=0):
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        total_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    # train_acc = self.reward_model.train_reward()
                    train_acc = self.reward_model.train_reward_image()
                # total_acc = np.mean(train_acc)
                total_acc = train_acc
                if total_acc > 0.97:
                    break

        print("Reward function is updated!! ACC: " + str(total_acc))

        if self.wlogger:
            log_dict = {
                'reward/acc': total_acc,
            }
            self.wlogger.log(log_dict, step=self.step)

        if self.reward_update_steps % self.cfg.ssl_update_freq == 0:
            ssl_acc = self.reward_model.train_ssl(self.replay_buffer, self.wlogger, self.step)
            print("SSL is updated!! ACC: " + str(ssl_acc))

        self.reward_update_steps += 1

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)

                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                                    self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                                    self.step)

                obs = self.env.reset()

                if self.wlogger:
                    log_dict = {
                        'train/episode_reward': episode_reward,
                        'train/true_episode_reward': true_episode_reward,
                        'train/total_feedback': self.total_feedback,
                        'train/labeled_feedback': self.labeled_feedback,
                    }
                    if self.step > 0:
                        log_dict['train/duration'] = time.time() - start_time
                    self.wlogger.log(log_dict, step=self.step)

                # self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.sample_action(obs)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                # first learn reward
                self.learn_reward(first_flag=1)

                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)

                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()

                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True, wlogger=self.wlogger)

                # # Update image based reward
                # self.image_reward_model.update(self.replay_buffer, 256, self.step, self.logger)

                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)

                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)

                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)

                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        # self.image_reward_model.update(self.replay_buffer, 256, self.step, self.logger)

                        interact_count = 0

                self.agent.update(self.replay_buffer, self.logger, self.step)

            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.cfg.topK)

            next_obs, reward, done, extra = self.env.step(action)
            state_obs = extra['state']
            # reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
            reward_obs = next_obs[np.newaxis, :]
            reward_hat = self.reward_model.r_hat_image(reward_obs, to_numpy=True)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat.item()
            true_episode_reward += reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            # adding data to the reward training data
            self.reward_model.add_data(next_obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat,
                next_obs, done, done_no_max, state_obs)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        self.agent.save(self.work_dir, self.step)
        # self.reward_model.save(self.work_dir, self.step)
        self.reward_model.save_image_model(self.work_dir, self.step, self.replay_buffer)
        # self.replay_buffer.save(self.work_dir)


@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
