import os

import numpy as np
import torch
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, image_size, pre_image_size, device, window=1,
                 transform=None):
        self.capacity = capacity
        self.device = device
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.image_size = image_size
        self.pre_image_size = pre_image_size
        self.transform = transform

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window

        # For unsupervised learning, hardcode the state obs to be 24
        self.state_obses = np.empty((capacity, 24), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.total_amount = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, state_obs=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        if state_obs is not None:
            np.copyto(self.state_obses[self.idx], state_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.total_amount += 1 if not self.full else 0
        self.full = self.full or self.idx == 0

    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):

        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)

            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)

            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)

            self.idx = next_index

    def relabel_with_predictor(self, predictor):
        batch_size = 256
        total_iter = int(self.idx / batch_size)

        if self.idx > batch_size * total_iter:
            total_iter += 1

        for index in range(total_iter):
            last_index = (index + 1) * batch_size
            if (index + 1) * batch_size > self.idx:
                last_index = self.idx

            obses = self.obses[index * batch_size:last_index]
            # actions = self.actions[index * batch_size:last_index]
            # inputs = np.concatenate([obses, actions], axis=-1)
            inputs = obses

            pred_reward = predictor.r_hat_batch_image(inputs)
            self.rewards[index * batch_size:last_index] = pred_reward

    def random_crop(self, imgs, out=84):
        """
            args:
            imgs: np.array shape (B,C,H,W)
            out: output size (e.g. 84)
            returns np.array
        """
        n, c, h, w = imgs.shape
        crop_max = h - out + 1
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)
        cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
        for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
            cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
        return cropped

    def sample(self, batch_size, sample_img=False):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        if sample_img and self.obses_img is not None:
            obses_img = self.random_crop(self.obses_img[idxs])
            obses_img = torch.as_tensor(obses_img,
                                        device=self.device).float() / 255.0
            return obses, actions, rewards, next_obses, not_dones, not_dones_no_max, obses_img

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def sample_rad(self, aug_funcs):
        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif 'translate' in aug:
                    og_obses = utils.center_crop_images(obses, self.pre_image_size)
                    og_next_obses = utils.center_crop_images(next_obses, self.pre_image_size)
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        # not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
        #                                    device=self.device)

        obses = obses / 255.
        next_obses = next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def sample_state_ent(self, batch_size, aug_funcs):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        state_obses = torch.as_tensor(self.state_obses[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = self.next_obses[idxs]
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        if self.full:
            full_state_obs = self.state_obses
        else:
            full_state_obs = self.state_obses[: self.idx]

        if aug_funcs:
            for aug, func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)
                elif 'translate' in aug:
                    # TODO: change this when applying to Quadruped
                    og_obses = utils.center_crop_images(obses, self.pre_image_size)
                    og_next_obses = utils.center_crop_images(next_obses, self.pre_image_size)
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    next_obses = func(og_next_obses, self.image_size, **rndm_idxs)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        full_state_obs = torch.as_tensor(full_state_obs, device=self.device)

        return obses, state_obses, full_state_obs, actions, rewards, next_obses, not_dones, not_dones_no_max

    def save(self, save_dir):
        max_len = 10000
        for i in range(0, len(self.rewards), max_len):
            start = i
            end = i + max_len
            path = os.path.join(save_dir, '{}_{}.pt'.format(start, end))
            tmp_obses_img = self.obses_img[i:i + max_len]
            tmp_rewards = self.rewards[i:i + max_len]
            payload = [
                tmp_obses_img,
                tmp_rewards
            ]
            torch.save(payload, path)
