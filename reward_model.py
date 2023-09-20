import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time

from scipy.stats import norm

from encoder import make_encoder

device = 'cuda'


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()

        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index + 1:]

        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs,
            obs[selected_index]],
            axis=0)
    return selected_index


def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device),
                            dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)

        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


class RewardModel:
    def __init__(self, ds, da,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0,
                 teacher_eps_skip=0,
                 teacher_eps_equal=0):

        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment

        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds + self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False

        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch

        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0

        self.label_margin = label_margin
        self.label_target = 1 - 2 * self.label_margin

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size * new_frac)

    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)

    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip

    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds + self.da,
                                           out_size=1, H=256, n_layers=3,
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew

        flat_input = sa_t.reshape(1, self.da + self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])

    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)

        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:, 0]

    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)

    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)

    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )

    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )

    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))

        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch + 1) * batch_size
            if (epoch + 1) * batch_size > max_len:
                last_index = max_len

            sa_t_1 = self.buffer_seg1[epoch * batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch * batch_size:last_index]
            labels = self.buffer_label[epoch * batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)

    def get_queries(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None

        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2]  # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2]  # Batch x T x 1

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1]  # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1]  # Batch x T x 1

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])  # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])  # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])  # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])  # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i * len_traj,
                                          i * len_traj + self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(
            -1, 1)
        time_index_1 = time_index + np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(
            -1, 1)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)  # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)  # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)  # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)  # Batch x size_seg x 1

        return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        # skip the query
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)

        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size - 1):
            temp_r_t_1[:, :index + 1] *= self.teacher_gamma
            temp_r_t_2[:, :index + 1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)

        rational_labels = 1 * (sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0:  # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat * self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels

        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels

    def kcenter_sampling(self):

        # get queries
        num_init = self.mb_size * self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def kcenter_disagree_sampling(self):

        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def kcenter_entropy_sampling(self):

        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=num_init)

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:, :, :self.ds]
        temp_sa_t_2 = sa_t_2[:, :, :self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                 tot_sa_2.reshape(max_len, -1)], axis=1)

        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size)

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def disagreement_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def entropy_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size * self.large_batch)

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)

        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0

        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member in range(self.de):

                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc


class RewardSimSSL(nn.Module):
    def __init__(self,
                 encoder_type,
                 encoder_feature_dim,
                 num_layers,
                 num_filters,
                 obs_shape,
                 dim_hidden,
                 learning_rate,
                 args,
                 update_per_epoch,
                 device,
                 add_ssl=False,
                 ssl_hidden_dim=None,
                 ssl_coef=None,
                 reward_scale=1, ):
        super(RewardSimSSL, self).__init__()
        self.args = args
        self.ssl_coef = ssl_coef
        self.device = device
        self.update_per_epoch = update_per_epoch
        self.encoder_feature_dim = encoder_feature_dim
        self.dim_hidden = dim_hidden.split("_")
        self.dim_hidden = [int(x) for x in self.dim_hidden]
        self.add_ssl = add_ssl
        self.ssl_dim_hidden = ssl_hidden_dim.split("_")
        self.ssl_dim_hidden = [int(x) for x in self.ssl_dim_hidden]
        self.reward_scale = reward_scale

        if self.args.collect_then_train:
            self.collect_start = 0

        self.augs_funcs = {}

        # aug_to_func = {
        #     'crop': rad.random_crop,
        #     'grayscale': rad.random_grayscale,
        #     'cutout': rad.random_cutout,
        #     'cutout_color': rad.random_cutout_color,
        #     'flip': rad.random_flip,
        #     'rotate': rad.random_rotation,
        #     'rand_conv': rad.random_convolution,
        #     'color_jitter': rad.random_color_jitter,
        #     'translate': rad.random_translate,
        #     'no_aug': rad.no_aug,
        # }

        # for aug_name in self.data_augs.split('-'):
        #     assert aug_name in aug_to_func, 'invalid data aug string'
        #     self.augs_funcs[aug_name] = aug_to_func[aug_name]

        # create encoder
        self.encoder = make_encoder(
            encoder_type=encoder_type,
            obs_shape=obs_shape,
            feature_dim=self.encoder_feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
            output_logits=True,
            add_norm=args.add_norm,
        )
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        if self.args.reward_sim_add_steplr:
            self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer,
                                                                     step_size=self.args.reward_train_epochs // 5,
                                                                     gamma=0.5)

        self.input_dim = encoder_feature_dim
        assert self.input_dim > 0, "reward dependency is wrong"

        # build regression layers with dim_hidden, input dim is feature_dim + action_dim, output dim is 1
        self.regression_layers = nn.Sequential()

        ## init hidden layers
        for i in range(len(self.dim_hidden)):
            if i == 0:
                self.regression_layers.add_module("linear_{}".format(i), nn.Linear(self.input_dim, self.dim_hidden[i]))
            else:
                self.regression_layers.add_module("linear_{}".format(i),
                                                  nn.Linear(self.dim_hidden[i - 1], self.dim_hidden[i]))
            self.regression_layers.add_module("relu_{}".format(i), nn.ReLU())

        ## init output layer
        self.regression_layers.add_module("linear_{}".format(len(self.dim_hidden)), nn.Linear(self.dim_hidden[-1], 1))
        # init optimizer
        self.regression_optimizer = torch.optim.Adam(self.regression_layers.parameters(), lr=learning_rate)
        if self.args.reward_sim_add_steplr:
            self.regression_scheduler = torch.optim.lr_scheduler.StepLR(self.regression_optimizer,
                                                                        step_size=self.args.reward_train_epochs // 5,
                                                                        gamma=0.5)

        if self.add_ssl:
            # build self-supervised-learning layers
            self.ssl_layers = nn.Sequential()
            for i in range(len(self.ssl_dim_hidden)):
                if i == 0:
                    self.ssl_layers.add_module("linear_{}".format(i), nn.Linear(self.input_dim, self.ssl_dim_hidden[i]))
                else:
                    self.ssl_layers.add_module("linear_{}".format(i),
                                               nn.Linear(self.ssl_dim_hidden[i - 1], self.ssl_dim_hidden[i]))
                self.ssl_layers.add_module("relu_{}".format(i), nn.ReLU())
            self.ssl_layers.add_module("linear_{}".format(len(self.ssl_dim_hidden)),
                                       nn.Linear(self.ssl_dim_hidden[-1], 4))
            self.ssl_layers.add_module("softmax_{}".format(len(self.ssl_dim_hidden)), nn.Softmax(dim=1))
            self.ssl_optimizer = torch.optim.Adam(self.ssl_layers.parameters(), lr=learning_rate)
            if self.args.reward_sim_add_steplr:
                self.ssl_scheduler = torch.optim.lr_scheduler.StepLR(self.ssl_optimizer,
                                                                     step_size=self.args.reward_train_epochs // 5,
                                                                     gamma=0.5)

    def forward(self, next_feature):
        # concate input as one tensor by the defined reward dependency
        th_in = self.encoder(next_feature)
        return self.regression_layers(th_in)

    def get_feature_state(self, next_obs):
        # concate input as one tensor by the defined reward dependency
        th_in = self.encoder(next_obs)
        return th_in

    def get_reward(self, next_feature):
        # concate input as one tensor by the defined reward dependency
        return self.forward(next_feature)

    def get_ssl_prediction(self, rotated_img):
        assert self.add_ssl, "no ssl layers"
        # concate input as one tensor by the defined reward dependency
        th_in = self.encoder(rotated_img)
        return self.ssl_layers(th_in)

    def get_feature_ssl_prediction(self, rotated_img):
        assert self.add_ssl, "no ssl layers"
        # concate input as one tensor by the defined reward dependency
        th_in = self.encoder(rotated_img)
        return th_in, self.ssl_layers(th_in)

    def state_dict(self):
        return_dict = {
            "encoder": self.encoder.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "regression_layers": self.regression_layers.state_dict(),
            "regression_optimizer": self.regression_optimizer.state_dict(),
        }
        if self.add_ssl:
            return_dict["ssl_layers"] = self.ssl_layers.state_dict()
            return_dict["ssl_optimizer"] = self.ssl_optimizer.state_dict()
        return return_dict

    def save(self, dir_name, file_name="reward_sim"):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        path = os.path.join(dir_name, file_name + ".pickle")
        torch.save(self.state_dict(), path)

    def load(self, dir_name, device, file_name="reward_sim"):
        path = os.path.join(dir_name, file_name + ".pickle")
        loaded_pickle = torch.load(path, map_location=device)
        self.encoder.load_state_dict(loaded_pickle["encoder"])
        self.encoder_optimizer.load_state_dict(loaded_pickle["encoder_optimizer"])
        self.regression_layers.load_state_dict(loaded_pickle["regression_layers"])
        self.regression_optimizer.load_state_dict(loaded_pickle["regression_optimizer"])
        if self.add_ssl:
            self.ssl_layers.load_state_dict(loaded_pickle["ssl_layers"])
            self.ssl_optimizer.load_state_dict(loaded_pickle["ssl_optimizer"])
        return self

    def update(self, replay_buffer, batch_size, step, wlogger=None):
        loss_sum = torch.zeros(1).to(self.device)
        reward_loss_sum = torch.zeros(1).to(self.device)
        if self.add_ssl:
            ssl_loss_sum = torch.zeros(1).to(self.device)
            ssl_acc_sum = torch.zeros(1).to(self.device)

        for _ in range(len(replay_buffer) // batch_size):   # iterate all data
            # reward, next_obs = replay_buffer.sample(batch_size)
            obses, actions, rewards, next_obses, not_dones, not_dones_no_max, obses_img = replay_buffer.sample(batch_size)
            loss = None
            # calculate reward loss
            predicted_reward = self.get_reward(obses_img)
            reward_loss = F.mse_loss(predicted_reward, rewards)
            loss = reward_loss

            # if self.add_ssl:
            #     # get true obs, labels, labels_number
            #     rotated_obs, labels, labels_number = utils.rotate(next_obs)
            #     # calculate ssl loss
            #     predicted_labels = self.get_ssl_prediction(rotated_obs)
            #     ssl_loss = F.cross_entropy(predicted_labels, labels)
            #
            #     # calculate ssl acc
            #     predicted_labels_number = th.argmax(predicted_labels, dim=1)
            #     correct_prediction = th.sum(predicted_labels_number == labels_number).float()
            #     ssl_acc = correct_prediction / labels.shape[0]
            #     ssl_acc_sum += ssl_acc.detach()
            #
            #     loss = reward_loss + self.ssl_coef * ssl_loss

            loss_sum += loss.detach()
            reward_loss_sum += reward_loss.detach()

            # if self.add_ssl:
            #     ssl_loss_sum += ssl_loss.detach()

            self.encoder_optimizer.zero_grad()
            self.regression_optimizer.zero_grad()
            if self.add_ssl:
                self.ssl_optimizer.zero_grad()

            loss.backward()

            self.encoder_optimizer.step()
            self.regression_optimizer.step()
            # if self.add_ssl:
            #     self.ssl_optimizer.step()

        if self.args.reward_sim_add_steplr:
            self.encoder_scheduler.step()
            self.regression_scheduler.step()
            # if self.add_ssl:
            #     self.ssl_scheduler.step()
        # logs
        if wlogger is not None:
            wandb_log = dict()
            wandb_log['train/loss'] = loss_sum / self.update_per_epoch
            wandb_log['train/reward_loss'] = reward_loss_sum / self.update_per_epoch
            if self.add_ssl:
                wandb_log['train/ssl_loss'] = ssl_loss_sum / self.update_per_epoch
                wandb_log['train/ssl_acc'] = ssl_acc_sum / self.update_per_epoch
            wlogger.wandb_log(wandb_log, step=step)
        return self


class RewardSimTENT(nn.Module):
    def __init__(self,
                 encoder_type,
                 encoder_feature_dim,
                 num_layers,
                 num_filters,
                 obs_shape,
                 dim_hidden,
                 learning_rate,
                 update_per_epoch,
                 device,
                 log_std_min=-10,
                 log_std_max=2,
                 ):
        super(RewardSimTENT, self).__init__()
        self.device = device
        self.update_per_epoch = update_per_epoch
        self.encoder_feature_dim = encoder_feature_dim
        # self.action_shape = action_shape
        self.learning_rate = learning_rate
        self.dim_hidden = dim_hidden.split("_")
        self.dim_hidden = [int(x) for x in self.dim_hidden]

        self.collect_start = 0

        self.augs_funcs = {}
        # self.data_augs = data_augs

        # aug_to_func = {
        #     'crop': rad.random_crop,
        #     'grayscale': rad.random_grayscale,
        #     'cutout': rad.random_cutout,
        #     'cutout_color': rad.random_cutout_color,
        #     'flip': rad.random_flip,
        #     'rotate': rad.random_rotation,
        #     'rand_conv': rad.random_convolution,
        #     'color_jitter': rad.random_color_jitter,
        #     'translate': rad.random_translate,
        #     'no_aug': rad.no_aug,
        # }
        #
        # for aug_name in self.data_augs.split('-'):
        #     assert aug_name in aug_to_func, 'invalid data aug string'
        #     self.augs_funcs[aug_name] = aug_to_func[aug_name]

        # create encoder
        self.encoder = make_encoder(
            encoder_type=encoder_type,
            obs_shape=obs_shape,
            feature_dim=self.encoder_feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
            output_logits=True,
            add_norm='group_norm',
        )
        self.input_dim = encoder_feature_dim
        assert self.input_dim > 0, "reward dependency is wrong"

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # build regression layers with dim_hidden, input dim is feature_dim + action_dim, output dim is 1
        self.regression_layers = nn.Sequential()
        ## init hidden layers
        for i in range(len(self.dim_hidden)):
            if i == 0:
                self.regression_layers.add_module("linear_{}".format(i), nn.Linear(self.input_dim, self.dim_hidden[i]))
            else:
                self.regression_layers.add_module("linear_{}".format(i),
                                                  nn.Linear(self.dim_hidden[i - 1], self.dim_hidden[i]))
            self.regression_layers.add_module("relu_{}".format(i), nn.ReLU())
        ## init output layer
        self.regression_layers.add_module("linear_{}".format(len(self.dim_hidden)), nn.Linear(self.dim_hidden[-1], 2))
        # init optimizer
        params = list(self.regression_layers.parameters()) + list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def forward(self, next_feature):
        # concate input as one tensor by the defined reward dependency
        th_in = self.encoder(next_feature)
        mu, log_std = self.regression_layers(th_in).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        distribution = torch.distributions.Normal(mu, log_std.exp())
        reward = distribution.rsample()
        return reward, mu, log_std

    # def get_feature_state(self, next_obs):
    #     # concate input as one tensor by the defined reward dependency
    #     th_in = self.encoder(next_obs)
    #     return th_in

    def get_reward(self, next_feature):
        reward, mu, log_std = self.forward(next_feature)
        return reward

    def state_dict(self):
        return_dict = {
            "encoder": self.encoder.state_dict(),
            "regression_layers": self.regression_layers.state_dict(),
            "regression_optimizer": self.optimizer.state_dict(),
        }
        return return_dict

    def save(self, dir_name, file_name="reward_sim_tent"):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        path = os.path.join(dir_name, file_name + ".pickle")
        torch.save(self.state_dict(), path)

    def load(self, dir_name, device, file_name="reward_sim_tent"):
        path = os.path.join(dir_name, file_name + ".pickle")
        loaded_pickle = torch.load(path, map_location=device)
        self.encoder.load_state_dict(loaded_pickle["encoder"])
        self.regression_layers.load_state_dict(loaded_pickle["regression_layers"])
        self.optimizer.load_state_dict(loaded_pickle["regression_optimizer"])
        return self

    def update(self, replay_buffer, batch_size, step, logger=None, gradient_update_steps=1):
        loss_sum = torch.zeros(1).to(self.device)
        log_std_mean = torch.zeros(1).to(self.device)
        ssl_loss_sum = torch.zeros(1).to(self.device)

        for _ in range(gradient_update_steps):
            # reward, next_obs = replay_buffer.sample_rad(self.augs_funcs)
            obses, actions, rewards, next_obses, not_dones, not_dones_no_max, obses_img = replay_buffer.sample(batch_size, sample_img=True)

            loss = None
            # calculate reward loss
            predicted_reward, mu, log_std = self.forward(obses_img)
            reward_loss = F.gaussian_nll_loss(predicted_reward, rewards, var=log_std.exp())
            loss = reward_loss

            loss_sum += loss.detach()
            # get ssl_loss, but it will not be used in training
            distribution = torch.distributions.Normal(mu, log_std.exp())
            ssl_loss = distribution.entropy().mean()
            ssl_loss_sum += ssl_loss.detach()

            log_std_mean += log_std.mean().detach()
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
        # logs
        if logger is not None:
            logger.log('train/img_reward_loss', loss_sum / (len(replay_buffer) // gradient_update_steps), step)

        return self

    def test_update(self, replay_buffer, step, update_times, buffer_mean_covr_tuple=None, wlogger=None, index=None):
        for update in range(update_times):
            obs, action, reward, next_obs, not_done = \
                replay_buffer.sample_rad_for_sim_within_range(self.collect_start, self.augs_funcs)

            # calculate ssl loss
            reward, mu, log_std = self.forward(next_obs)
            distribution = torch.distributions.Normal(mu, log_std.exp())
            # loss equals to the entropy of the distribution
            loss = distribution.entropy().mean()

            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self

    def cal_loss(self, replay_buffer, step, buffer_mean_covr_tuple=None, wlogger=None):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_rad_for_sim(self.augs_funcs)
        with torch.no_grad():
            # calculate reward loss
            predicted_reward, mu, log_std = self.forward(next_obs)
            reward_loss = F.mse_loss(predicted_reward, reward)
            # get ssl_loss
            distribution = torch.distributions.Normal(mu, log_std.exp())
            ssl_loss = distribution.entropy().mean()

        # logs
        if wlogger is not None:
            wandb_log = dict()
            wandb_log['train/reward_loss'] = reward_loss.cpu().detach().numpy()
            wandb_log['train/ssl_loss'] = ssl_loss.cpu().detach().numpy()
            wlogger.wandb_log(wandb_log, step=step)
        return self

    def update_collect_start(self, replay_buffer):
        if self.args.collect_then_train:
            self.collect_start = len(replay_buffer)
        else:
            raise ValueError("collect_then_train is False, cannot update collect_start")

    def reinit_optimizer(self):
        params = []
        names = []
        for nm, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
            if isinstance(m, nn.GroupNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

    def prepare_for_test_time_train(self):
        self.train()
        self.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, torch.nn.GroupNorm):
                m.requires_grad_(True)
                m.affine = True
        self.reinit_optimizer()

