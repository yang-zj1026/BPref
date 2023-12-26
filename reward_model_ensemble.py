import numpy as np
import torch
import torch.nn as nn
import os
import pickle

from encoder import make_encoder
from reward_model import RewardModel
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RewardModelEnsemble(RewardModel):
    def __init__(self, ds, da, obs_shape,
                 pre_image_size, image_size, data_augs, cfg,
                 add_ssl=False, ssl_update_freq=8, ssl_coeff=0.01,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,
                 large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1,
                 teacher_eps_mistake=0,
                 teacher_eps_skip=0,
                 teacher_eps_equal=0,
                 ):
        super().__init__(ds, da, obs_shape, pre_image_size, image_size, data_augs, cfg,
                         add_ssl=add_ssl, ssl_update_freq=ssl_update_freq, ssl_coeff=ssl_coeff,
                         lr=lr, mb_size=mb_size, size_segment=size_segment,
                         env_maker=env_maker, max_size=max_size, activation=activation,
                         capacity=capacity, large_batch=large_batch, label_margin=label_margin,
                         teacher_beta=teacher_beta, teacher_gamma=teacher_gamma,
                         teacher_eps_mistake=teacher_eps_mistake,
                         teacher_eps_skip=teacher_eps_skip,
                         teacher_eps_equal=teacher_eps_equal)
        self.cfg = cfg
        self.de = ensemble_size

        self.construct_image_model_ensemble()
        self.test_time_update_cnt = 0

    def construct_image_model_ensemble(self):
        self.reward_models = []
        self.encoders, self.regressions = [], []
        self.ssl_layers = []
        encoder_params, regression_params, ssl_params = [], [], []
        for i in range(self.de):
            dim_hidden = '256_256'.split("_")
            dim_hidden = [int(x) for x in dim_hidden]
            encoder_feature_dim = 50
            self.encoder_feature_dim = encoder_feature_dim
            encoder = make_encoder(
                encoder_type='pixel',
                obs_shape=self.obs_shape,
                feature_dim=encoder_feature_dim,
                num_layers=4,
                num_filters=32,
                output_logits=True,
                add_norm='batch_norm',
            )
            regression_layers = nn.Sequential()

            # init hidden layers
            for i in range(len(dim_hidden)):
                if i == 0:
                    regression_layers.add_module("linear_{}".format(i), nn.Linear(encoder_feature_dim, dim_hidden[i]))
                else:
                    regression_layers.add_module("linear_{}".format(i),
                                                 nn.Linear(dim_hidden[i - 1], dim_hidden[i]))
                regression_layers.add_module("relu_{}".format(i), nn.ReLU())

            # init output layer
            regression_layers.add_module("linear_{}".format(len(dim_hidden)), nn.Linear(dim_hidden[-1], 1))

            self.encoders.append(encoder)
            self.regressions.append(regression_layers)
            model = nn.Sequential(encoder, regression_layers)
            model.to(device)
            self.reward_models.append(model)

            encoder_params.extend(encoder.parameters())
            regression_params.extend(regression_layers.parameters())

            if self.add_ssl:
                ssl_dim_hidden = '64_64'.split("_")
                ssl_dim_hidden = [int(x) for x in ssl_dim_hidden]
                ssl_layers = nn.Sequential()
                for i in range(len(ssl_dim_hidden)):
                    if i == 0:
                        ssl_layers.add_module("linear_{}".format(i), nn.Linear(encoder_feature_dim, ssl_dim_hidden[i]))
                    else:
                        ssl_layers.add_module("linear_{}".format(i),
                                              nn.Linear(ssl_dim_hidden[i - 1], ssl_dim_hidden[i]))
                    ssl_layers.add_module("relu_{}".format(i), nn.ReLU())
                ssl_layers.add_module("linear_{}".format(len(ssl_dim_hidden)),
                                      nn.Linear(ssl_dim_hidden[-1], 4))
                ssl_layers.to(device)
                self.ssl_layers.append(ssl_layers)
                ssl_params.extend(ssl_layers.parameters())
                self.ssl_criteria = nn.CrossEntropyLoss(reduction='none')

        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=self.lr)
        self.regression_optimizer = torch.optim.Adam(regression_params, lr=self.lr)
        self.ssl_optimizer = torch.optim.Adam(ssl_params, lr=self.lr)

    def apply_data_augmentation(self, x):
        for aug, func in self.augs_funcs.items():
            # apply crop and cutout first
            if 'crop' in aug or 'cutout' in aug:
                x = func(x)
            elif 'translate' in aug:
                og_x = utils.center_crop_images(x, self.pre_image_size)
                x, rndm_idxs = func(og_x, self.image_size, return_random_idxs=True)
        return x

    def r_hat_image(self, x, to_numpy=False):
        r_hats = []
        x = self.apply_data_augmentation(x)
        x = torch.from_numpy(x).float().to(device)

        for reward_model in self.reward_models:
            r_hat = reward_model(x)
            r_hats.append(r_hat)
        r_hats = torch.stack(r_hats)
        if to_numpy:
            return np.mean(r_hats.detach().cpu().numpy())

        return torch.mean(r_hats)

    def r_hat_image_member(self, x, member_idx):
        r_hat = self.reward_models[member_idx](torch.from_numpy(x).float().to(device))
        return r_hat

    def r_hat_batch_image(self, obs):
        r_hats = []
        # data augmentation is applied to the input image x
        obs = self.apply_data_augmentation(obs)
        for reward_model in self.reward_models:
            r_hats.append(reward_model(torch.from_numpy(obs).float().to(device)))
        r_hats = torch.stack(r_hats)

        return torch.mean(r_hats, dim=0).detach().cpu().numpy()

    def get_feature_ssl_prediction(self, x):
        feature_states, ssl_states = [], []
        for i in range(self.de):
            feature_state = self.encoders[i](x)
            ssl_state = self.ssl_layers[i](feature_state)
            feature_states.append(feature_state)
            ssl_states.append(ssl_state)
        return torch.stack(feature_states), torch.stack(ssl_states)

    def get_ssl_loss(self, next_obs, reduction='mean'):
        ssl_losses = []
        for i in range(self.de):
            rotated_obs, labels, labels_number = utils.rotate(next_obs)
            feature = self.encoders[i](rotated_obs)
            predicted_labels = self.ssl_layers[i](feature)
            if reduction == 'none':
                ssl_loss = self.ssl_criteria(predicted_labels, labels)
            else:
                ssl_loss = self.ssl_criteria(predicted_labels, labels).mean()
            ssl_losses.append(ssl_loss)
        return torch.stack(ssl_losses)

    def get_ssl_acc(self, next_obs):
        ssl_accs = []
        for i in range(self.de):
            rotated_obs, labels, labels_number = utils.rotate(next_obs)
            feature = self.encoders[i](rotated_obs)
            predicted_labels = self.ssl_layers[i](feature)
            predicted_labels_number = torch.argmax(predicted_labels, dim=1)
            correct_prediction = torch.sum(predicted_labels_number == labels_number).float()
            ssl_acc = correct_prediction / labels.shape[0]
            ssl_accs.append(ssl_acc.detach().cpu().numpy())
        return np.array(ssl_accs)

    def train_reward_image(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])

        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))

        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0

        ssl_acc = None
        for epoch in range(num_epochs):
            self.encoder_optimizer.zero_grad()
            self.regression_optimizer.zero_grad()

            loss = 0.

            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            for member_idx in range(self.de):
                idxs = total_batch_index[member_idx][epoch * self.train_batch_size:last_index]

                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)

                if member_idx == 0:
                    total += labels.shape[0]

                # get logits
                sa_t_1 = sa_t_1.reshape(-1, *sa_t_1.shape[2:])
                sa_t_2 = sa_t_2.reshape(-1, *sa_t_2.shape[2:])

                # apply augmentation
                sa_t_1 = self.apply_data_augmentation(sa_t_1)
                sa_t_2 = self.apply_data_augmentation(sa_t_2)

                r_hat1 = self.r_hat_image_member(sa_t_1, member_idx)
                r_hat1 = r_hat1.reshape(-1, self.size_segment, 1)
                r_hat2 = self.r_hat_image_member(sa_t_2, member_idx)
                r_hat2 = r_hat2.reshape(-1, self.size_segment, 1)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member_idx].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member_idx] += correct

            if self.add_ssl and self.reward_update_steps % self.ssl_update_freq == 0:
                self.ssl_optimizer.zero_grad()
                # apply data augmentation
                sa_t_1 = torch.from_numpy(sa_t_1).to(device)
                # ssl loss and ssl acc are both list
                ssl_losses = self.get_ssl_loss(sa_t_1)
                ssl_acc = self.get_ssl_acc(sa_t_1)
                for ssl_loss in ssl_losses:
                    loss += self.ssl_coeff * ssl_loss
                print("SSL is updated, Loss: {:.4f}, ACC:{}".format(np.mean(ssl_losses.detach().cpu().numpy()),
                                                                    np.mean(ssl_acc)))

            loss.backward()
            self.encoder_optimizer.step()
            self.regression_optimizer.step()

            if self.add_ssl and self.reward_update_steps % self.ssl_update_freq == 0:
                self.ssl_optimizer.step()

            self.reward_update_steps += 1

        ensemble_acc = ensemble_acc / total
        info = {
            'acc': np.mean(ensemble_acc),
        }
        for member_idx in range(self.de):
            info['acc_{}'.format(member_idx)] = ensemble_acc[member_idx]

        if ssl_acc is not None:
            info['ssl_acc'] = np.mean(ssl_acc)
            for member_idx in range(self.de):
                info['ssl_acc_{}'.format(member_idx)] = ssl_acc[member_idx]

        return info

    # TODO: save_state_dict and load
    def state_dict(self, *args, **kwargs):
        return_dict = {}
        for i in range(self.de):
            return_dict["encoder_{}".format(i)] = self.encoders[i].state_dict()
            return_dict["regression_{}".format(i)] = self.regressions[i].state_dict()
        return_dict["encoder_optimizer"] = self.encoder_optimizer.state_dict()
        return_dict["regression_optimizer"] = self.regression_optimizer.state_dict()
        if self.add_ssl:
            for i in range(self.de):
                return_dict["ssl_layers_{}".format(i)] = self.ssl_layers[i].state_dict()
            return_dict["ssl_optimizer"] = self.ssl_optimizer.state_dict()
        return return_dict

    def get_buffer_feature_ssl_mean_covr(self, replay_buffer, device):
        # take out all obs
        buffer_len = len(replay_buffer)
        obs = replay_buffer.next_obses
        obs = self.apply_data_augmentation(obs)

        # get feature states by using reward sim
        iter_times = buffer_len // self.train_batch_size
        ensemble_size = self.de
        feature_states = np.zeros((ensemble_size, iter_times * self.train_batch_size, self.encoder_feature_dim))
        ssl_states = np.zeros((ensemble_size, iter_times * self.train_batch_size, 4))
        for i in range(iter_times):
            # get obs batch
            obs_batch = obs[i * self.train_batch_size: (i + 1) * self.train_batch_size]
            obs_batch = torch.as_tensor(obs_batch).to(device).float()
            # rotate obs_batch
            rotated_obs_batch, _, _ = utils.rotate(obs_batch)
            # get feature and ssl state
            with torch.no_grad():
                feature_state, ssl_state = self.get_feature_ssl_prediction(rotated_obs_batch)
            # store feature and ssl state
            feature_states[:, i * self.train_batch_size: (i + 1) * self.train_batch_size] = feature_state.cpu().numpy()
            ssl_states[:, i * self.train_batch_size: (i + 1) * self.train_batch_size] = ssl_state.cpu().numpy()

        # get mean and covariance
        feature_mean = np.zeros((ensemble_size, self.encoder_feature_dim))
        feature_covr = np.zeros((ensemble_size, self.encoder_feature_dim, self.encoder_feature_dim))
        ssl_mean = np.zeros((ensemble_size, 4))
        ssl_covr = np.zeros((ensemble_size, 4, 4))

        if self.cfg.new_alignment_coef:
            feature_mean_coef = np.zeros(ensemble_size)
            feature_covr_coef = np.zeros(ensemble_size)
            ssl_mean_coef = np.zeros(ensemble_size)
            ssl_covr_coef = np.zeros(ensemble_size)

        for i in range(ensemble_size):
            feature_state, ssl_state = feature_states[i], ssl_states[i]
            feature_mean[i], feature_covr[i] = utils.get_mean_covr_numpy(feature_state)
            ssl_mean[i], ssl_covr[i] = utils.get_mean_covr_numpy(ssl_state)

            if self.cfg.new_alignment_coef:
                NMD_SCALE_FACTOR = 0.5
                feature_mean_loss, feature_covr_loss = utils.get_mean_covr_loss(feature_state, self.train_batch_size,
                                                                                self.cfg)
                feature_mean_coef[i] = self.cfg.alignment_loss_coef_feature / feature_mean_loss * NMD_SCALE_FACTOR
                feature_covr_coef[i] = self.cfg.alignment_loss_coef_feature / feature_covr_loss

                ssl_mean_loss, ssl_covr_loss = utils.get_mean_covr_loss(ssl_state, self.train_batch_size, self.cfg)
                ssl_mean_coef[i] = self.cfg.alignment_loss_coef_ssl_feature / ssl_mean_loss * NMD_SCALE_FACTOR
                ssl_covr_coef[i] = self.cfg.alignment_loss_coef_ssl_feature / ssl_covr_loss

        # transform to tensor
        buffer_mean_covr_tuple = []
        for i in range(ensemble_size):
            feature_mean_tensor = torch.as_tensor(feature_mean[i]).to(device).float()
            feature_covr_tensor = torch.as_tensor(feature_covr[i]).to(device).float()
            ssl_mean_tensor = torch.as_tensor(ssl_mean[i]).to(device).float()
            ssl_covr_tensor = torch.as_tensor(ssl_covr[i]).to(device).float()
            buffer_mean_covr_tuple.append((feature_mean_tensor, feature_covr_tensor, ssl_mean_tensor, ssl_covr_tensor))

            if self.cfg.new_alignment_coef:
                print("feature_mean_coef: ", feature_mean_coef[i])
                print("feature_covr_coef: ", feature_covr_coef[i])
                print("ssl_mean_coef: ", ssl_mean_coef[i])
                print("ssl_covr_coef: ", ssl_covr_coef[i])
                buffer_mean_covr_tuple.append((feature_mean_tensor, feature_covr_tensor, ssl_mean_tensor, ssl_covr_tensor,
                                               feature_mean_coef[i], feature_covr_coef[i], ssl_mean_coef[i], ssl_covr_coef[i]))
            else:
                buffer_mean_covr_tuple.append((feature_mean_tensor, feature_covr_tensor, ssl_mean_tensor, ssl_covr_tensor))

        return buffer_mean_covr_tuple

    def save_image_model(self, model_dir, step, replay_buffer):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        path = os.path.join(model_dir, "reward_sim.pickle")
        torch.save(self.state_dict(), path)
        if self.add_ssl:
            buffer_mean_covr_tuple = self.get_buffer_feature_ssl_mean_covr(replay_buffer, device)
            save_path = os.path.join(model_dir, "buffer_mean_covr_tuple_new.pickle")
            with open(save_path, 'wb') as f:
                pickle.dump(buffer_mean_covr_tuple, f)
