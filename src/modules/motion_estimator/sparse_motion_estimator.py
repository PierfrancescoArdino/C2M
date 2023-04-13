import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Sequential, GATv2Conv
from torch import Tensor
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F


class SparseMotionGenerator(nn.Module):
    def __init__(self, num_features_x=7, num_features_y=2, z_dim=64, h_dim=64, num_head=4,
                 input_scene_features=256, h_scene_features=256,  num_predicted_frames=5,
                 num_input_frames=1):
        super(SparseMotionGenerator, self).__init__()
        self.input_scene_features = input_scene_features
        self.h_scene_features = h_scene_features
        self.num_predicted_frames = num_predicted_frames
        self.num_input_frames = num_input_frames
        self.decoder = SparseMotionDecoder(h_dim, h_dim, z_dim, h_dim, num_predicted_frames, num_head)
        self.x_encoder = torch.nn.Sequential(torch.nn.Linear(num_features_x, int(h_dim/2)),
                                             torch.nn.LeakyReLU(0.2),
                                             torch.nn.Linear(int(h_dim/2), h_dim))
        self.y_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features_y, int(h_dim / 2)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(int(h_dim / 2), h_dim))
        self.encode_scene_features = \
            torch.nn.Sequential(torch.nn.Linear((h_dim + input_scene_features) * self.num_input_frames,
                                                int(input_scene_features / 2)),
                                torch.nn.BatchNorm1d(int(input_scene_features / 2)),
                                torch.nn.LeakyReLU(0.2),
                                torch.nn.Linear(int(input_scene_features / 2), h_dim * 2),
                                torch.nn.BatchNorm1d(h_dim * 2),
                                torch.nn.LeakyReLU(0.2),
                                torch.nn.Linear(h_dim * 2, h_dim))

    def forward(self, data, scene_features, latent):
        out_dict = {}
        x_n, targets_theta, edge_index = data.x, data.targets_theta, data.edge_index
        num_real_nodes = torch.LongTensor([data.num_real_nodes]) if isinstance(
            data.num_real_nodes, int) else data.num_real_nodes
        total_number_nodes = 0
        index_user_guidance = []
        for num_real_node in num_real_nodes:
            index_user_guidance.append(np.random.random_integers(0, int(num_real_node) - 1) + total_number_nodes)
            total_number_nodes += int(num_real_node)
        index_user_guidance = torch.LongTensor(index_user_guidance).to(x_n.device)
        u_n = torch.FloatTensor(data.num_nodes).zero_().to(x_n.device)
        u_n[index_user_guidance] = 1
        u_n = u_n.unsqueeze(1)

        x_n_mapped = self.x_encoder(x_n)
        theta_n_mapped = self.y_encoder(targets_theta)
        x_n_concat = self.encode_scene_features(torch.cat(torch.unbind(torch.cat([x_n_mapped, scene_features],
                                                                                 dim=2),
                                                                       1),
                                                          1))
        out_dict = self.decoder(x_n_concat, x_n[:, :2], theta_n_mapped, edge_index, u_n, latent, targets_theta)
        return out_dict

    @staticmethod
    def reparameterize(mu, logvar, mode="train"):
        if mode == 'train':
            std = torch.exp(0.5 * logvar)
            eps = Variable(std.data.new(std.size()).normal_())
            return mu + eps * std
        else:
            return mu

    def inference(self, data, z, index_user_guidance, scene_features):
        out_dict = {}
        x_n, targets_theta, edge_index = data.x, data.targets_theta, data.edge_index
        z = z.to(x_n.device)
        u_n = torch.FloatTensor(data.num_nodes).zero_().to(x_n.device)
        u_n[index_user_guidance]=1
        u_n = u_n.unsqueeze(1)
        x_n_mapped = self.x_encoder(x_n)
        theta_n_mapped = self.y_encoder(targets_theta)
        x_n_concat = self.encode_scene_features(torch.cat(torch.unbind(torch.cat([x_n_mapped, scene_features],
                                                                                 dim=2),
                                                                       1),
                                                          1))
        out_dict = self.decoder(x_n_concat, x_n[:, :2], theta_n_mapped, edge_index, u_n, z, targets_theta)
        return out_dict


class SparseMotionDecoder(torch.nn.Module):
    def __init__(self, num_features_x, num_features_y=2 , z_dim=2, h_dim=64, num_predicted_frames=5, num_head=4):
        super(SparseMotionDecoder, self).__init__()
        self.num_predicted_frames = num_predicted_frames
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.act = torch.nn.LeakyReLU(0.2)
        self.num_features_x = num_features_x
        self.num_features_y = num_features_y
        self.linear_z = torch.nn.Sequential(torch.nn.Linear(z_dim, h_dim * 2),
                                            torch.nn.LeakyReLU(0.2),
                                            torch.nn.Linear(h_dim * 2, h_dim))
        conv_time_steps = []
        loc_time_steps = []
        for i in range(self.num_predicted_frames):
            """layer = Sequential('x, edge_index',
                               [(GATv2Conv(self.num_features_x, self.num_features_x, add_self_loops=False,
                                           heads=6, concat=False),
                                 'x, edge_index -> x'),
                                nn.ReLU(inplace=True),
                                (GATv2Conv(self.num_features_x, self.num_features_x, add_self_loops=False,
                                           heads=6, concat=False),
                                 'x, edge_index -> x'),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.num_features_x, self.num_features_x)])
            conv_time_steps.append(layer)"""
            conv_time_steps.append(
                GATv2Conv(self.num_features_x, self.num_features_x, add_self_loops=False, heads=num_head, concat=False))
            # conv_time_steps.append(C2MMessagePassing(self.num_features_x, self.num_features_y, self.h_dim))
            fc_loc = torch.nn.Sequential(torch.nn.Linear(num_features_x, h_dim), torch.nn.LeakyReLU(0.2),
                                         torch.nn.Linear(h_dim, 3 * 2))
            fc_loc[2].weight.data.zero_()
            fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            loc_time_steps.append(fc_loc)
        self.conv_time_steps = nn.ModuleList(conv_time_steps)
        self.loc_time_steps = nn.ModuleList(loc_time_steps)

    def forward(self, x_n, x_start_pos, y_n, edge_index, u_n, z, targets_theta):
        for t in range(self.num_predicted_frames):
            y_n[:, t, ...] = self.linear_z(z[:, t, ...]) * (1 - u_n) + (y_n[:, t, ...] * u_n)
        out_dict = {}
        for t in range(self.num_predicted_frames):
            if t == 0:
                x = self.conv_time_steps[t](x_n, edge_index)
            else:
                x = self.conv_time_steps[t](x, edge_index)
            """if t == 0:
                x, y = self.conv_time_steps[t](x_n, y_n[:, t, ...], edge_index, u_n)
            else:
                y = y * (1 - u_n) + (y_n[:, t, ...] * u_n)
                x, y = self.conv_time_steps[t](x, y, edge_index, u_n)"""
            out_dict[f"theta_{t}"] = self.loc_time_steps[t](x) * (1 - u_n) + (targets_theta[:, t, ...] * u_n)
        return out_dict

    def theta2affine(self, theta):
        bs = theta.size()[0]
        size = torch.empty(bs, 2, 3)
        affine_matrix = torch.zeros_like(size).cuda()
        sx, sy = theta[:, 0], theta[:, 1]
        rotation, shear = theta[:, 2], theta[:, 3]
        tx, ty = theta[:, 4], theta[:, 5]
        sx = sx.clamp(min=0.6, max=1.4)
        sy = sy.clamp(min=0.6, max=1.4)
        affine_matrix[:, 0, 0] = sx * torch.cos(rotation)
        affine_matrix[:, 0, 1] = - sy * torch.sin(rotation + shear)
        affine_matrix[:, 0, 2] = tx
        affine_matrix[:, 1, 0] = sx * torch.sin(rotation)
        affine_matrix[:, 1, 1] = sy * torch.cos(rotation + shear)
        affine_matrix[:, 1, 2] = ty
        affine_matrix = affine_matrix.view(-1, 2, 3)
        return affine_matrix


class C2MMessagePassing(MessagePassing):
    def __init__(self, in_channels_xn, in_channels_yn, h_dim):
        super(C2MMessagePassing, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.in_channels_xn = in_channels_xn
        self.in_channels_yn = in_channels_yn
        self.h_dim = h_dim
        self.lin_xn = torch.nn.Linear(in_channels_xn + in_channels_yn, in_channels_xn)
        self.lin_yn = torch.nn.Linear(in_channels_xn + in_channels_yn, in_channels_yn)
        self.message_mlp_x = torch.nn.Sequential(torch.nn.Linear(in_channels_xn + in_channels_xn, self.h_dim),
                                                 torch.nn.LeakyReLU(0.2),
                                                 torch.nn.Linear(self.h_dim, in_channels_xn))
        self.message_mlp_y = torch.nn.Sequential(torch.nn.Linear(in_channels_yn + in_channels_yn, self.h_dim),
                                                 torch.nn.LeakyReLU(0.2),
                                                 torch.nn.Linear(self.h_dim, in_channels_yn))
        self.update_mlp_x = torch.nn.Sequential(torch.nn.Linear(in_channels_xn + in_channels_xn, self.h_dim),
                                                torch.nn.LeakyReLU(0.2),
                                                torch.nn.Linear(self.h_dim, in_channels_xn))
        self.update_mlp_y = torch.nn.Sequential(torch.nn.Linear(in_channels_yn + in_channels_yn, self.h_dim),
                                                torch.nn.LeakyReLU(0.2),
                                                torch.nn.Linear(self.h_dim, in_channels_yn))

    def forward(self, x, y_n, edge_index, u_n):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        self.x_n = x
        self.y_n = y_n
        self.u_n = u_n

        # Step 2: Linearly transform node feature matrix.
        self.x_n = self.lin_xn(torch.cat([x, y_n], dim=1))
        self.y_n = self.lin_yn(torch.cat([x, y_n], dim=1))
        # Step 3: Compute normalization.
        edge_index = edge_index.long()
        updated_x_n = self.propagate(edge_index, output=self.x_n)
        # node choosen by the user does not have the updated y_n, go back to the original one
        updated_y_n = self.propagate(edge_index, output=self.y_n) * (1 - self.u_n) + (y_n * self.u_n)
        # Step 4-5: Start propagating messages.
        return updated_x_n, updated_y_n

    def message(self, output_i, output_j):
        # x_j has shape [E, out_channels]
        if output_j.shape[1] == self.in_channels_xn:
            return self.message_mlp_x(torch.cat([output_i, output_j], dim=1))
        elif output_j.shape[1] == self.in_channels_yn:
            return self.message_mlp_y(torch.cat([output_i, output_j], dim=1))

    def update(self, inputs: Tensor) -> Tensor:
        if inputs.shape[1] == self.in_channels_xn:
            return self.update_mlp_x(torch.cat([self.x_n, inputs], dim=1))
        elif inputs.shape[1] == self.in_channels_yn:
            return self.update_mlp_y(torch.cat([self.y_n, inputs], dim=1))