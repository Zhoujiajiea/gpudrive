# import copy
# from typing import List, Union
# import torch
# from torch import nn
# from torch.distributions.utils import logits_to_probs
# import pufferlib.models
# from gpudrive.env import constants
# from huggingface_hub import PyTorchModelHubMixin
# from box import Box

# import madrona_gpudrive

# TOP_K_ROAD_POINTS = madrona_gpudrive.kMaxAgentMapObservationsCount


# def log_prob(logits, value):
#     value = value.long().unsqueeze(-1)
#     value, log_pmf = torch.broadcast_tensors(value, logits)
#     value = value[..., :1]
#     return log_pmf.gather(-1, value).squeeze(-1)


# def entropy(logits):
#     min_real = torch.finfo(logits.dtype).min
#     logits = torch.clamp(logits, min=min_real)
#     p_log_p = logits * logits_to_probs(logits)
#     return -p_log_p.sum(-1)


# def sample_logits(
#     logits: Union[torch.Tensor, List[torch.Tensor]],
#     action=None,
#     deterministic=False,
# ):
#     """Sample logits: Supports deterministic sampling."""

#     normalized_logits = [logits - logits.logsumexp(dim=-1, keepdim=True)]
#     logits = [logits]

#     if action is None:
#         if deterministic:
#             # Select the action with the maximum probability
#             action = torch.stack([l.argmax(dim=-1) for l in logits])
#         else:
#             # Sample actions stochastically from the logits
#             action = torch.stack(
#                 [
#                     torch.multinomial(logits_to_probs(l), 1).squeeze()
#                     for l in logits
#                 ]
#             )
#     else:
#         batch = logits[0].shape[0]
#         action = action.view(batch, -1).T

#     assert len(logits) == len(action)

#     logprob = torch.stack(
#         [log_prob(l, a) for l, a in zip(normalized_logits, action)]
#     ).T.sum(1)

#     logits_entropy = torch.stack(
#         [entropy(l) for l in normalized_logits]
#     ).T.sum(1)

#     return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)


# # class NeuralNet(
# #     nn.Module,
# #     PyTorchModelHubMixin,
# #     repo_url="https://github.com/Emerge-Lab/gpudrive",
# #     docs_url="https://arxiv.org/abs/2502.14706",
# #     tags=["ffn"],
# # ):
# #     def __init__(
# #         self,
# #         action_dim=91,  # Default: 7 * 13
# #         input_dim=64,
# #         hidden_dim=128,
# #         dropout=0.00,
# #         act_func="tanh",
# #         max_controlled_agents=64,
# #         obs_dim=2984,  # Size of the flattened observation vector (hardcoded)
# #         config=None,  # Optional config
# #     ):
# #         super().__init__()
# #         self.input_dim = input_dim
# #         self.hidden_dim = hidden_dim
# #         self.action_dim = action_dim
# #         self.max_controlled_agents = max_controlled_agents
# #         self.max_observable_agents = max_controlled_agents - 1
# #         self.obs_dim = obs_dim
# #         self.num_modes = 3  # Ego, partner, road graph
# #         self.dropout = dropout
# #         self.act_func = nn.Tanh() if act_func == "tanh" else nn.GELU()

# #         # Indices for unpacking the observation
# #         self.ego_state_idx = constants.EGO_FEAT_DIM
# #         self.partner_obs_idx = (
# #             constants.PARTNER_FEAT_DIM * self.max_controlled_agents
# #         )
# #         if config is not None:
# #             self.config = Box(config)
# #             if "reward_type" in self.config:
# #                 if self.config.reward_type == "reward_conditioned":
# #                     # Agents know their "type", consisting of three weights
# #                     # that determine the reward (collision, goal, off-road)
# #                     self.ego_state_idx += 3
# #                     self.partner_obs_idx += 3

# #             self.vbd_in_obs = self.config.vbd_in_obs

# #         # Calculate the VBD predictions size: 91 timesteps * 5 features = 455
# #         self.vbd_size = 91 * 5

# #         self.ego_embed = nn.Sequential(
# #             pufferlib.pytorch.layer_init(
# #                 nn.Linear(self.ego_state_idx, input_dim)
# #             ),
# #             nn.LayerNorm(input_dim),
# #             self.act_func,
# #             nn.Dropout(self.dropout),
# #             pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
# #         )

# #         self.partner_embed = nn.Sequential(
# #             pufferlib.pytorch.layer_init(
# #                 nn.Linear(constants.PARTNER_FEAT_DIM, input_dim)
# #             ),
# #             nn.LayerNorm(input_dim),
# #             self.act_func,
# #             nn.Dropout(self.dropout),
# #             pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
# #         )

# #         self.road_map_embed = nn.Sequential(
# #             pufferlib.pytorch.layer_init(
# #                 nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, input_dim)
# #             ),
# #             nn.LayerNorm(input_dim),
# #             self.act_func,
# #             nn.Dropout(self.dropout),
# #             pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
# #         )

# #         if self.vbd_in_obs:
# #             self.vbd_embed = nn.Sequential(
# #                 pufferlib.pytorch.layer_init(
# #                     nn.Linear(self.vbd_size, input_dim)
# #                 ),
# #                 nn.LayerNorm(input_dim),
# #                 self.act_func,
# #                 nn.Dropout(self.dropout),
# #                 pufferlib.pytorch.layer_init(nn.Linear(input_dim, input_dim)),
# #             )

# #         self.shared_embed = nn.Sequential(
# #             nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
# #             nn.Dropout(self.dropout),
# #         )

# #         self.actor = pufferlib.pytorch.layer_init(
# #             nn.Linear(hidden_dim, action_dim), std=0.01
# #         )
# #         self.critic = pufferlib.pytorch.layer_init(
# #             nn.Linear(hidden_dim, 1), std=1
# #         )

# #     def encode_observations(self, observation):

# #         if self.vbd_in_obs:
# #             (
# #                 ego_state,
# #                 road_objects,
# #                 road_graph,
# #                 vbd_predictions,
# #             ) = self.unpack_obs(observation)
# #         else:
# #             ego_state, road_objects, road_graph = self.unpack_obs(observation)

# #         # Embed the ego state
# #         ego_embed = self.ego_embed(ego_state)

# #         if self.vbd_in_obs:
# #             vbd_embed = self.vbd_embed(vbd_predictions)
# #             # Concatenate the VBD predictions with the ego state embedding
# #             ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

# #         # Max pool
# #         partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
# #         road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)

# #         # Concatenate the embeddings
# #         embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

# #         return self.shared_embed(embed)

# #     def forward(self, obs, action=None, deterministic=False):

# #         # Encode the observations
# #         hidden = self.encode_observations(obs)

# #         # Decode the actions
# #         value = self.critic(hidden)
# #         logits = self.actor(hidden)

# #         action, logprob, entropy = sample_logits(logits, action, deterministic)

# #         return action, logprob, entropy, value

# #     def unpack_obs(self, obs_flat):
# #         """
# #         Unpack the flattened observation into the ego state, visible simulator state.

# #         Args:
# #             obs_flat (torch.Tensor): Flattened observation tensor of shape (batch_size, obs_dim).

# #         Returns:
# #             tuple: If vbd_in_obs is True, returns (ego_state, road_objects, road_graph, vbd_predictions).
# #                 Otherwise, returns (ego_state, road_objects, road_graph).
# #         """

# #         # Unpack modalities
# #         ego_state = obs_flat[:, : self.ego_state_idx]
# #         partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]

# #         if self.vbd_in_obs:
# #             # Extract the VBD predictions (last 455 elements)
# #             vbd_predictions = obs_flat[:, -self.vbd_size :]

# #             # The rest (excluding ego_state and partner_obs) is the road graph
# #             roadgraph_obs = obs_flat[:, self.partner_obs_idx : -self.vbd_size]
# #         else:
# #             # Without VBD, all remaining elements are road graph observations
# #             roadgraph_obs = obs_flat[:, self.partner_obs_idx :]

# #         road_objects = partner_obs.view(
# #             -1, self.max_observable_agents, constants.PARTNER_FEAT_DIM
# #         )
# #         road_graph = roadgraph_obs.view(
# #             -1, TOP_K_ROAD_POINTS, constants.ROAD_GRAPH_FEAT_DIM
# #         )

# #         if self.vbd_in_obs:
# #             return ego_state, road_objects, road_graph, vbd_predictions
# #         else:
# #             return ego_state, road_objects, road_graph
# # ----------------- 粘贴这个完整的类定义 -----------------

# class NeuralNet(
#     nn.Module,
#     PyTorchModelHubMixin,
#     repo_url="https://github.com/Emerge-Lab/gpudrive",
#     docs_url="https://arxiv.org/abs/2502.14706",
#     tags=["ffn"],
# ):
#     def __init__(
#         self,
#         **kwargs,  # <-- 核心修改：接受所有参数
#     ):
#         """
#         重构后的 __init__。
#         它会从 kwargs (即 config.json) 中提取它认识的参数，
#         并忽略所有不认识的参数 (如 action_space_accel_disc)。
#         """
#         super().__init__()
        
#         # 1. 从 kwargs 中提取网络参数, 如果不存在则使用默认值
#         self.action_dim = kwargs.get("action_dim", 91)
#         self.input_dim = kwargs.get("input_dim", 64)
#         self.hidden_dim = kwargs.get("hidden_dim", 128)
#         self.dropout = kwargs.get("dropout", 0.0)
#         act_func = kwargs.get("act_func", "tanh")
#         self.max_controlled_agents = kwargs.get("max_controlled_agents", 64)
#         self.obs_dim = kwargs.get("obs_dim", 2984)
        
#         # 2. 将 *所有* 传入的参数 (包括环境参数) 存入 config
#         #    这样我们就可以安全地从中读取 vbd_in_obs 等参数
#         self.config = Box(kwargs)

#         # 3. 原始 __init__ 的剩余逻辑
#         self.max_observable_agents = self.max_controlled_agents - 1
#         self.num_modes = 3  # Ego, partner, road graph
#         self.act_func = nn.Tanh() if act_func == "tanh" else nn.GELU()

#         # Indices for unpacking the observation
#         self.ego_state_idx = constants.EGO_FEAT_DIM
#         self.partner_obs_idx = (
#             constants.PARTNER_FEAT_DIM * self.max_controlled_agents
#         )
        
#         # 这段逻辑现在可以安全地从 self.config 中读取
#         if "reward_type" in self.config:
#             if self.config.reward_type == "reward_conditioned":
#                 self.ego_state_idx += 3
#                 self.partner_obs_idx += 3

#         # 这是之前错误的根源, 现在可以安全地读取了
#         self.vbd_in_obs = self.config.get("vbd_in_obs", False)

#         # Calculate the VBD predictions size: 91 timesteps * 5 features = 455
#         self.vbd_size = 91 * 5

#         self.ego_embed = nn.Sequential(
#             pufferlib.pytorch.layer_init(
#                 nn.Linear(self.ego_state_idx, self.input_dim)
#             ),
#             nn.LayerNorm(self.input_dim),
#             self.act_func,
#             nn.Dropout(self.dropout),
#             pufferlib.pytorch.layer_init(nn.Linear(self.input_dim, self.input_dim)),
#         )

#         self.partner_embed = nn.Sequential(
#             pufferlib.pytorch.layer_init(
#                 nn.Linear(constants.PARTNER_FEAT_DIM, self.input_dim)
#             ),
#             nn.LayerNorm(self.input_dim),
#             self.act_func,
#             nn.Dropout(self.dropout),
#             pufferlib.pytorch.layer_init(nn.Linear(self.input_dim, self.input_dim)),
#         )

#         self.road_map_embed = nn.Sequential(
#             pufferlib.pytorch.layer_init(
#                 nn.Linear(constants.ROAD_GRAPH_FEAT_DIM, self.input_dim)
#             ),
#             nn.LayerNorm(self.input_dim),
#             self.act_func,
#             nn.Dropout(self.dropout),
#             pufferlib.pytorch.layer_init(nn.Linear(self.input_dim, self.input_dim)),
#         )

#         if self.vbd_in_obs:
#             self.vbd_embed = nn.Sequential(
#                 pufferlib.pytorch.layer_init(
#                     nn.Linear(self.vbd_size, self.input_dim)
#                 ),
#                 nn.LayerNorm(self.input_dim),
#                 self.act_func,
#                 nn.Dropout(self.dropout),
#                 pufferlib.pytorch.layer_init(nn.Linear(self.input_dim, self.input_dim)),
#             )

#         self.shared_embed = nn.Sequential(
#             nn.Linear(self.input_dim * self.num_modes, self.hidden_dim),
#             nn.Dropout(self.dropout),
#         )

#         self.actor = pufferlib.pytorch.layer_init(
#             nn.Linear(self.hidden_dim, self.action_dim), std=0.01
#         )
#         self.critic = pufferlib.pytorch.layer_init(
#             nn.Linear(self.hidden_dim, 1), std=1
#         )

#     def encode_observations(self, observation):

#         if self.vbd_in_obs:
#             (
#                 ego_state,
#                 road_objects,
#                 road_graph,
#                 vbd_predictions,
#             ) = self.unpack_obs(observation)
#         else:
#             ego_state, road_objects, road_graph = self.unpack_obs(observation)

#         # Embed the ego state
#         ego_embed = self.ego_embed(ego_state)

#         if self.vbd_in_obs:
#             vbd_embed = self.vbd_embed(vbd_predictions)
#             # Concatenate the VBD predictions with the ego state embedding
#             ego_embed = torch.cat([ego_embed, vbd_embed], dim=1)

#         # Max pool
#         partner_embed, _ = self.partner_embed(road_objects).max(dim=1)
#         road_map_embed, _ = self.road_map_embed(road_graph).max(dim=1)

#         # Concatenate the embeddings
#         embed = torch.cat([ego_embed, partner_embed, road_map_embed], dim=1)

#         return self.shared_embed(embed)

#     def forward(self, obs, action=None, deterministic=False):

#         # Encode the observations
#         hidden = self.encode_observations(obs)

#         # Decode the actions
#         value = self.critic(hidden)
#         logits = self.actor(hidden)

#         action, logprob, entropy = sample_logits(logits, action, deterministic)

#         return action, logprob, entropy, value

#     def unpack_obs(self, obs_flat):
#         """
#         Unpack the flattened observation into the ego state, visible simulator state.

#         Args:
#             obs_flat (torch.Tensor): Flattened observation tensor of shape (batch_size, obs_dim).

#         Returns:
#             tuple: If vbd_in_obs is True, returns (ego_state, road_objects, road_graph, vbd_predictions).
#                    Otherwise, returns (ego_state, road_objects, road_graph).
#         """

#         # Unpack modalities
#         ego_state = obs_flat[:, : self.ego_state_idx]
#         partner_obs = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]

#         if self.vbd_in_obs:
#             # Extract the VBD predictions (last 455 elements)
#             vbd_predictions = obs_flat[:, -self.vbd_size :]

#             # The rest (excluding ego_state and partner_obs) is the road graph
#             roadgraph_obs = obs_flat[:, self.partner_obs_idx : -self.vbd_size]
#         else:
#             # Without VBD, all remaining elements are road graph observations
#             roadgraph_obs = obs_flat[:, self.partner_obs_idx :]

#         road_objects = partner_obs.view(
#             -1, self.max_observable_agents, constants.PARTNER_FEAT_DIM
#         )
#         road_graph = roadgraph_obs.view(
#             -1, TOP_K_ROAD_POINTS, constants.ROAD_GRAPH_FEAT_DIM
#         )

#         if self.vbd_in_obs:
#             return ego_state, road_objects, road_graph, vbd_predictions
#         else:
#             return ego_state, road_objects, road_graph

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from box import Box
import madrona_gpudrive

# =============================================================================
# GIGAFLOW 组件: Deep Set Encoder
# =============================================================================
class DeepSetEncoder(nn.Module):
    """
    对集合数据进行编码: MLP(x) -> MaxPool -> GlobalFeature
    """
    def __init__(self, input_dim, hidden_dims=[64, 128], out_dim=128):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, Set_Size, Feature_Dim]
        x = self.mlp(x)
        # MaxPool over set dimension (dim=1)
        x = torch.max(x, dim=1)[0]
        return x

# =============================================================================
# GIGAFLOW 组件: Single Body Network (Backbone)
# =============================================================================
class GigaflowBody(nn.Module):
    """
    包含编码器和主干网络。Actor 和 Critic 各自持有一个实例。
    """
    def __init__(self, ego_dim, partner_dim, map_dim, max_agents, max_map_points):
        super().__init__()
        
        self.ego_dim = ego_dim
        self.partner_dim = partner_dim
        self.map_dim = map_dim
        self.max_agents = max_agents
        self.max_map_points = max_map_points

        # --- Encoders ---
        # 1. Ego Encoder
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        
        # 2. Partner Encoder (Deep Set)
        self.partner_encoder = DeepSetEncoder(input_dim=partner_dim, out_dim=128)
        
        # 3. Map Encoder (Deep Set)
        # GIGAFLOW 将车道和边界分开处理，这里为简化合并为一个 Map Encoder
        # 如果需要严格复现，可以拆分为 LaneEncoder 和 BoundEncoder
        self.map_encoder = DeepSetEncoder(input_dim=map_dim, out_dim=128)

        # --- Backbone (Huge MLP) ---
        concat_dim = 64 + 128 + 128
        self.backbone = nn.Sequential(
            nn.Linear(concat_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU()
        )

    def forward(self, ego, partners, road_map):
        # Encoders
        ego_emb = self.ego_encoder(ego)
        partner_emb = self.partner_encoder(partners)
        map_emb = self.map_encoder(road_map)
        
        # Concatenate & Backbone
        features = torch.cat([ego_emb, partner_emb, map_emb], dim=1)
        latent = self.backbone(features)
        return latent

# =============================================================================
# 主类: 替换原有的 LateFusionNet
# =============================================================================
class LateFusionNet(ActorCriticPolicy):
    """
    [The Trojan Horse]
    名字叫 LateFusionNet (为了兼容 imports)，但内核是 GIGAFLOW Deep Sets。
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        env_config: Box = None,
        exp_config: Box = None,
        **kwargs,
    ):
        # 禁用正交初始化 (GIGAFLOW 不需要)
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

        self.env_config = env_config if env_config else Box({})
        
        # --- 获取维度信息 (硬编码或从 config 读取) ---
        # 注意：这些必须与 C++ 层的输出一致
        self.ego_dim = 6  # Speed(1) + Size(3) + Goal(2)
        self.partner_dim = 13 # Pos(2) + Vel(2) + Heading + ...
        self.map_dim = 9      # Pos(2) + Type + ...
        
        # 从 madrona_gpudrive 获取最大数量
        self.max_agents = madrona_gpudrive.kMaxAgentCount 
        self.max_map_points = madrona_gpudrive.kMaxAgentMapObservationsCount

        # --- GIGAFLOW Architecture: Independent Actor & Critic ---
        self.actor_body = GigaflowBody(self.ego_dim, self.partner_dim, self.map_dim, self.max_agents, self.max_map_points)
        self.critic_body = GigaflowBody(self.ego_dim, self.partner_dim, self.map_dim, self.max_agents, self.max_map_points)

        # Heads
        self.action_head = self.action_net # 使用 SB3 基类定义的 action_net
        self.value_head = self.value_net   # 使用 SB3 基类定义的 value_net
        
        # 重定义 value_net 输入维度 (1024)
        self.value_net = nn.Linear(1024, 1)
        # 重定义 action_net 输入维度 (1024)
        # 注意: SB3 会自动根据 action_space 创建 action_net，我们需要覆盖其输入层
        if isinstance(action_space, spaces.Discrete):
            self.action_net = nn.Linear(1024, action_space.n)
        else:
            self.action_net = nn.Linear(1024, action_space.shape[0])

    def _unpack_obs(self, flat_obs):
        """
        核心逻辑: 将扁平的 Full Tensor 切割还原为结构化数据
        """
        # 1. Ego State
        ego = flat_obs[:, :self.ego_dim]
        
        # 2. Partners
        start_p = self.ego_dim
        end_p = start_p + (self.max_agents * self.partner_dim)
        partners = flat_obs[:, start_p:end_p].view(-1, self.max_agents, self.partner_dim)
        
        # 3. Map
        # 剩下的部分是 Map
        road_map = flat_obs[:, end_p:].view(-1, self.max_map_points, self.map_dim)
        
        return ego, partners, road_map

    def forward(self, obs, deterministic=False):
        """
        SB3 标准接口: 返回 action, value, log_prob
        """
        ego, partners, road_map = self._unpack_obs(obs)
        
        # Actor Path
        actor_latent = self.actor_body(ego, partners, road_map)
        distribution = self._get_action_dist_from_latent(actor_latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # Critic Path
        critic_latent = self.critic_body(ego, partners, road_map)
        values = self.value_net(critic_latent)
        
        return actions, values, log_prob

    def predict_values(self, obs):
        """
        仅计算 Value (用于 PPO 计算 Advantage)
        """
        ego, partners, road_map = self._unpack_obs(obs)
        critic_latent = self.critic_body(ego, partners, road_map)
        return self.value_net(critic_latent)

    def evaluate_actions(self, obs, actions):
        """
        PPO 更新时调用: 计算 log_prob, entropy, value
        """
        ego, partners, road_map = self._unpack_obs(obs)
        
        # Actor Path
        actor_latent = self.actor_body(ego, partners, road_map)
        distribution = self._get_action_dist_from_latent(actor_latent)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Critic Path
        critic_latent = self.critic_body(ego, partners, road_map)
        values = self.value_net(critic_latent)
        
        return values, log_prob, entropy

    def _get_action_dist_from_latent(self, latent_tensor):
        """
        辅助函数: 从 latent 生成分布
        """
        mean_actions = self.action_net(latent_tensor)
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    # 兼容性接口 (如果 SB3 试图调用 extract_features)
    def extract_features(self, obs):
        # 这是一个 Dummy 实现，因为我们在 forward 里自己处理了 feature extraction
        return obs