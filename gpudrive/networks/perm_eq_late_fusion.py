# import copy
# from typing import Callable, Dict, List, Optional, Tuple, Type, Union
# import os

# import torch
# import torch.nn.functional as F
# from box import Box
# from gymnasium import spaces
# from stable_baselines3.common.policies import ActorCriticPolicy
# from torch import nn
# import wandb

# # Import env wrapper that makes gym env compatible with stable-baselines3
# from gpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
# from gpudrive.env.config import EnvConfig
# from gpudrive.env import constants
# import madrona_gpudrive
# TOP_K_ROAD_POINTS = madrona_gpudrive.kMaxAgentMapObservationsCount

# class LateFusionNet(nn.Module):
#     """Processes the env observation using a late fusion architecture."""

#     # def __init__(
#     #     self,
#     #     observation_space: spaces.Box,
#     #     env_config: EnvConfig,
#     #     exp_config,
#     # ):
#     #     super().__init__()

#     #     self.config = env_config
#     #     self.net_config = exp_config

#     #     # Unpack feature dimensions
#     #     self.ego_input_dim = constants.EGO_FEAT_DIM if self.config.ego_state else 0
#     #     self.ro_input_dim = constants.PARTNER_FEAT_DIM if self.config.partner_obs else 0
        
#     #     # [GIGAFLOW] 动态获取或修正 road_graph 输入维度
#     #     # 原代码直接使用 constants.ROAD_GRAPH_FEAT_DIM (可能为 13)
#     #     # 但现在环境似乎没有产生那么多特征。
#     #     # 我们暂时保持引用常量，但在下面 _unpack_obs 里做容错
#     #     self.rg_input_dim = constants.ROAD_GRAPH_FEAT_DIM if self.config.road_map_obs else 0
#     #     self.ego_state_idx = self.ego_input_dim
#     #     self.ro_max = self.config.max_num_agents_in_scene-1
#     #     self.rg_max = self.config.roadgraph_top_k

#     #     # Network architectures
#     #     self.arch_ego_state = self.net_config.ego_state_layers
#     #     self.arch_road_objects = self.net_config.road_object_layers
#     #     self.arch_road_graph = self.net_config.road_graph_layers
#     #     self.arch_shared_net = self.net_config.shared_layers
#     #     self.act_func = (
#     #         nn.Tanh() if self.net_config.act_func == "tanh" else nn.ReLU()
#     #     )
#     #     self.dropout = self.net_config.dropout

#     #     # Save output dimensions, used to create the action distribution & value
#     #     self.latent_dim_pi = self.net_config.last_layer_dim_pi
#     #     self.latent_dim_vf = self.net_config.last_layer_dim_vf

#     #     # If using max pool across object dim
#     #     self.shared_net_input_dim = (
#     #         self.net_config.ego_state_layers[-1]
#     #         + self.net_config.road_object_layers[-1]
#     #         + self.net_config.road_graph_layers[-1]
#     #     )

#     #     # Build the networks
#     #     # Actor network
#     #     self.actor_ego_state_net = self._build_network(
#     #         input_dim=self.ego_input_dim,
#     #         net_arch=self.arch_ego_state,
#     #     )
#     #     self.actor_ro_net = self._build_network(
#     #         input_dim=self.ro_input_dim,
#     #         net_arch=self.arch_road_objects,
#     #     )
#     #     self.actor_rg_net = self._build_network(
#     #         input_dim=self.rg_input_dim,
#     #         net_arch=self.arch_road_graph,
#     #     )
#     #     self.actor_out_net = self._build_out_network(
#     #         input_dim=self.shared_net_input_dim,
#     #         output_dim=self.latent_dim_pi,
#     #         net_arch=self.arch_shared_net,
#     #     )

#     #     # Value network
#     #     self.val_ego_state_net = copy.deepcopy(self.actor_ego_state_net)
#     #     self.val_ro_net = copy.deepcopy(self.actor_ro_net)
#     #     self.val_rg_net = copy.deepcopy(self.actor_rg_net)
#     #     self.val_out_net = self._build_out_network(
#     #         input_dim=self.shared_net_input_dim,
#     #         output_dim=self.latent_dim_vf,
#     #         net_arch=self.arch_shared_net,
#     #     )
#     def __init__(
#         self,
#         observation_space: spaces.Box,
#         env_config: EnvConfig,
#         exp_config,
#     ):
#         super().__init__()

#         self.config = env_config
#         self.net_config = exp_config

#         # -----------------------------------------------------------
#         # [GIGAFLOW 修改 A] 维度定义 (严格对应 C++ Sim 925维)
#         # -----------------------------------------------------------
        
#         # 1. 自车特征: 6 维
#         # (Speed, Len, Wid, GoalX, GoalY, Collision)
#         self.ego_input_dim = 6 if self.config.ego_state else 0
        
#         # 2. 邻居特征: 9 维 (原代码可能错写为其他值)
#         # (PosX, PosY, VelX, VelY, Yaw, Len, Wid, Type, Padding)
#         self.ro_input_dim = 9 if self.config.partner_obs else 0
        
#         # 3. 地图特征: 10 维 (这是最关键的修改，必须是 10)
#         # (PosX, PosY, ScaleX, ScaleY, Yaw, Type, Padding x 4)
#         self.rg_input_dim = 10 if self.config.road_map_obs else 0

#         # -----------------------------------------------------------
#         # [GIGAFLOW 修改 B] 数量定义 & 索引计算
#         # -----------------------------------------------------------
        
#         # 数量定义
#         # 邻居数量: 32 - 1 = 31
#         self.num_partners = self.config.max_num_agents_in_scene - 1 
#         # 地图点数量: 64 (从 C++ 绑定获取)
#         self.num_road_points = TOP_K_ROAD_POINTS 

#         # 切片索引计算
#         # Index 0 ~ 6
#         self.ego_state_idx = self.ego_input_dim
        
#         # Index 6 ~ 285 (6 + 31*9)
#         self.partner_obs_idx = self.ego_state_idx + (self.num_partners * self.ro_input_dim)
        
#         # 用于 MaxPool 的 kernel size
#         self.ro_max = self.num_partners
#         self.rg_max = self.num_road_points

#         # Network architectures
#         self.arch_ego_state = self.net_config.ego_state_layers
#         self.arch_road_objects = self.net_config.road_object_layers
#         self.arch_road_graph = self.net_config.road_graph_layers
#         self.arch_shared_net = self.net_config.shared_layers
#         self.act_func = (
#             nn.Tanh() if self.net_config.act_func == "tanh" else nn.ReLU()
#         )
#         self.dropout = self.net_config.dropout

#         # Save output dimensions
#         self.latent_dim_pi = self.net_config.last_layer_dim_pi
#         self.latent_dim_vf = self.net_config.last_layer_dim_vf

#         self.shared_net_input_dim = (
#             self.net_config.ego_state_layers[-1]
#             + self.net_config.road_object_layers[-1]
#             + self.net_config.road_graph_layers[-1]
#         )

#         # Build the networks
#         # 注意: 这里的 input_dim 已经被上面修正为 6, 9, 10
#         self.actor_ego_state_net = self._build_network(
#             input_dim=self.ego_input_dim,
#             net_arch=self.arch_ego_state,
#         )
#         self.actor_ro_net = self._build_network(
#             input_dim=self.ro_input_dim,
#             net_arch=self.arch_road_objects,
#         )
#         self.actor_rg_net = self._build_network(
#             input_dim=self.rg_input_dim,
#             net_arch=self.arch_road_graph,
#         )
#         self.actor_out_net = self._build_out_network(
#             input_dim=self.shared_net_input_dim,
#             output_dim=self.latent_dim_pi,
#             net_arch=self.arch_shared_net,
#         )

#         # Value network (Copy structure)
#         self.val_ego_state_net = copy.deepcopy(self.actor_ego_state_net)
#         self.val_ro_net = copy.deepcopy(self.actor_ro_net)
#         self.val_rg_net = copy.deepcopy(self.actor_rg_net)
#         self.val_out_net = self._build_out_network(
#             input_dim=self.shared_net_input_dim,
#             output_dim=self.latent_dim_vf,
#             net_arch=self.arch_shared_net,
#         )

#     def _build_network(
#         self,
#         input_dim: int,
#         net_arch: List[int],
#     ) -> nn.Module:
#         """Build a network with the specified architecture."""
#         layers = []
#         last_dim = input_dim
#         for layer_dim in net_arch:
#             layers.append(nn.Linear(last_dim, layer_dim))
#             layers.append(nn.Dropout(self.dropout))
#             layers.append(nn.LayerNorm(layer_dim))
#             layers.append(self.act_func)
#             last_dim = layer_dim
#         return nn.Sequential(*layers)

#     def _build_out_network(
#         self, input_dim: int, output_dim: int, net_arch: List[int]
#     ):
#         """Create the output network architecture."""
#         layers = []
#         prev_dim = input_dim
#         for layer_dim in net_arch:
#             layers.append(nn.Linear(prev_dim, layer_dim))
#             layers.append(nn.LayerNorm(layer_dim))
#             layers.append(self.act_func)
#             layers.append(nn.Dropout(self.dropout))
#             prev_dim = layer_dim

#         # Add final layer
#         layers.append(nn.Linear(prev_dim, output_dim))
#         layers.append(nn.LayerNorm(output_dim))

#         return nn.Sequential(*layers)

#     def forward(
#         self, features: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Args:
#             features (torch.Tensor): input tensor of shape (batch_size, feature_dim)
#         Return:
#             (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
#             If all layers are shared, then ``latent_policy == latent_value``
#         """
#         return self.forward_actor(features), self.forward_critic(features)

#     def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
#         """Forward step for the actor network."""

#         # Unpack observation
#         ego_state, road_objects, road_graph = self._unpack_obs(features)

#         # [GIGAFLOW 优化] Feature Dropout
#         # 训练时随机丢弃特征，提升泛化性并减轻显存压力（如果后续有优化）
#         road_graph = self._apply_set_dropout(road_graph, drop_prob=0.5)   # 50% 丢弃率
#         road_objects = self._apply_set_dropout(road_objects, drop_prob=0.2) # 20% 丢弃率

#         # Embed features
#         ego_state = self.actor_ego_state_net(ego_state)
#         road_objects = self.actor_ro_net(road_objects)
#         road_graph = self.actor_rg_net(road_graph)

#         # Max pooling across the object dimension
#         # (M, E) -> (1, E) (max pool across features)
#         road_objects = F.max_pool1d(
#             road_objects.permute(0, 2, 1), kernel_size=self.ro_max
#         ).squeeze(-1)
#         road_graph = F.max_pool1d(
#             road_graph.permute(0, 2, 1), kernel_size=self.rg_max
#         ).squeeze(-1)

#         # Concatenate processed ego state and observation and pass through the output layer
#         out = self.actor_out_net(
#             torch.cat((ego_state, road_objects, road_graph), dim=1)
#         )

#         return out

#     def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
#         """Forward step for the value network."""

#         ego_state, road_objects, road_graph = self._unpack_obs(features)

#         # [GIGAFLOW 优化] Feature Dropout (与 Actor 保持一致)
#         road_graph = self._apply_set_dropout(road_graph, drop_prob=0.5)
#         road_objects = self._apply_set_dropout(road_objects, drop_prob=0.2)

#         # Embed features
#         ego_state = self.val_ego_state_net(ego_state)
#         road_objects = self.val_ro_net(road_objects)
#         road_graph = self.val_rg_net(road_graph)

#         # Max pooling across the object dimension
#         # (M, E) -> (1, E) (max pool across features)
#         road_objects = F.max_pool1d(
#             road_objects.permute(0, 2, 1), kernel_size=self.ro_max
#         ).squeeze(-1)
#         road_graph = F.max_pool1d(
#             road_graph.permute(0, 2, 1), kernel_size=self.rg_max
#         ).squeeze(-1)

#         # Concatenate processed ego state and observation and pass through the output layer
#         out = self.val_out_net(
#             torch.cat((ego_state, road_objects, road_graph), dim=1)
#         )

#         return out

#     def _unpack_obs(self, obs_flat):
#         """
#         [GIGAFLOW FIXED] Unpack Logic for 925-dim Observation
#         Input: (Batch, 925)
#         Output: Ego(B,6), Partner(B,31,9), Map(B,64,10)
#         """
#         # 1. 切 Ego [0 : 6]
#         ego_state = obs_flat[:, : self.ego_state_idx]

#         # 2. 切 Partner [6 : 285]
#         # 注意：这里使用我们在 __init__ 中计算好的 partner_obs_idx
#         partner_obs_flat = obs_flat[:, self.ego_state_idx : self.partner_obs_idx]

#         # 3. 切 Map [285 : ]
#         # 如果有 VBD，逻辑要稍微调整 (假设 VBD 在最后)
#         if hasattr(self, 'vbd_in_obs') and self.vbd_in_obs:
#             roadgraph_obs_flat = obs_flat[:, self.partner_obs_idx : -self.vbd_size]
#         else:
#             roadgraph_obs_flat = obs_flat[:, self.partner_obs_idx :]

#         # 4. Reshape / View (恢复结构)
        
#         # Partner -> (Batch, 31, 9)
#         road_objects = None
#         if self.ro_input_dim > 0:
#             road_objects = partner_obs_flat.view(
#                 -1, self.num_partners, self.ro_input_dim
#             )

#         # Map -> (Batch, 64, 10)
#         # [重要] 这里的 dim 必须是 10，否则报错
#         road_graph = None
#         if self.rg_input_dim > 0:
#             road_graph = roadgraph_obs_flat.view(
#                 -1, self.num_road_points, self.rg_input_dim
#             )
            
#         return ego_state, road_objects, road_graph
        
#     # [GIGAFLOW NEW] 添加此方法到 LateFusionNet 类中，位于 _unpack_obs 之后
#     def _apply_set_dropout(self, features: torch.Tensor, drop_prob: float) -> torch.Tensor:
#         """
#         [GIGAFLOW] Apply random dropout to a set of features (masking out entire objects).
#         features: (Batch, Num_Objects, Feat_Dim)
#         """
#         if not self.training or drop_prob <= 0.0:
#             return features
            
#         B, N, _ = features.shape
#         # 生成掩码: 1 代表保留, 0 代表丢弃
#         keep_prob = 1.0 - drop_prob
#         # 生成伯努利分布掩码
#         mask = torch.bernoulli(torch.full((B, N, 1), keep_prob, device=features.device))
        
#         # 应用掩码 (被丢弃的特征变为全0)
#         # 注意: GIGAFLOW 论文中使用 Deep Sets (MaxPool)，全0特征在 max_pool 时会被忽略（如果使用ReLU）
#         return features * mask


# class LateFusionPolicy(ActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         env_config: Box,
#         exp_config: Box,
#         action_space: spaces.Space,
#         lr_schedule: Callable[[float], float],
#         mlp_class: Type[LateFusionNet] = LateFusionNet,
#         mlp_config: Optional[Box] = None,
#         *args,
#         **kwargs,
#     ):
#         # Disable orthogonal initialization
#         kwargs["ortho_init"] = False
#         self.observation_space = observation_space
#         self.env_config = env_config
#         self.exp_config = exp_config
#         self.mlp_class = mlp_class
#         self.mlp_config = mlp_config if mlp_config is not None else Box({})
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             # Pass remaining arguments to base class
#             *args,
#             **kwargs,
#         )

#     def _build_mlp_extractor(self) -> None:
#         # Build the network architecture
#         self.mlp_extractor = self.mlp_class(
#             self.observation_space,
#             self.env_config,
#             self.exp_config,
#             **self.mlp_config,
#         )
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from box import Box
import madrona_gpudrive
from gpudrive.env import constants

# =============================================================================
# GIGAFLOW 组件: Deep Set Encoder (置换不变性核心)
# =============================================================================
class DeepSetEncoder(nn.Module):
    """
    对集合数据进行编码: 
    Input (B, N, Fin) -> MLP -> (B, N, Hidden) -> MaxPool -> (B, Hidden)
    """
    def __init__(self, input_dim, hidden_dims=[256, 256], out_dim=256):
        super().__init__()
        layers = []
        curr_dim = input_dim
        
        # Point-wise MLP (处理集合中的每个元素)
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim)) # GIGAFLOW 推荐使用 LN
            layers.append(nn.ReLU())
            curr_dim = h_dim
        
        # 映射到输出维度前的一层
        layers.append(nn.Linear(curr_dim, out_dim))
        layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        # x shape: [Batch, Set_Size, Feature_Dim]
        x = self.mlp(x)
        
        # [关键] MaxPool over set dimension (dim=1)
        # 这保证了输入顺序不影响输出结果 (Permutation Invariance)
        # 同时也处理了 Padding (只要 Padding 值足够小，如 0，ReLU 后为 0)
        x = torch.max(x, dim=1)[0] 
        return x

# =============================================================================
# GIGAFLOW 组件: Single Body Network (独立骨干)
# =============================================================================
class GigaflowBackbone(nn.Module):
    """
    这是 GIGAFLOW 的"大脑"。Actor 和 Critic 将各自实例化一个这样的网络。
    """
    def __init__(self, ego_dim, partner_dim, map_dim, max_agents, max_map_points, net_config=None):
        super().__init__()
        
        # --- 1. 读取配置 (如果有) 或使用默认大容量 ---
        if net_config is not None:
            # 使用 getattr 提供默认值，防止 yaml 中缺少某些字段
            ego_layers = getattr(net_config, "ego_state_layers", [128, 128])
            partner_layers = getattr(net_config, "road_object_layers", [128, 256])
            map_layers = getattr(net_config, "road_graph_layers", [128, 256])
            fusion_layers = getattr(net_config, "shared_layers", [1024, 1024, 1024])
        else:
            # 默认 High-Capacity
            ego_layers = [128, 128]
            partner_layers = [128, 256]
            map_layers = [128, 256]
            fusion_layers = [1024, 1024, 1024]

        # --- 2. Encoders (特征提取) ---
        
        # Ego Encoder: 处理自车状态
        layers = []
        curr = ego_dim
        for h in ego_layers:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            curr = h
        self.ego_encoder = nn.Sequential(*layers)
        ego_out_dim = curr
        
        # Partner Encoder (Deep Set): 处理邻居车辆
        # 输入维度 9 (pos, vel, size, etc.) -> 映射到高维
        self.partner_encoder = DeepSetEncoder(
            input_dim=partner_dim, 
            hidden_dims=partner_layers[:-1], 
            out_dim=partner_layers[-1]
        )
        partner_out_dim = partner_layers[-1]
        
        # Map Encoder (Deep Set): 处理地图点
        # 输入维度 10 (pos, scale, type, etc.) -> 映射到高维
        self.map_encoder = DeepSetEncoder(
            input_dim=map_dim, 
            hidden_dims=map_layers[:-1], 
            out_dim=map_layers[-1]
        )
        map_out_dim = map_layers[-1]

        # --- 3. Late Fusion & Large MLP (大容量融合) ---
        
        concat_dim = ego_out_dim + partner_out_dim + map_out_dim
        
        f_layers = []
        curr = concat_dim
        for h in fusion_layers:
            f_layers.append(nn.Linear(curr, h))
            f_layers.append(nn.LayerNorm(h))
            f_layers.append(nn.ReLU())
            curr = h
            
        self.fusion_mlp = nn.Sequential(*f_layers)
        self.latent_dim = curr

    def forward(self, ego, partners, road_map):
        # 1. 独立编码
        ego_emb = self.ego_encoder(ego)          # (B, H_ego)
        partner_emb = self.partner_encoder(partners) # (B, H_partner)
        map_emb = self.map_encoder(road_map)     # (B, H_map)
        
        # 2. 拼接 (Late Fusion)
        features = torch.cat([ego_emb, partner_emb, map_emb], dim=1)
        
        # 3. 深度推理
        latent = self.fusion_mlp(features)
        return latent

# =============================================================================
# 主类: 替换 SB3 Policy
# =============================================================================
class LateFusionNet(ActorCriticPolicy):
    """
    GIGAFLOW 完整策略网络实现。
    继承自 SB3 ActorCriticPolicy，但完全重写了特征提取和前向传播。
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        env_config: Box = None,
        exp_config: Box = None,
        # [修复关键点] 显式接收旧接口传来的参数，防止它们进入 **kwargs 导致父类报错
        mlp_class=None,
        mlp_config=None, 
        **kwargs,
    ):
        # 禁用正交初始化 (GIGAFLOW 推荐使用默认初始化或 Xavier)
        kwargs["ortho_init"] = False
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

        self.env_config = env_config if env_config else Box({})
        self.net_config = exp_config # 保存配置用于 Backbone
        
        # --- 维度定义 (必须与 env_torch.py 一致) ---
        # 参见 env_torch.py 中的 _normalize_reconstructed_obs 方法
        self.ego_dim = 6  
        self.partner_dim = 9
        self.map_dim = 10 
        
        self.max_agents = constants.MAX_PARTNER_COUNT # 31
        self.max_map_points = constants.MAX_ROAD_OBS_COUNT # 64

        # =================================================================
        # [关键] 完全分离的 Actor 和 Critic 网络
        # =================================================================
        self.actor_body = GigaflowBackbone(
            self.ego_dim, self.partner_dim, self.map_dim, 
            self.max_agents, self.max_map_points,
            net_config=self.net_config
        )
        
        self.critic_body = GigaflowBackbone(
            self.ego_dim, self.partner_dim, self.map_dim, 
            self.max_agents, self.max_map_points,
            net_config=self.net_config
        )

        # Heads (输出层)
        latent_dim = self.actor_body.latent_dim
        
        # Value Head
        self.value_net = nn.Linear(latent_dim, 1)
        
        # Action Head
        if isinstance(action_space, spaces.Discrete):
            self.action_net = nn.Linear(latent_dim, action_space.n)
        else:
            self.action_net = nn.Linear(latent_dim, action_space.shape[0])

    def _unpack_obs(self, flat_obs):
        """
        将 SB3 传入的扁平 Tensor (B, 925) 还原为结构化数据
        """
        # 1. Ego State (B, 6)
        ego = flat_obs[:, :self.ego_dim]
        
        # 2. Partners (B, 31, 9)
        start_p = self.ego_dim
        end_p = start_p + (self.max_agents * self.partner_dim)
        partners = flat_obs[:, start_p:end_p].view(-1, self.max_agents, self.partner_dim)
        
        # 3. Map (B, 64, 10)
        road_map = flat_obs[:, end_p:].view(-1, self.max_map_points, self.map_dim)
        
        return ego, partners, road_map

    def forward(self, obs, deterministic=False):
        """
        SB3 标准接口: 用于 Action 选择
        """
        ego, partners, road_map = self._unpack_obs(obs)
        
        # Actor 路径 (只使用 actor_body)
        actor_latent = self.actor_body(ego, partners, road_map)
        distribution = self._get_action_dist_from_latent(actor_latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # Critic 路径 (只使用 critic_body - 完全独立)
        critic_latent = self.critic_body(ego, partners, road_map)
        values = self.value_net(critic_latent)
        
        return actions, values, log_prob

    def predict_values(self, obs):
        """
        仅计算 Value
        """
        ego, partners, road_map = self._unpack_obs(obs)
        critic_latent = self.critic_body(ego, partners, road_map)
        return self.value_net(critic_latent)

    def evaluate_actions(self, obs, actions):
        """
        PPO 训练更新时调用
        """
        ego, partners, road_map = self._unpack_obs(obs)
        
        # 1. Actor 计算 LogProb & Entropy
        actor_latent = self.actor_body(ego, partners, road_map)
        distribution = self._get_action_dist_from_latent(actor_latent)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # 2. Critic 计算 Value
        critic_latent = self.critic_body(ego, partners, road_map)
        values = self.value_net(critic_latent)
        
        return values, log_prob, entropy

    def _get_action_dist_from_latent(self, latent_tensor):
        mean_actions = self.action_net(latent_tensor)
        return self.action_dist.proba_distribution(action_logits=mean_actions)
    
    # 兼容性 Dummy
    def extract_features(self, obs):
        return obs

# =============================================================================
# [修复] 别名映射
# =============================================================================
LateFusionPolicy = LateFusionNet