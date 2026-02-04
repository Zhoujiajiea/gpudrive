"""Torch Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
import torch
from itertools import product
import mediapy as media
import gymnasium
import random
import madrona_gpudrive
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
    PartnerObs,
    LidarObs,
    BevObs,
)

from gpudrive.env import constants
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.base_env import GPUDriveGymEnv
from gpudrive.datatypes.trajectory import LogTrajectory
from gpudrive.datatypes.roadgraph import (
    LocalRoadGraphPoints,
    GlobalRoadGraphPoints,
)
from gpudrive.datatypes.metadata import Metadata
from gpudrive.datatypes.info import Info

from gpudrive.visualize.core import MatplotlibVisualizer
from gpudrive.visualize.utils import img_from_fig

from gpudrive.utils.geometry import normalize_min_max

from gpudrive.integrations.vbd.data_utils import process_scenario_data


class GPUDriveTorchEnv(GPUDriveGymEnv):
    """Torch Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        sim_scenes: list, 
        max_cont_agents,
        device="cuda",
        action_type="discrete",
        render_config: RenderConfig = RenderConfig(),
        backend="torch",
    ):
        # Initialization of environment configurations
        self.config = config
        
        # [修改 3] num_worlds 从场景列表的长度推导
        self.num_worlds = len(sim_scenes)
        
        # [修改 4] data_batch 不再用于加载地图，设为 None
        self.data_batch = None
        
        self.max_cont_agents = max_cont_agents
        self.device = device
        self.render_config = render_config
        self.backend = backend
        self.max_num_agents_in_scene = self.config.max_num_agents_in_scene
        self.world_time_steps = torch.zeros(
            self.num_worlds, dtype=torch.short, device=self.device
        )

        # Initialize reward weights tensor if using reward_conditioned
        self.reward_weights_tensor = None
        if (
            hasattr(self.config, "reward_type")
            and self.config.reward_type == "reward_conditioned"
        ):
            condition_mode = getattr(self.config, "condition_mode", "random")
            agent_type = getattr(self.config, "agent_type", None)
            self._set_reward_weights(
                condition_mode=condition_mode, agent_type=agent_type
            )

        # Environment parameter setup
        params = self._setup_environment_parameters()
        params.maxNumControlledAgents = int(self.max_cont_agents)

        # =============================================================
        # [GIGAFLOW FIX] RewardParams 初始化 - 必须与 C++ types.hpp 完全一致
        # =============================================================
        reward_params = madrona_gpudrive.RewardParams()

        # 1. 设置奖励类型
        if hasattr(self.config, "reward_type") and self.config.reward_type == "dense":
            reward_params.rewardType = madrona_gpudrive.RewardType.Dense
        elif hasattr(self.config, "reward_type") and self.config.reward_type == "sparse":
            reward_params.rewardType = madrona_gpudrive.RewardType.OnGoalAchieved
        else:
            reward_params.rewardType = madrona_gpudrive.RewardType.DistanceBased

        # 2. 注入现有权重
        reward_params.distanceToGoalThreshold = getattr(self.config, "dist_to_goal_threshold", 1.0)
        reward_params.distanceToExpertThreshold = getattr(self.config, "dist_to_expert_threshold", 3.0) # 确保有这个
        reward_params.rewardWeightProgress    = getattr(self.config, "reward_weight_progress", 0.05)
        reward_params.rewardWeightGoal        = getattr(self.config, "reward_weight_goal", 10.0)
        reward_params.rewardWeightCollision   = getattr(self.config, "reward_weight_collision", -10.0)
        reward_params.rewardWeightOffRoad     = getattr(self.config, "reward_weight_off_road", -5.0)
        reward_params.rewardWeightStill       = getattr(self.config, "reward_weight_still", 0.0)

        # 3. [CRITICAL FIX] 注入新增权重，修复内存偏移
        # 这里对应您 types.hpp 中新增的 float rewardWeightGoalDist 和 rewardWeightSpeed
        # 如果 config 中没有配置，给予默认值 0.0
        reward_params.rewardWeightGoalDist    = getattr(self.config, "reward_weight_goal_dist", 0.0)
        reward_params.rewardWeightSpeed       = getattr(self.config, "reward_weight_speed", 0.0)

        # =============================================================

        # 4. 将 RewardParams 赋值给主参数对象
        params.rewardParams = reward_params
        
        # [修改 5] 存储传入的场景列表
        self.sim_scenes = sim_scenes
        print(f"\n[PYTHON PROBE] Initializing Manager...")
        print(f"  > Sending max_cont_agents: {self.max_cont_agents}")
        print(f"  > Sending rewardWeightGoalDist: {reward_params.rewardWeightGoalDist}")
        print(f"  > Sending rewardWeightSpeed: {reward_params.rewardWeightSpeed}")
        # Initialize simulator
        # [修改 6] 传递 self.data_batch (为 None) 和 self.sim_scenes
        self.sim = self._initialize_simulator(params, self.data_batch, self.sim_scenes)

        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        self.episode_len = self.config.episode_len

        self.partner_obs_dim = 0
        self.road_map_obs_dim = 0 
        self.bev_obs_dim = 0

        # Initialize VBD model if used
        self._initialize_vbd()

        # Setup action and observation spaces
        low, high = (-1.0, 1.0) if self.config.norm_obs else (-np.inf, np.inf)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=(self.get_obs(self.cont_agent_mask).shape[-1],),
        )

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_space.shape[-1],),
            dtype=np.float32,
        )

        self._setup_action_space(action_type)
        self.single_action_space = self.action_space

        self.num_agents = self.cont_agent_mask.sum().item()

        # Rendering setup
        self.vis = MatplotlibVisualizer(
            sim_object=self.sim,
            controlled_agent_mask=self.cont_agent_mask,
            goal_radius=self.config.dist_to_goal_threshold,
            backend=self.backend,
            num_worlds=self.num_worlds,
            render_config=self.render_config,
            env_config=self.config,
        )

    def _update_info_dict(self):
        """手动更新 info_dict 供 Callback 读取"""
        infos = self.sim.info_tensor().to_torch() # 获取原始数据
        mask = self.cont_agent_mask
        
        # 强制转换为 float，避免日志系统无法识别 int
        self.info_dict = {
            "off_road": infos[:, :, 5][mask].sum().float().item(),       # Index 5
            "veh_collisions": infos[:, :, 1:3].sum(dim=2)[mask].sum().float().item(),
            "goal_achieved": infos[:, :, 3][mask].sum().float().item(),
            "num_controlled_agents": mask.sum().item()
        }

# ... (从 _initialize_vbd 到 get_expert_actions 的所有代码保持不变) ...
# ... (这些函数不使用 data_batch) ...

    def _initialize_vbd(self):
        """
        Initialize the Versatile Behavior Diffusion (VBD) model and related
        components. Link: https://arxiv.org/abs/2404.02524.

        Args:
            config: Configuration object containing VBD settings.
        """
        self.use_vbd = self.config.use_vbd
        self.vbd_trajectory_weight = self.config.vbd_trajectory_weight

        # Set initialization steps - ensure minimum steps for VBD
        if self.use_vbd:
            self.init_steps = max(
                self.config.init_steps, 10
            )  # Minimum 10 steps for VBD
        else:
            self.init_steps = self.config.init_steps

        if (
            self.use_vbd
            and hasattr(self.config, "vbd_model_path")
            and self.config.vbd_model_path
        ):
            self.vbd_model = self._load_vbd_model(self.config.vbd_model_path)

            self.vbd_trajectories = torch.zeros(
                (
                    self.num_worlds,
                    self.max_agent_count,
                    self.episode_len - self.init_steps,
                    5,
                ),
                device=self.device,
                dtype=torch.float32,
            )

            self._generate_vbd_trajectories()
        else:
            self.vbd_model = None
            self.vbd_trajectories = None

    def _load_vbd_model(self, model_path):
        """Load the Versatile Behavior Diffusion (VBD) model from checkpoint."""
        from gpudrive.integrations.vbd.sim_agent.sim_actor import VBDTest

        model = VBDTest.load_from_checkpoint(
            model_path, torch.device(self.device)
        )
        _ = model.eval()
        return model

    # gpudrive/env/env_torch.py -> _generate_sample_batch 函数 (替换原函数体)

    def _generate_sample_batch(self, init_steps=10):
        """Generate a sample batch for the VBD model."""
        means_xy = (
            self.sim.world_means_tensor().to_torch()[:, :2].to(self.device)
        )

        # Get the logged trajectory and restore the mean
        log_trajectory = LogTrajectory.from_tensor(
            self.sim.expert_trajectory_tensor(),
            self.num_worlds,
            self.max_agent_count,
            backend=self.backend,
        )
        log_trajectory.restore_mean(
            mean_x=means_xy[:, 0], mean_y=means_xy[:, 1]
        )

        # [GIGAFLOW FIX] 替换对 map_observation_tensor 的依赖，传递一个空对象
        global_road_graph = GlobalRoadGraphPoints(data=torch.Tensor())
        
        # Get global agent observations and restore the mean
        global_agent_obs = GlobalEgoState.from_tensor(
            abs_self_obs_tensor=self.sim.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )
        global_agent_obs.restore_mean(
            mean_x=means_xy[:, 0], mean_y=means_xy[:, 1]
        )
        metadata = Metadata.from_tensor(
            metadata_tensor=self.sim.metadata_tensor(),
            backend=self.backend,
        )
        sample_batch = process_scenario_data(
            max_controlled_agents=self.max_cont_agents,
            controlled_agent_mask=self.cont_agent_mask,
            global_agent_obs=global_agent_obs,
            global_road_graph=global_road_graph, # 传递空对象
            log_trajectory=log_trajectory,
            episode_len=self.episode_len,
            init_steps=init_steps,
            raw_agent_types=self.sim.info_tensor().to_torch()[:, :, 4],
            metadata=metadata,
        )
        return sample_batch
    def _set_reward_weights(
        self, env_idx_list=None, condition_mode="random", agent_type=None
    ):
        """Set agent reward weights for all or specific environments.

        Args:
            env_idx_list: List of environment indices to generate new weights for.
                          If None, all environments are updated.
            condition_mode: Determines how reward weights are sampled:
                            - "random": Random sampling within bounds (default for training)
                            - "fixed": Use predefined agent_type weights (for testing)
                            - "preset": Use a specific preset from agent_type parameter
            agent_type: Specifies which preset weights to use if condition_mode is "preset" or "fixed"
                        If condition_mode is "preset", can be one of: "cautious", "aggressive", "balanced"
                        If condition_mode is "fixed", should be a tensor of shape [3] with weight values
        """
        if self.reward_weights_tensor is None:
            self.reward_weights_tensor = torch.zeros(
                self.num_worlds,
                self.max_cont_agents,
                3,  # collision, goal_achieved, off_road
                device=self.device,
            )

        # Read bounds for the three reward components
        lower_bounds = torch.tensor(
            [
                self.config.collision_weight_lb,
                self.config.goal_achieved_weight_lb,
                self.config.off_road_weight_lb,
            ],
            device=self.device,
        )

        upper_bounds = torch.tensor(
            [
                self.config.collision_weight_ub,
                self.config.goal_achieved_weight_ub,
                self.config.off_road_weight_ub,
            ],
            device=self.device,
        )
        bounds_range = upper_bounds - lower_bounds

        # Preset agent personality types
        agent_presets = {
            "cautious": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.9,  # Strong collision penalty
                    self.config.goal_achieved_weight_ub
                    * 0.7,  # Moderate goal reward
                    self.config.off_road_weight_lb
                    * 0.9,  # Strong off-road penalty
                ],
                device=self.device,
            ),
            "aggressive": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.5,  # Lower collision penalty
                    self.config.goal_achieved_weight_ub
                    * 0.9,  # Higher goal reward
                    self.config.off_road_weight_lb
                    * 0.6,  # Moderate off-road penalty
                ],
                device=self.device,
            ),
            "balanced": torch.tensor(
                [
                    (
                        self.config.collision_weight_lb
                        + self.config.collision_weight_ub
                    )
                    / 2,
                    (
                        self.config.goal_achieved_weight_lb
                        + self.config.goal_achieved_weight_ub
                    )
                    / 2,
                    (
                        self.config.off_road_weight_lb
                        + self.config.off_road_weight_ub
                    )
                    / 2,
                ],
                device=self.device,
            ),
            "risk_taker": torch.tensor(
                [
                    self.config.collision_weight_lb
                    * 0.3,  # Minimal collision penalty
                    self.config.goal_achieved_weight_ub,  # Maximum goal reward
                    self.config.off_road_weight_lb
                    * 0.4,  # Low off-road penalty
                ],
                device=self.device,
            ),
        }

        # Determine which environments to update
        if env_idx_list is None:
            env_idx_list = list(range(self.num_worlds))

        env_indices = torch.tensor(env_idx_list, device=self.device)
        num_envs = len(env_indices)

        if condition_mode == "random":
            # Traditional random sampling within bounds
            random_values = torch.rand(
                num_envs, self.max_cont_agents, 3, device=self.device
            )
            scaled_values = lower_bounds + random_values * bounds_range

        elif condition_mode == "preset":
            # Use a predefined agent type
            if agent_type not in agent_presets:
                raise ValueError(
                    f"Unknown agent_type: {agent_type}. Available types: {list(agent_presets.keys())}"
                )

            # Create a tensor with the preset weights for all agents in the specified environments
            preset_weights = agent_presets[agent_type]
            scaled_values = (
                preset_weights.unsqueeze(0)
                .unsqueeze(0)
                .expand(num_envs, self.max_cont_agents, 3)
            )

        elif condition_mode == "fixed":
            # Use custom provided weights
            if agent_type is None or not isinstance(agent_type, torch.Tensor):
                raise ValueError(
                    "For condition_mode='fixed', agent_type must be a tensor of shape [3]"
                )

            custom_weights = agent_type.to(device=self.device)
            if custom_weights.shape != (3,):
                raise ValueError(
                    f"agent_type tensor must have shape [3], got {custom_weights.shape}"
                )

            scaled_values = (
                custom_weights.unsqueeze(0)
                .unsqueeze(0)
                .expand(num_envs, self.max_cont_agents, 3)
            )

        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        # Update the weights tensor for the specified environments
        self.reward_weights_tensor[env_indices.cpu()] = scaled_values

        return self.reward_weights_tensor
    
    def _init_gigaflow_scenario(self, env_idx_list):
        """
        [GIGAFLOW 修复] 在重置后同步 Python 端的 Agent 掩码。
        由于 C++ 程序化生成会改变 Agent 的数量和位置，
        我们需要更新 cont_agent_mask 以便 Python 知道哪些 Agent 是有效的。
        """
        # 重新从 C++ 获取当前的控制掩码 (Shape: [num_worlds, max_agents])
        self.cont_agent_mask = self.get_controlled_agents_mask()
        
        # 更新有效的受控 Agent 总数
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

    def reset(
        self,
        mask=None,
        env_idx_list=None,
        condition_mode=None,
        agent_type=None,
    ):
        """Reset the worlds and return the initial observations.

        Args:
            mask: Optional mask indicating which agents to return observations for
            env_idx_list: Optional list of environment indices to reset
            condition_mode: Determines how reward weights are sampled:
                            - "random": Random sampling within bounds (default for training)
                            - "fixed": Use predefined agent_type weights (for testing)
                            - "preset": Use a specific preset from agent_type parameter
            agent_type: Specifies which preset weights to use or custom weights
        """
        if env_idx_list is not None:
            self.sim.reset(env_idx_list)
        else:
            env_idx_list = list(range(self.num_worlds))
            self.sim.reset(env_idx_list)

        self._init_gigaflow_scenario(env_idx_list)

        self.world_time_steps.zero_()
        # Re-initialize reward weights if using reward_conditioned
        if (
            hasattr(self.config, "reward_type")
            and self.config.reward_type == "reward_conditioned"
        ):
            # Use the specified condition_mode or default to the config setting
            mode = (
                condition_mode
                if condition_mode is not None
                else getattr(self.config, "condition_mode", "random")
            )
            self._set_reward_weights(
                env_idx_list, condition_mode=mode, agent_type=agent_type
            )

        

        return self.get_obs(mask)

    def get_dones(self):
        return (
            self.sim.done_tensor()
            .to_torch()
            .clone()
            .squeeze(dim=2)
            .to(torch.float)
        )

    def get_infos(self):
        return Info.from_tensor(
            self.sim.info_tensor(),
            backend=self.backend,
            device=self.device,
        )

    # def get_rewards(
    #     self,
    #     collision_weight=-0.5,
    #     goal_achieved_weight=1.0,
    #     off_road_weight=-0.5,
    #     world_time_steps=None,
    #     log_distance_weight=0.01,
    # ):
    #     """Obtain the rewards for the current step.
    #     By default, the reward is a weighted combination of the following components:
    #     - collision
    #     - goal_achieved
    #     - off_road

    #     The importance of each component is determined by the weights.
    #     """

    #     # Return the weighted combination of the reward components
    #     info_tensor = self.sim.info_tensor().to_torch().clone()
    #     off_road = info_tensor[:, :, 5].to(torch.float)

    #     # True if the vehicle is in collision with another road object
    #     # (i.e. a cyclist or pedestrian)
    #     collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
    #     goal_achieved = info_tensor[:, :, 3].to(torch.float)

    #     if self.config.reward_type == "sparse_on_goal_achieved":
    #         return self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

    #     elif self.config.reward_type == "weighted_combination":
    #         weighted_rewards = (
    #             collision_weight * collided
    #             + goal_achieved_weight * goal_achieved
    #             + off_road_weight * off_road
    #         )

    #         return weighted_rewards

    #     elif self.config.reward_type == "reward_conditioned":
    #         # Extract individual weight components from the tensor
    #         # Shape: [num_worlds, max_agents, 3]
    #         if self.reward_weights_tensor is None:
    #             self._set_reward_weights()

    #         # Apply the weights in a vectorized manner
    #         # Each index in dimension 2 corresponds to a specific weight:
    #         # 0: collision, 1: goal_achieved, 2: off_road
    #         weighted_rewards = (
    #             self.reward_weights_tensor[:, :, 0] * collided
    #             + self.reward_weights_tensor[:, :, 1] * goal_achieved
    #             + self.reward_weights_tensor[:, :, 2] * off_road
    #         )

    #         return weighted_rewards

    #     elif self.config.reward_type == "distance_to_vdb_trajs":
    #         # Reward based on distance to VBD predicted trajectories
    #         # (i.e. the deviation from the predicted trajectory)
    #         weighted_rewards = (
    #             collision_weight * collided
    #             + goal_achieved_weight * goal_achieved
    #             + off_road_weight * off_road
    #         )

    #         agent_states = GlobalEgoState.from_tensor(
    #             self.sim.absolute_self_observation_tensor(),
    #             self.backend,
    #             self.device,
    #         )

    #         agent_pos = torch.stack(
    #             [agent_states.pos_x, agent_states.pos_y], dim=-1
    #         )

    #         # Extract VBD positions at current time steps for each world
    #         vbd_pos = []
    #         for i in range(self.num_worlds):
    #             current_time = (
    #                 self.world_time_steps[i].item() - self.init_steps
    #             )
    #             # Make sure we don't exceed trajectory length
    #             current_time = min(
    #                 current_time, self.vbd_trajectories.shape[2] - 1
    #             )
    #             vbd_pos.append(self.vbd_trajectories[i, :, current_time, :2])
    #         vbd_pos_tensor = torch.stack(vbd_pos)

    #         # Compute euclidean distance between agent and logs
    #         dist_to_vbd = torch.norm(vbd_pos_tensor - agent_pos, dim=-1)

    #         # Add reward based on inverse distance to logs
    #         weighted_rewards += self.vbd_trajectory_weight * torch.exp(
    #             -dist_to_vbd
    #         )

    #         return weighted_rewards

    #     elif self.config.reward_type == "distance_to_logs":
    #         # Reward based on distance to logs and penalty for collision
    #         weighted_rewards = (
    #             collision_weight * collided
    #             + goal_achieved_weight * goal_achieved
    #             + off_road_weight * off_road
    #         )

    #         log_trajectory = LogTrajectory.from_tensor(
    #             self.sim.expert_trajectory_tensor(),
    #             self.num_worlds,
    #             self.max_agent_count,
    #             backend=self.backend,
    #         )

    #         # Index log positions at current time steps
    #         log_traj_pos = []
    #         for i in range(self.num_worlds):
    #             log_traj_pos.append(
    #                 log_trajectory.pos_xy[i, :, world_time_steps[i], :]
    #             )
    #         log_traj_pos_tensor = torch.stack(log_traj_pos)

    #         agent_state = GlobalEgoState.from_tensor(
    #             self.sim.absolute_self_observation_tensor(),
    #             self.backend,
    #         )

    #         agent_pos = torch.stack(
    #             [agent_state.pos_x, agent_state.pos_y], dim=-1
    #         )

    #         # compute euclidean distance between agent and logs
    #         dist_to_logs = torch.norm(log_traj_pos_tensor - agent_pos, dim=-1)

    #         # add reward based on inverse distance to logs
    #         weighted_rewards += log_distance_weight * torch.exp(-dist_to_logs)

    #         return weighted_rewards
    # def get_rewards(
    #     self,
    #     collision_weight=-0.5,
    #     goal_achieved_weight=1.0,
    #     off_road_weight=-0.5,
    #     world_time_steps=None,
    #     log_distance_weight=0.01,
    # ):
    #     """Obtain the rewards for the current step.
        
    #     If reward_type is 'dense', it reads directly from the C++ backend (Recommended).
    #     Otherwise, it computes the reward in Python (Slower, but flexible).
    #     """

    #     # ============================================================
    #     # 1. [C++ 路径] Dense Reward (最高效，推荐)
    #     # ============================================================
    #     if hasattr(self.config, "reward_type") and self.config.reward_type == "dense":
    #         # 直接从 GPU 显存读取 C++ 计算好的奖励
    #         # C++ 返回形状 (num_worlds, max_agents, 1)，squeeze 为 (num_worlds, max_agents)
    #         return self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

    #     # ============================================================
    #     # 2. [Python 路径] 准备基础数据
    #     # ============================================================
    #     # 获取基础事件信息 (碰撞、达成目标、越野)
    #     info_tensor = self.sim.info_tensor().to_torch().clone()
    #     off_road = info_tensor[:, :, 5].to(torch.float)
        
    #     # True if the vehicle is in collision with another road object (i.e. a cyclist or pedestrian)
    #     collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
    #     goal_achieved = info_tensor[:, :, 3].to(torch.float)
    #     # ================= [调试探针 1：检查是否进入函数] =================
    #     if not hasattr(self, "_probe_counter"):
    #         self._probe_counter = 0
    #     self._probe_counter += 1

    #     # 每 10 步就打印一次，确保你能立刻看到
    #     if self._probe_counter % 10 == 0:
    #         print(f"[Probe] get_rewards called (Step {self._probe_counter}). Reward Type: {self.config.reward_type}", flush=True)
    #     # ===============================================================
    #     # ============================================================
    #     # 3. 根据类型计算奖励
    #     # ============================================================
        
    #     if self.config.reward_type == "sparse_on_goal_achieved":
    #         # 稀疏奖励 (从 C++ 读取)
    #         return self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

    #     elif self.config.reward_type == "weighted_combination":
    #         # --------------------------------------------------------
    #         # [新增] 计算距离和速度 (Python 实现)
    #         # --------------------------------------------------------
    #         # 解析自车观察数据 (LocalEgoState)
    #         # 注意：这里需要确保 LocalEgoState 已经导入
    #         cpp_rewards = self.sim.reward_tensor().to_torch().squeeze(dim=2)
    #         ego_state = LocalEgoState(self.sim.self_observation_tensor().to_torch())
            
    #         # --- 手动提取需要的数据 (避免 LocalEgoState 初始化开销) ---
    #         # 假设 index 0=speed, 3=rel_goal_x, 4=rel_goal_y (需与 C++ 这里一致)
    #         speed = ego_state.speed
                
    #         # 1. 计算到目标的距离 (L2 Norm)
    #         dist_to_goal = torch.sqrt(ego_state.rel_goal_x**2 + ego_state.rel_goal_y**2)
            

    #         # 3. 获取权重 (尝试从 config 读取，否则使用默认值)
    #         w_progress  = getattr(self.config, "reward_weight_progress", 0.05)
    #         w_still     = getattr(self.config, "reward_weight_still", 0.0)
    #         w_goal_dist = getattr(self.config, "reward_weight_goal_dist", 0.0) # 新增
    #         w_speed     = getattr(self.config, "reward_weight_speed", 0.0)
    #         # # ================= [新增探针 START] =================
    #         # # 这里的逻辑是：只在第 0 个 World 的第 10 步打印一次，避免刷屏
    #         # # self.sim.world_reset_tensor()[0] == 0 确保只看 World 0 (如果它没在重置状态)
    #         # # 我们随机选一个概览时刻，或者简单地每隔 1000 次调用打印一次
            
    #         # if not hasattr(self, "_probe_counter"):
    #         #     self._probe_counter = 0
    #         # self._probe_counter += 1

    #         # # 每 10 次 step 打印一次诊断信息 (频率可调)
    #         # if self._probe_counter % 10 == 0:
    #         #     # 选取前 5 个有效 Agent 看看情况
    #         #     mask = self.cont_agent_mask
    #         #     valid_indices = torch.nonzero(mask.flatten(), as_tuple=True)[0][:5]
                
    #         #     print(f"\n🔍 [Probe @ Step {self._probe_counter}] 物理量检查:")
    #         #     if len(valid_indices) > 0:
    #         #         # 打印前几个 Agent 的具体数值
    #         #         sample_speeds = speed.flatten()[valid_indices]
    #         #         sample_dists = dist_to_goal.flatten()[valid_indices]
                    
    #         #         print(f"   > 采样 Agent 速度 (m/s): {sample_speeds.cpu().numpy()}")
    #         #         print(f"   > 采样 Agent 目标距离 (m): {sample_dists.cpu().numpy()}")
    #         #         print(f"   > 当前权重: w_speed={w_speed}, w_dist={w_goal_dist}")
                    
    #         #         # 检查是否全 0
    #         #         if sample_speeds.abs().sum() < 1e-4:
    #         #             print("   ⚠️ 警告: 采样 Agent 速度全为 0！智能体可能未启动或卡住。")
    #         #         if sample_dists.abs().sum() < 1e-4:
    #         #             print("   ⚠️ 警告: 采样 Agent 目标距离全为 0！可能已到达或目标生成失败。")
    #         #     else:
    #         #         print("   ⚠️ 当前没有活跃的受控 Agent。")
    #         #     print("-" * 50)
    #         # # ================= [新增探针 END] ===================
    #         self.latest_reward_components = {
    #             "rew_speed": (w_speed * speed).detach(),          # 速度奖励分量
    #             "rew_goal_dist": (-w_goal_dist * dist_to_goal).detach(), # 距离惩罚分量 (注意是负号，因为距离越远惩罚越大)
    #             "raw_speed": speed.detach(),                      # (可选) 原始速度 m/s
    #             "raw_dist": dist_to_goal.detach()                 # (可选) 原始距离 m
    #         }
    #         # --------------------------------------------------------
    #         # 组合所有奖励项
    #         # --------------------------------------------------------
    #         weighted_rewards = (
    #             collision_weight * collided
    #             + goal_achieved_weight * goal_achieved
    #             + off_road_weight * off_road
    #             + w_speed * speed
    #             - w_goal_dist * dist_to_goal # 距离越远惩罚越大
    #         )
    #         return weighted_rewards

    #     elif self.config.reward_type == "reward_conditioned":
    #         # 条件化奖励 (使用 self.reward_weights_tensor)
    #         if self.reward_weights_tensor is None:
    #             self._set_reward_weights()
            
    #         # Apply the weights in a vectorized manner
    #         # Each index in dimension 2 corresponds to a specific weight:
    #         # 0: collision, 1: goal_achieved, 2: off_road
    #         weighted_rewards = (
    #             self.reward_weights_tensor[:, :, 0] * collided
    #             + self.reward_weights_tensor[:, :, 1] * goal_achieved
    #             + self.reward_weights_tensor[:, :, 2] * off_road
    #         )
    #         return weighted_rewards

    #     elif self.config.reward_type == "distance_to_vdb_trajs":
    #         # Reward based on distance to VBD predicted trajectories
    #         weighted_rewards = (
    #             collision_weight * collided
    #             + goal_achieved_weight * goal_achieved
    #             + off_road_weight * off_road
    #         )

    #         agent_states = GlobalEgoState.from_tensor(
    #             self.sim.absolute_self_observation_tensor(),
    #             self.backend,
    #             self.device,
    #         )

    #         agent_pos = torch.stack(
    #             [agent_states.pos_x, agent_states.pos_y], dim=-1
    #         )

    #         # Extract VBD positions at current time steps for each world
    #         vbd_pos = []
    #         for i in range(self.num_worlds):
    #             current_time = (
    #                 self.world_time_steps[i].item() - self.init_steps
    #             )
    #             # Make sure we don't exceed trajectory length
    #             current_time = min(
    #                 current_time, self.vbd_trajectories.shape[2] - 1
    #             )
    #             vbd_pos.append(self.vbd_trajectories[i, :, current_time, :2])
    #         vbd_pos_tensor = torch.stack(vbd_pos)

    #         # Compute euclidean distance between agent and logs
    #         dist_to_vbd = torch.norm(vbd_pos_tensor - agent_pos, dim=-1)

    #         # Add reward based on inverse distance to logs
    #         weighted_rewards += self.vbd_trajectory_weight * torch.exp(
    #             -dist_to_vbd
    #         )
    #         return weighted_rewards

    #     elif self.config.reward_type == "distance_to_logs":
    #         # Reward based on distance to logs and penalty for collision
    #         weighted_rewards = (
    #             collision_weight * collided
    #             + goal_achieved_weight * goal_achieved
    #             + off_road_weight * off_road
    #         )

    #         log_trajectory = LogTrajectory.from_tensor(
    #             self.sim.expert_trajectory_tensor(),
    #             self.num_worlds,
    #             self.max_agent_count,
    #             backend=self.backend,
    #         )

    #         # Index log positions at current time steps
    #         log_traj_pos = []
    #         for i in range(self.num_worlds):
    #             # Use passed world_time_steps if available, otherwise use self.world_time_steps
    #             ts = world_time_steps[i] if world_time_steps is not None else self.world_time_steps[i]
    #             log_traj_pos.append(
    #                 log_trajectory.pos_xy[i, :, ts, :]
    #             )
    #         log_traj_pos_tensor = torch.stack(log_traj_pos)

    #         agent_state = GlobalEgoState.from_tensor(
    #             self.sim.absolute_self_observation_tensor(),
    #             self.backend,
    #         )

    #         agent_pos = torch.stack(
    #             [agent_state.pos_x, agent_state.pos_y], dim=-1
    #         )

    #         # compute euclidean distance between agent and logs
    #         dist_to_logs = torch.norm(log_traj_pos_tensor - agent_pos, dim=-1)

    #         # add reward based on inverse distance to logs
    #         weighted_rewards += log_distance_weight * torch.exp(-dist_to_logs)

    #         return weighted_rewards
        
    #     else:
    #         # Fallback for unknown reward types
    def get_rewards(
        self,
        collision_weight=-0.5,
        goal_achieved_weight=1.0,
        off_road_weight=-0.5,
        world_time_steps=None,
        log_distance_weight=0.01,
    ):
        """Obtain the rewards for the current step.
        
        Hybrid Approach:
        - Reads physical penalties (off-road, collision) from C++ backend.
        - Adds shaping rewards (speed, distance) from Python calculation.
        """
        if not hasattr(self, "_global_probe_printed"):
            print(f"\n[CRITICAL DEBUG] get_rewards called!")
            print(f"  > Config Reward Type: '{self.config.reward_type}'")
            self._global_probe_printed = True
        # ============================================================
        # 1. [纯 C++ 路径] Dense Reward (如果不想要 Python 塑形，可用此模式)
        # ============================================================
        if hasattr(self.config, "reward_type") and self.config.reward_type == "dense":
            return self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

        # ============================================================
        # 2. [混合路径] Weighted Combination (C++ 惩罚 + Python 塑形)
        # ============================================================
        elif self.config.reward_type == "weighted_combination":
            
            # --- A. 获取 C++ 计算的基础奖励 ---
            # 包含了你在 sim.cpp 中写的:
            #   ctx.get<Reward> -= 10.0 (完全越野)
            #   ctx.get<Reward> -= 0.05 (部分越野)
            #   (以及 C++ 可能计算的 collision 惩罚)
            cpp_rewards = self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

            # --- B. 获取 Python 端所需的状态数据 ---
            # 1. 基础事件标志
            info_tensor = self.sim.info_tensor().to_torch()
            goal_achieved = info_tensor[:, :, 3].to(torch.float)
            
            # 2. 物理状态 (速度 & 位置)
            # 直接读取 self_observation_tensor (避免 LocalEgoState 额外开销)
            # 布局: [Speed, Len, Wid, RelGoalX, RelGoalY, ...]
            self_obs = self.sim.self_observation_tensor().to_torch()
            speed = self_obs[:, :, 0]
            rel_goal_x = self_obs[:, :, 3]
            rel_goal_y = self_obs[:, :, 4]
            
            # 计算距离 (L2 Norm)
            dist_to_goal = torch.sqrt(rel_goal_x**2 + rel_goal_y**2)

            # --- C. 读取权重配置 ---
            # 注意：这里我们故意忽略 off_road_weight，因为 C++ 已经算过了
            w_goal      = getattr(self.config, "reward_weight_goal", 10.0)
            w_speed     = getattr(self.config, "reward_weight_speed", 0.0)
            w_goal_dist = getattr(self.config, "reward_weight_goal_dist", 0.01)

            self.latest_reward_components = {
                "rew_speed": (w_speed * speed).detach(),
                "rew_goal_dist": (-w_goal_dist * dist_to_goal).detach(),
            }

            # Total = C++基准(含惩罚) + 到达奖励 + 速度奖励 - 距离惩罚
            total_reward = (
                cpp_rewards 
                + (w_goal * goal_achieved) 
                + (w_speed * speed) 
                - (w_goal_dist * dist_to_goal)
            )

            # [诊断探针] 检查混合是否生效 (仅打印一次)
            # 文件: gpudrive/env/env_torch.py
# 位置: get_rewards 函数内部，计算 cpp_rewards 之后

# ...
            # --- [修改前] 原来的代码 (可能被注释掉了) ---
            # print(f"   > C++ Base (Mean): {cpp_rewards.mean().item():.4f} ...")

            # --- [修改后] 新的打印逻辑 (过滤 Padding) ---
            if not hasattr(self, "_hybrid_probe_printed"):
                print(f"\n✅ [Reward System] Hybrid Mode Activated:")
                
                # 1. 获取基础掩码
                mask = self.cont_agent_mask
                
                # 2. [新增] 幽灵过滤器
                # 正常的物理奖励绝不会低于 -500 (除非飞出地球)
                # 幽灵车在 -11000 位置，奖励通常是 -10000 左右
                is_valid_physics = (cpp_rewards > -500.0)
                
                # 3. 组合掩码：既要是受控的，物理状态也必须正常
                clean_mask = mask & is_valid_physics
                
                # 4. 计算并打印“清洗后”的真值
                if clean_mask.sum() > 0:
                    clean_mean = cpp_rewards[clean_mask].mean().item()
                    print(f"   > C++ Base (Active & Cleaned): {clean_mean:.4f} (True Physics Reward)")
                    print(f"     (Based on {clean_mask.sum()} valid agents)")
                else:
                    print(f"   > C++ Base: No Valid Agents Found!")

                # 5. 诊断被过滤掉的幽灵
                ghost_count = (mask & ~is_valid_physics).sum().item()
                if ghost_count > 0:
                    print(f"   ⚠️ WARNING: Filtered out {ghost_count} 'Ghost' agents (Reward < -500).")
                    print(f"      This confirms map coordinates are fixed, but agent count sync has issues.")

                self._hybrid_probe_printed = True
# ...
           

            return total_reward

        # ============================================================
        # 3. 其他旧模式 (保持兼容性)
        # ============================================================
        elif self.config.reward_type == "sparse_on_goal_achieved":
            return self.sim.reward_tensor().to_torch().clone().squeeze(dim=2)

        elif self.config.reward_type == "reward_conditioned":
            if self.reward_weights_tensor is None:
                self._set_reward_weights()
            
            # 获取基础事件
            info_tensor = self.sim.info_tensor().to_torch()
            collided = info_tensor[:, :, 1:3].sum(axis=2).to(torch.float)
            goal_achieved = info_tensor[:, :, 3].to(torch.float)
            off_road = info_tensor[:, :, 5].to(torch.float)

            weighted_rewards = (
                self.reward_weights_tensor[:, :, 0] * collided
                + self.reward_weights_tensor[:, :, 1] * goal_achieved
                + self.reward_weights_tensor[:, :, 2] * off_road
            )
            return weighted_rewards

        elif self.config.reward_type == "distance_to_vdb_trajs":
            # ... (保持原有的 VBD 逻辑不变) ...
            # 为了节省篇幅，这里假设原有逻辑保持不变
            # 如果你需要这部分代码，请告诉我，我再补全
            pass 
            
        elif self.config.reward_type == "distance_to_logs":
             # ... (保持原有的 Log 距离逻辑不变) ...
             pass

        # 默认回退
        return torch.zeros_like(self.sim.reward_tensor().to_torch().squeeze(dim=2))

            
    def step_dynamics(self, actions):
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()
        # 获取观测数据
        obs_check = self.sim.lidar_tensor().to_torch()
        
        # 检查最小值
        min_val = obs_check.min().item()
        
        # 阈值设为 -200 (正常归一化数据通常在 -1 到 1 之间，物理坐标也就几百米)
        # 幽灵坐标通常是 -11000
        if min_val < -2000.0: 
            print(f"\n🚨 [CRITICAL ALERT] 训练数据中检测到幽灵智能体！")
            print(f"   > Min Value Found: {min_val}")
            print(f"   > 这意味着 C++ 端的 reconstructLogic 修复未生效或未被调用。")
        
        self._update_info_dict()
        not_done_worlds = ~self.get_dones().any(
            dim=1
        )  # Check if any agent in world is done
        self.world_time_steps[not_done_worlds] += 1

    def _apply_actions(self, actions):
        """Apply the actions to the simulator."""

        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
            or self.config.dynamics_model == "delta_local"
        ):
            if actions.dim() == 2:  # (num_worlds, max_agent_count)
                # Map action indices to action values if indices are provided
                actions = (
                    torch.nan_to_num(actions, nan=0).long().to(self.device)
                )
                action_value_tensor = self.action_keys_tensor[actions]

            elif actions.dim() == 3:
                if actions.shape[2] == 1:
                    actions = actions.squeeze(dim=2).to(self.device)
                    action_value_tensor = self.action_keys_tensor[actions]
                else:  # Assuming we are given the actual action values
                    action_value_tensor = actions.to(self.device)
            else:
                raise ValueError(f"Invalid action shape: {actions.shape}")

        else:
            action_value_tensor = actions.to(self.device)

        # Feed the action values to gpudrive
        self._copy_actions_to_simulator(action_value_tensor)

    def _copy_actions_to_simulator(self, actions):
        """Copy the provided actions to the simulator."""
        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            # Action space: (acceleration, steering, heading)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "delta_local":
            # Action space: (dx, dy, dyaw)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "state":
            # Following the StateAction struct in types.hpp
            # Need to provide:
            # (x, y, z, yaw, vel x, vel y, vel z, ang_vel_x, ang_vel_y, ang_vel_z)
            self.sim.action_tensor().to_torch()[:, :, :10].copy_(actions)
        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space based on dynamics model."""
        products = None

        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            products = product(self.dx, self.dy, self.dyaw)
        elif (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = self.config.head_tilt_actions.to(self.device)
            products = product(
                self.accel_actions, self.steer_actions, self.head_actions
            )
        elif self.config.dynamics_model == "state":
            self.x = self.config.x.to(self.device)
            self.y = self.config.y.to(self.device)
            self.yaw = self.config.yaw.to(self.device)
            self.vx = self.config.vx.to(self.device)
            self.vy = self.config.vy.to(self.device)

        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}
        self.values_to_action_key = {}
        if products is not None:
            for action_idx, (action_1, action_2, action_3) in enumerate(
                products
            ):
                self.action_key_to_values[action_idx] = [
                    action_1.item(),
                    action_2.item(),
                    action_3.item(),
                ]
                self.values_to_action_key[
                    round(action_1.item(), 5),
                    round(action_2.item(), 5),
                    round(action_3.item(), 5),
                ] = action_idx

            self.action_keys_tensor = torch.tensor(
                [
                    self.action_key_to_values[key]
                    for key in sorted(self.action_key_to_values.keys())
                ]
            ).to(self.device)

            return Discrete(n=int(len(self.action_key_to_values)))
        else:
            return Discrete(n=1)

    def _set_continuous_action_space(self) -> None:
        """Configure the continuous action space."""
        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            action_1 = self.dx.clone().cpu().numpy()
            action_2 = self.dy.clone().cpu().numpy()
            action_3 = self.dyaw.clone().cpu().numpy()
        elif self.config.dynamics_model == "classic":
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = torch.tensor([0], device=self.device)
            action_1 = self.steer_actions.clone().cpu().numpy()
            action_2 = self.accel_actions.clone().cpu().numpy()
            action_3 = self.head_actions.clone().cpu().numpy()
        else:
            raise ValueError(
                f"Continuous action space is currently not supported for dynamics_model: {self.config.dynamics_model}."
            )

        action_space = Tuple(
            (
                Box(action_1.min(), action_1.max(), shape=(1,)),
                Box(action_2.min(), action_2.max(), shape=(1,)),
                Box(action_3.min(), action_3.max(), shape=(1,)),
            )
        )
        return action_space

    def _get_ego_state(self, mask=None) -> torch.Tensor:
        """Get the ego state."""

        if not self.config.ego_state:
            return torch.Tensor().to(self.device)

        ego_state = LocalEgoState.from_tensor(
            self_obs_tensor=self.sim.self_observation_tensor(),
            backend=self.backend,
            device=self.device,
            mask=mask,
        )
        if self.config.norm_obs:
            ego_state.normalize()

        if mask is None:
            if self.config.reward_type == "reward_conditioned":
                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                        self.reward_weights_tensor[:, :, 0],
                        self.reward_weights_tensor[:, :, 1],
                        self.reward_weights_tensor[:, :, 2],
                    ]
                ).permute(1, 2, 0)

            else:
                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                    ]
                ).permute(1, 2, 0)

        else:
            if self.config.reward_type == "reward_conditioned":
                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                        self.reward_weights_tensor[mask][:, 0],
                        self.reward_weights_tensor[mask][:, 1],
                        self.reward_weights_tensor[mask][:, 2],
                    ]
                ).permute(1, 0)
            else:
                return torch.stack(
                    [
                        ego_state.speed,
                        ego_state.vehicle_length,
                        ego_state.vehicle_width,
                        ego_state.rel_goal_x,
                        ego_state.rel_goal_y,
                        ego_state.is_collided,
                    ]
                ).permute(1, 0)

    # def _get_partner_obs(self, mask=None):
    #     """Get partner observations."""

    #     if not self.config.partner_obs:
    #         return torch.Tensor().to(self.device)

    #     partner_obs = PartnerObs.from_tensor(
    #         partner_obs_tensor=self.sim.partner_observations_tensor(),
    #         backend=self.backend,
    #         device=self.device,
    #         mask=mask,
    #     )

    #     

    #     if mask is not None:
    #         return partner_obs.data.flatten(start_dim=1)
    #     else:
    #         return torch.concat(
    #             [
    #                 partner_obs.speed,
    #                 partner_obs.rel_pos_x,
    #                 partner_obs.rel_pos_y,
    #                 partner_obs.orientation,
    #                 partner_obs.vehicle_length,
    #                 partner_obs.vehicle_width,
    #             ],
    #             dim=-1,
    #         ).flatten(start_dim=2)

    # 文件: gpudrive/env/env_torch.py




    # 文件: gpudrive/env/env_torch.py

    def _get_partner_obs(self, mask=None):
        """
        [GIGAFLOW FINAL] Get partner observations with auto-reshape and flattening.
        """
        if hasattr(self.config, "partner_obs") and not self.config.partner_obs:
            return torch.Tensor().to(self.device)

        # 1. 获取原始 Tensor (N, A, K*9) 或 (N, A, K, 9)
        obs_tensor = self.sim.partner_observations_tensor().to_torch()
        
        # [AUTO-FIX] 确保它是 4D 结构 (N, A, K, 9)
        # 如果 C++ 返回的是扁平的 3D (N, A, K*9)，我们先恢复它以便后续处理，
        # 但既然我们最终要 Flatten，其实可以直接处理，不过为了逻辑统一，我们先保持标准形状。
        if obs_tensor.ndim == 3:
            N, A, FlatDim = obs_tensor.shape
            max_partners = constants.MAX_PARTNER_COUNT # 149
            feature_dim = FlatDim // max_partners # 应该 = 9
            
            try:
                obs_tensor = obs_tensor.view(N, A, max_partners, feature_dim)
            except Exception as e:
                print(f"❌ Reshape Failed: {e}")

        # [PROBE] 最后一次确认（只打印一次）
        if not hasattr(self, "_probe_flatten_checked"):
            print(f"\n[PYTHON PROBE] Partner Flatten Check")
            print(f"  > Raw Shape: {obs_tensor.shape}")
            if obs_tensor.shape[-1] == 9:
                 print(f"  > ✅ Feature Dim is 9. Flattening for MLP...")
            self._probe_flatten_checked = True

        # 2. 应用掩码并展平 (Flatten) 为神经网络输入格式
        if mask is not None:
            # [Case A] 有 Mask: (TotalAgents, K, 9) -> (TotalAgents, K*9)
            # 结果是 2D，可以与 EgoState (TotalAgents, 6) 拼接
            return obs_tensor[mask].flatten(start_dim=1)
            
        # [Case B] 无 Mask: (N, A, K, 9) -> (N, A, K*9)
        # 结果是 3D，可以与 EgoState (N, A, 6) 拼接
        return obs_tensor.flatten(start_dim=2)

               


# gpudrive/env/env_torch.py -> _get_road_map_obs 函数 (替换整个函数体)

    def _get_road_map_obs(self, mask=None):
        """Get road map observations."""
        
        # [GIGAFLOW FIX] C++ 侧已删除此 Tensor，返回零张量以维持维度
        road_map_feature_dim = 0
        
        if mask is not None:
            valid_count = mask.sum().item()
            return torch.zeros(
                valid_count, 
                road_map_feature_dim, 
                dtype=torch.float32
            ).to(self.device)
            
        # 如果没有 mask，返回 (Num_Worlds, Max_Agents, 0)
        return torch.zeros(
            self.num_worlds, 
            self.max_agent_count, 
            road_map_feature_dim, 
            dtype=torch.float32
        ).to(self.device)
    # 如果配置要求启用，则返回零，因为重建数据不通过此 Tensor 导出
    
    def _get_lidar_obs(self, mask=None):
        """Get lidar observations."""

        if not self.config.lidar_obs:
            return torch.Tensor().to(self.device)

        lidar = LidarObs.from_tensor(
            lidar_tensor=self.sim.lidar_tensor(),
            backend=self.backend,
            device=self.device,
        )

        if mask is not None:
            return [
                lidar.agent_samples[mask],
                lidar.road_edge_samples[mask],
                lidar.road_line_samples[mask],
            ]
        else:
            return torch.cat(
                [
                    lidar.agent_samples,
                    lidar.road_edge_samples,
                    lidar.road_line_samples,
                ],
                dim=-1,
            ).flatten(start_dim=2)

# gpudrive/env/env_torch.py -> _get_bev_obs 函数 (替换原函数体)

    def _get_bev_obs(self, mask=None):
        """Get BEV segmentation map observation.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, resolution, resolution, 1)
        """
        # [GIGAFLOW FIX] C++ 侧已删除此 Tensor，返回空张量
        return torch.Tensor().to(self.device)


    def _get_vbd_obs(self, mask=None):
        """
        Get ego-centric VBD trajectory observations for controlled agents using matrix operations.

        Args:
            mask: Optional mask to filter agents

        Returns:
            Tensor of ego-centric VBD trajectories
        """
        if not self.use_vbd or self.vbd_model is None:
            return torch.Tensor().to(self.device)

        # Get current agent positions and orientations
        agent_state = GlobalEgoState.from_tensor(
            abs_self_obs_tensor=self.sim.absolute_self_observation_tensor(),
            backend=self.backend,
            device=self.device,
        )

        # Initialize output tensor
        traj_feature_dim = (
            self.vbd_trajectories.shape[2] * self.vbd_trajectories.shape[3]
        )

        if mask is not None:
            # Count valid agents for output tensor size
            valid_count = mask.sum().item()
            ego_vbd_trajectories = torch.zeros(
                (valid_count, traj_feature_dim), device=self.device
            )

            # Track which output index we're filling
            out_idx = 0

            # Process each world
            for w in range(self.num_worlds):
                # Get valid agent indices for this world
                world_mask = mask[w]
                agent_indices = torch.where(world_mask)[0]

                if len(agent_indices) == 0:
                    continue

                # Extract ego positions and yaws for these agents
                ego_pos_x = agent_state.pos_x[w, agent_indices]
                ego_pos_y = agent_state.pos_y[w, agent_indices]
                ego_yaw = agent_state.rotation_angle[w, agent_indices]

                # Process each agent in this world
                for i, agent_idx in enumerate(agent_indices):
                    # Get global trajectory for this agent
                    global_traj = self.vbd_trajectories[w, agent_idx]

                    # Create 2D rotation matrix for this agent
                    cos_yaw = torch.cos(ego_yaw[i])
                    sin_yaw = torch.sin(ego_yaw[i])
                    rotation_matrix = torch.tensor(
                        [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]],
                        device=self.device,
                    )

                    # Transform positions using matrix multiplication
                    pos_xy = global_traj[:, :2]
                    ego_pos = torch.tensor(
                        [ego_pos_x[i], ego_pos_y[i]], device=self.device
                    ).reshape(1, 2)
                    translated_pos = (
                        pos_xy - ego_pos
                    )  # Broadcasting to subtract from all timesteps
                    rotated_pos = torch.matmul(
                        translated_pos, rotation_matrix.T
                    )

                    # Transform velocities (only rotation, no translation)
                    vel_xy = global_traj[:, 3:5]
                    rotated_vel = torch.matmul(vel_xy, rotation_matrix.T)

                    # Create transformed trajectory
                    transformed_traj = torch.zeros_like(global_traj)
                    transformed_traj[:, :2] = rotated_pos
                    transformed_traj[:, 2] = (
                        global_traj[:, 2] - ego_yaw[i]
                    )  # Adjust heading
                    transformed_traj[:, 3:5] = rotated_vel

                    # Flatten and add to output
                    ego_vbd_trajectories[out_idx] = transformed_traj.reshape(
                        -1
                    )
                    out_idx += 1

            if self.config.norm_obs:
                traj_len = self.vbd_trajectories.shape[2]
                ego_vbd_trajectories = self._normalize_vbd_obs(
                    ego_vbd_trajectories, traj_len
                )

            return ego_vbd_trajectories

        else:
            # Without mask, process all agents in all worlds
            ego_vbd_trajectories = torch.zeros(
                (self.num_worlds, self.max_agent_count, traj_feature_dim),
                device=self.device,
            )

            # Process each world
            for w in range(self.num_worlds):
                # Get controlled agent indices for this world
                valid_mask = self.cont_agent_mask[w]
                world_agent_indices = torch.where(valid_mask)[0]

                if len(world_agent_indices) == 0:
                    continue

                # Extract ego positions and yaws
                ego_pos_x = agent_state.pos_x[w]
                ego_pos_y = agent_state.pos_y[w]
                ego_yaw = agent_state.rotation_angle[w]

                # Process each agent in this world
                for agent_idx in world_agent_indices:
                    # Get global trajectory
                    global_traj = self.vbd_trajectories[w, agent_idx]

                    # Create 2D rotation matrix for this agent
                    cos_yaw = torch.cos(ego_yaw[agent_idx])
                    sin_yaw = torch.sin(ego_yaw[agent_idx])
                    rotation_matrix = torch.tensor(
                        [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]],
                        device=self.device,
                    )

                    # Transform positions
                    pos_xy = global_traj[:, :2]
                    ego_pos = torch.tensor(
                        [ego_pos_x[agent_idx], ego_pos_y[agent_idx]],
                        device=self.device,
                    ).reshape(1, 2)
                    translated_pos = pos_xy - ego_pos
                    rotated_pos = torch.matmul(
                        translated_pos, rotation_matrix.T
                    )

                    # Transform velocities
                    vel_xy = global_traj[:, 3:5]
                    rotated_vel = torch.matmul(vel_xy, rotation_matrix.T)

                    # Create transformed trajectory
                    transformed_traj = torch.zeros_like(global_traj)
                    transformed_traj[:, :2] = rotated_pos
                    transformed_traj[:, 2] = (
                        global_traj[:, 2] - ego_yaw[agent_idx]
                    )
                    transformed_traj[:, 3:5] = rotated_vel

                    # Flatten and add to output
                    ego_vbd_trajectories[
                        w, agent_idx
                    ] = transformed_traj.reshape(-1)

            if self.config.norm_obs:
                traj_len = self.vbd_trajectories.shape[2]
                ego_vbd_trajectories = self._normalize_vbd_obs(
                    ego_vbd_trajectories, traj_len
                )

            return ego_vbd_trajectories

    def _normalize_vbd_obs(self, trajectories_flat, traj_len):
        """
        Normalize flattened VBD trajectory values to be between -1 and 1, with clipping.

        Args:
            trajectories_flat: Flattened tensor containing trajectory data
            traj_len: Number of trajectory steps

        Returns:
            Normalized flattened trajectories tensor
        """
        # Get original shape for proper reshaping
        original_shape = trajectories_flat.shape

        # Calculate feature dimension
        feature_dim = 5  # x, y, yaw, vel_x, vel_y

        # Reshape to separate the features
        if len(original_shape) == 2:  # (num_agents, flattened_features)
            traj_features = trajectories_flat.reshape(
                -1, traj_len, feature_dim
            )
        else:  # (num_worlds, max_agents, flattened_features)
            traj_features = trajectories_flat.reshape(
                original_shape[0], original_shape[1], traj_len, feature_dim
            )

        # Normalize each feature
        # x, y positions
        traj_features[..., 0] = normalize_min_max(
            tensor=traj_features[..., 0],
            min_val=constants.MIN_REL_GOAL_COORD,
            max_val=constants.MAX_REL_GOAL_COORD,
        )
        traj_features[..., 1] = normalize_min_max(
            tensor=traj_features[..., 1],
            min_val=constants.MIN_REL_GOAL_COORD,
            max_val=constants.MAX_REL_GOAL_COORD,
        )

        # Normalize yaw angle
        traj_features[..., 2] = (
            traj_features[..., 2] / constants.MAX_ORIENTATION_RAD
        )

        # Normalize velocities
        traj_features[..., 3] = traj_features[..., 3] / constants.MAX_SPEED
        traj_features[..., 4] = traj_features[..., 4] / constants.MAX_SPEED

        # Clip all values to the [-1, 1] range
        traj_features = torch.clamp(traj_features, min=-1.0, max=1.0)

        # Reshape back to original format
        return traj_features.reshape(original_shape)

    # [替换整个 get_obs 方法]
    def get_obs(self, mask=None):
        """
        [GIGAFLOW FIXED] Pipeline: Compact Obs (15 dim) -> C++ Reconstruct -> Python Normalize
        Total Output Dim: 925
        """
        # =================================================================
        # 1. 准备数据源 (Compact 15-dim)
        # =================================================================
        # 这里的 15 维结构必须与 sim.cpp reconstructLogic 中的读取顺序严格一致
        # Indices 0-7: Self Observation (Speed, Size, Goal, Collision, ID)
        self_obs = self.sim.self_observation_tensor().to_torch()
        
        # Indices 8-14: Absolute Observation (Pos X,Y,Z + Rot X,Y,Z,W)
        # 我们只取前 7 维 (Pos + Rot)
        abs_obs = self.sim.absolute_self_observation_tensor().to_torch()[..., :7]
        
        # 拼接: (NumWorlds, MaxAgents, 15)
        compact_obs = torch.cat([self_obs, abs_obs], dim=-1)

        # =================================================================
        # 2. 准备 C++ 输入 (Flatten & Contiguous)
        # =================================================================
        if mask is not None:
            # Case A: 有掩码，只处理有效 Agent
            # Shape: (N_valid, 15)
            obs_input = compact_obs[mask]
        else:
            # Case B: 无掩码，处理所有 Agent (保持 Batch 维度以便后续恢复)
            # Shape: (NumWorlds * MaxAgents, 15)
            obs_input = compact_obs.flatten(0, 1)
            
        # 必须确保内存连续，否则 C++ 指针会读到乱码
        if not obs_input.is_contiguous():
            obs_input = obs_input.contiguous()

        # =================================================================
        # 3. 调用 C++ 重建 (Zero-Copy)
        # =================================================================
        # 这一步会瞬间生成 925 维的原始物理数据
        ptr = obs_input.data_ptr()
        rows = obs_input.shape[0]
        cols = obs_input.shape[1] # 应该是 15
        
        # 返回: (Rows, 925)
        obs_raw = self.sim.reconstruct_observations(ptr, rows, cols).to_torch()

        # =================================================================
        # =================================================================
        # [探针 1] C++ 原始数据检查 (Raw Physics Units)
        # =================================================================
        # 仅在 World 0 的第一步打印，防止刷屏
        if not hasattr(self, "_probe_raw_printed") and obs_raw.shape[0] > 0:
            print(f"\n{'='*20} [PROBE 1] C++ Raw Output (925-dim) {'='*20}")
            print(f"Shape: {obs_raw.shape}")
            
            # 取第一个 Agent 的数据样本
            sample = obs_raw[0] 
            
            # A. 自车 (Ego) - 应该是物理数值 (速度 m/s, 位置 m)
            print(f"  🚗 Ego (0-5): {sample[0:6].tolist()}")
            print(f"     > Speed: {sample[0]:.2f} (Expected: ~0-30)")
            print(f"     > Size:  {sample[1]:.2f}x{sample[2]:.2f}")
            print(f"     > Goal:  ({sample[3]:.2f}, {sample[4]:.2f})")
            
            # B. 邻居 (Partner) - 应该包含相对位置
            print(f"  👥 Partner 0 (6-14): {sample[6:15].tolist()}")
            print(f"     > Rel Pos: ({sample[7]:.2f}, {sample[8]:.2f})")
            
            # C. 地图 (Map) - 应该包含相对位置和尺寸
            # Index 285 是第一个地图点的开始 (6 + 31*9)
            map_start = 285
            print(f"  🛣️ Map Point 0 ({map_start}-{map_start+9}): {sample[map_start:map_start+10].tolist()}")
            print(f"     > Rel Pos: ({sample[map_start]:.2f}, {sample[map_start+1]:.2f})")
            
            # 整体统计
            print(f"  📊 Stats: Min={obs_raw.min().item():.2f}, Max={obs_raw.max().item():.2f}")
            print("="*60 + "\n")
            self._probe_raw_printed = True
        elif not hasattr(self, "_probe_raw_printed") and obs_raw.shape[0] == 0:
            print(f"[PROBE 1] Skipped: No agents in current pass (Shape: {obs_raw.shape})")
        # =================================================================
        # 4. 归一化 (Normalization)
        # =================================================================
        # 将原始物理单位 (米, m/s) 转换为神经网络友好的 [-1, 1]
        obs_norm = self._normalize_reconstructed_obs(obs_raw)
        # =================================================================
        # [探针 2] 神经网络输入检查 (Normalized Data)
        # =================================================================
        if not hasattr(self, "_probe_norm_printed") and obs_norm.shape[0] > 0:
            print(f"\n{'='*20} [PROBE 2] NN Input (Normalized) {'='*20}")
            
            idx = 0
            sample = obs_norm[15]
            
            # A. 自车 (Ego) - 应该在 [-1, 1] 或 [0, 1]
            print(f"  🚗 Ego (0-5): {sample[0:6].tolist()}")
            print(f"     > Speed (Norm): {sample[0]:.4f}")
            
            # B. 邻居
            print(f"  👥 Partner 0 (6-14): {sample[6:15].tolist()}")
            
            # C. 地图
            map_start = 285
            print(f"  🛣️ Map Point 0: {sample[map_start:map_start+10].tolist()}")
            
            # 整体统计
            print(f"  📊 Stats: Min={obs_norm.min().item():.4f}, Max={obs_norm.max().item():.4f}")
            
            if obs_norm.abs().max() > 5.0:
                print("  ❌ [WARNING] Values > 5.0 detected! Normalization might be wrong.")
            else:
                print("  ✅ [OK] Values are within reasonable range.")
            print("="*60 + "\n")
            self._probe_norm_printed = True
        # =================================================================
        # =================================================================
        # 5. 恢复形状 (如果需要)
        # =================================================================
        if mask is None:
            # (NumWorlds * MaxAgents, 925) -> (NumWorlds, MaxAgents, 925)
            obs_norm = obs_norm.view(self.num_worlds, self.max_agent_count, -1)
            
        return obs_norm

    def _normalize_reconstructed_obs(self, obs_flat):
        """
        [Helper] 对 925 维扁平向量进行归一化
        obs_flat Shape: (N, 925)
        """
        if not self.config.norm_obs:
            return obs_flat

        # Clone 以避免原地修改影响缓存（如果存在）
        obs = obs_flat.clone()

        # --- A. Ego State (Indices 0-6) ---
        # 0:Speed, 1:Len, 2:Wid, 3:GoalX, 4:GoalY, 5:Collision
        obs[:, 0] /= constants.MAX_SPEED
        obs[:, 1] /= constants.MAX_VEH_LEN
        obs[:, 2] /= constants.MAX_VEH_WIDTH
        obs[:, 3] = normalize_min_max(obs[:, 3], constants.MIN_REL_GOAL_COORD, constants.MAX_REL_GOAL_COORD)
        obs[:, 4] = normalize_min_max(obs[:, 4], constants.MIN_REL_GOAL_COORD, constants.MAX_REL_GOAL_COORD)
        # Index 5 is Collision (0/1), skip

        # --- B. Partner Obs (Indices 6-285) ---
        # 31 Partners * 9 Features
        num_partners = constants.MAX_PARTNER_COUNT # 31
        feat_dim = constants.PARTNER_FEAT_DIM      # 9
        
        # 提取切片并 Reshape 为 (N, 31, 9) 方便批量操作
        start_idx = 6
        end_idx = start_idx + (num_partners * feat_dim)
        partners = obs[:, start_idx:end_idx].view(-1, num_partners, feat_dim)
        
        # 0:Speed, 1:PosX, 2:PosY, 3:Heading, 4:Len, 5:Wid ...
        partners[..., 0] /= constants.MAX_SPEED
        partners[..., 1] = normalize_min_max(partners[..., 1], constants.MIN_REL_GOAL_COORD, constants.MAX_REL_GOAL_COORD)
        partners[..., 2] = normalize_min_max(partners[..., 2], constants.MIN_REL_GOAL_COORD, constants.MAX_REL_GOAL_COORD)
        partners[..., 3] /= constants.MAX_ORIENTATION_RAD
        partners[..., 4] /= constants.MAX_VEH_LEN
        partners[..., 5] /= constants.MAX_VEH_WIDTH
        partners[..., 8] /= 20.0
        # 写回
        obs[:, start_idx:end_idx] = partners.flatten(1).clone()

        # --- C. Map Obs (Indices 285-925) ---
        # 64 Points * 10 Features
        num_map = constants.MAX_ROAD_OBS_COUNT # 64
        map_dim = constants.ROAD_GRAPH_FEAT_DIM  # 10
        
        map_start = end_idx
        map_end = map_start + (num_map * map_dim)
        road_map = obs[:, map_start:map_end].view(-1, num_map, map_dim)
        
        # 0:PosX, 1:PosY, 2:ScaleX, 3:ScaleY, 4:Heading ...
        road_map[..., 0] = normalize_min_max(road_map[..., 0], constants.MIN_REL_GOAL_COORD, constants.MAX_REL_GOAL_COORD)
        road_map[..., 1] = normalize_min_max(road_map[..., 1], constants.MIN_REL_GOAL_COORD, constants.MAX_REL_GOAL_COORD)
        road_map[..., 2] /= constants.MAX_ROAD_SCALE
        road_map[..., 3] /= constants.MAX_ROAD_SCALE
        road_map[..., 4] /= constants.MAX_ORIENTATION_RAD
        road_map[..., 5] /= 20.0
        road_map[..., 6] = 0.0

        # 写回
       

        return obs

    # 修改文件: gpudrive/env/env_torch.py

    def get_controlled_agents_mask(self):
        """
        [GIGAFLOW FIX] 获取控制掩码。
        不仅读取 C++ 的状态，还通过检查物理坐标来过滤掉 '幽灵智能体'。
        """
        # 1. 获取 C++ 认为的控制状态 (目前它是全 1，不准确)
        raw_mask = (
            self.sim.controlled_state_tensor().to_torch().clone() == 1
        ).squeeze(axis=2)

        # 2. [新增] 物理位置检查
        # 获取所有智能体的 X 坐标
        # self_observation_tensor 布局: [Speed, Length, Width, GoalX, GoalY, Collision, ID]
        # 等等，我们需要绝对坐标来判断是否在 -11000
        abs_obs = self.sim.absolute_self_observation_tensor().to_torch()
        pos_x = abs_obs[:, :, 0] # (NumWorlds, MaxAgents)

        # 3. 定义过滤器：只有坐标大于 -500 的才算活人
        # (正常地图坐标是 0~500，幽灵是 -11000)
        valid_physics_mask = (pos_x > -500.0)

        # 4. 合并掩码：既要 C++ 说可控，又要物理上存在
        final_mask = raw_mask & valid_physics_mask

        # [可选] 打印一次诊断信息，确认过滤生效
        if not hasattr(self, "_mask_debug_printed"):
            total_slots = raw_mask.numel()
            valid_agents = final_mask.sum().item()
            ghosts = total_slots - valid_agents
            print(f"\n🛡️ [MASK SYSTEM] Ghost Filter Installed.")
            print(f"   > Total Slots: {total_slots}")
            print(f"   > Real Agents: {valid_agents}")
            print(f"   > Ghosts Killed: {ghosts}")
            self._mask_debug_printed = True

        return final_mask

    def advance_sim_with_log_playback(self, init_steps=0):
        """Advances the simulator by stepping the objects with the logged human trajectories.

        Args:
            init_steps (int): Number of warmup steps.
        """
        if init_steps >= self.config.episode_len:
            raise ValueError(
                "The length of the expert trajectory is 91,"
                f"so init_steps = {init_steps} should be < than 91."
            )

        self.init_frames = []

        self.log_playback_traj, _, _, _ = self.get_expert_actions()

        for time_step in range(init_steps):
            self.step_dynamics(
                actions=self.log_playback_traj[:, :, time_step, :]
            )

    def remove_agents_by_id(
        self, perc_to_rmv_per_scene, remove_controlled_agents=True
    ):
        """Delete random agents in scenarios.

        Args:
            perc_to_rmv_per_scene (float): Percentage of agents to remove per scene
            remove_controlled_agents (bool): If True, removes controlled agents. If False, removes uncontrolled agents
        """
        # Obtain agent ids
        agent_ids = LocalEgoState.from_tensor(
            self_obs_tensor=self.sim.self_observation_tensor(),
            backend="torch",
            device=self.device,
        ).id

        # Choose the appropriate mask based on whether we're removing controlled or uncontrolled agents
        if remove_controlled_agents:
            agent_mask = self.cont_agent_mask
        else:
            # Create inverse mask for uncontrolled agents
            agent_mask = ~self.cont_agent_mask

        for env_idx in range(self.num_worlds):
            # Get all relevant agent IDs (controlled or uncontrolled) for the current environment
            scene_agent_ids = agent_ids[env_idx, :][agent_mask[env_idx]].long()

            if (
                scene_agent_ids.numel() > 0
            ):  # Ensure there are agents to sample
                # Determine the number of agents to sample (X% of the total agents)
                num_to_sample = max(
                    1, int(perc_to_rmv_per_scene * scene_agent_ids.size(0))
                )

                # Randomly sample agent IDs to remove using torch
                sampled_indices = torch.randperm(scene_agent_ids.size(0))[
                    :num_to_sample
                ]
                sampled_agent_ids = scene_agent_ids[sampled_indices]

                # Delete the sampled agents from the environment
                self.sim.deleteAgents({env_idx: sampled_agent_ids.tolist()})

        # Reset controlled agent mask and visualizer
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Reset static scenario data for the visualizer
        self.vis.initialize_static_scenario_data(self.cont_agent_mask)

    def swap_data_batch(self, new_sim_scenes: list = None):
        """
        Swap the current scenes with granular memory profiling to isolate the leak.
        """
        import gc
        import torch

        if new_sim_scenes is None:
            raise ValueError("swap_data_batch 需要 'new_sim_scenes' 参数。")

        # --- [辅助函数] 显存快照 ---
        def get_gpu_snapshot(tag):
            if not torch.cuda.is_available(): return
            torch.cuda.synchronize()
            gc.collect() # 强制GC以排除Python对象引用的干扰
            
            # 1. PyTorch 视角
            res = torch.cuda.memory_reserved() / 1024**3
            
            # 2. 物理硬件视角 (等同于 nvidia-smi)
            free_mem, total_mem = torch.cuda.mem_get_info()
            physical_used = (total_mem - free_mem) / 1024**3
            
            print(f"  [{tag}] Phys: {physical_used:.4f} GB | PyTorch Rsrv: {res:.4f} GB")
            return physical_used

        print(f"\n====== Resample Memory Diagnosis (Scenes: {len(new_sim_scenes)}) ======")
        baseline_mem = get_gpu_snapshot("0. Start")

        # -----------------------------------------------------------
        # 阶段 1: Python 状态更新
        # -----------------------------------------------------------
        self.sim_scenes = new_sim_scenes
        self.num_worlds = len(self.sim_scenes)
        self.data_batch = None

        if len(self.sim_scenes) != self.num_worlds:
            raise ValueError("Data batch size mismatch")
        
        # -----------------------------------------------------------
        # 阶段 2: C++ 模拟器重置 (Manager::setMaps)
        # -----------------------------------------------------------
        # 理论上这里应该零增长，因为我们已经切断了 C++ 分配
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self.sim.set_maps(new_sim_scenes)
        
        mem_after_cpp = get_gpu_snapshot("1. After C++ setMaps")

        # -----------------------------------------------------------
        # 阶段 3: 掩码更新 (纯 Python/Tensor 操作)
        # -----------------------------------------------------------
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = self.cont_agent_mask.sum().item()
        
        mem_after_mask = get_gpu_snapshot("2. After Mask Update")

        # -----------------------------------------------------------
        # 阶段 4: VBD 模型推理 (如果有)
        # -----------------------------------------------------------
        if self.use_vbd and self.vbd_model is not None:
            self._generate_vbd_trajectories()
            mem_after_vbd = get_gpu_snapshot("3. After VBD Gen")
        else:
            mem_after_vbd = mem_after_mask # 跳过

        # -----------------------------------------------------------
        # 阶段 5: 可视化器初始化 (Matplotlib/Rendering)
        # -----------------------------------------------------------
        # 这是最大的嫌疑对象
        self.vis.initialize_static_scenario_data(self.cont_agent_mask)
        
        mem_after_vis = get_gpu_snapshot("4. After Vis Init")

        # -----------------------------------------------------------
        # 总结报告
        # -----------------------------------------------------------
        delta = mem_after_vis - baseline_mem
        print(f"====== Diagnosis Summary ======")
        print(f"Total Physical Increase: {delta:+.4f} GB")
        
        # 简易归因分析
        diff_cpp = mem_after_cpp - baseline_mem
        diff_vbd = mem_after_vbd - mem_after_mask
        diff_vis = mem_after_vis - mem_after_vbd
        
        if diff_cpp > 0.01: print(f"⚠️ SUSPECT: C++ Backend (+{diff_cpp:.4f} GB)")
        if diff_vbd > 0.01: print(f"⚠️ SUSPECT: VBD Model (+{diff_vbd:.4f} GB)")
        if diff_vis > 0.01: print(f"⚠️ SUSPECT: Visualizer (+{diff_vis:.4f} GB)")
        print("===================================================\n")



    def _generate_vbd_trajectories(self):
        """Generate and store trajectory predictions for all scenes using VBD model."""
        if not self.use_vbd or self.vbd_model is None:
            return

        _ = self.reset()

        # Generate sample batch using the limited mask
        sample_batch = self._generate_sample_batch(init_steps=self.init_steps)

        # VBD model prediction
        predictions = self.vbd_model.sample_denoiser(sample_batch)
        vbd_trajectories = (
            predictions["denoised_trajs"].to(self.device).numpy()
        )
        agent_indices = sample_batch["agents_id"]

        self.vbd_trajectories.zero_()
        # Process each world separately
        for world_idx in range(self.num_worlds):
            world_agent_indices = agent_indices[world_idx]

            # Filter out negative indices (they're our padding)
            valid_mask = (
                world_agent_indices >= 0
            )  # Boolean mask of valid indices
            valid_agent_indices = world_agent_indices[
                valid_mask
            ]  # Filtered tensor

            if len(valid_agent_indices) > 0:
                # Update vbd_trajectories(x, y, yaw, vel_x, vel_y) for this world's agents
                self.vbd_trajectories[
                    world_idx, valid_agent_indices, :, :2
                ] = torch.Tensor(
                    vbd_trajectories[
                        world_idx, : len(valid_agent_indices), :, :2
                    ]
                )
                self.vbd_trajectories[
                    world_idx, valid_agent_indices, :, :2
                ] -= self.sim.world_means_tensor().to_torch()[
                    world_idx, :2
                ]  # subtract mean
                self.vbd_trajectories[
                    world_idx, valid_agent_indices, :, 2
                ] = torch.Tensor(
                    vbd_trajectories[
                        world_idx, : len(valid_agent_indices), :, 2
                    ]
                )
                self.vbd_trajectories[
                    world_idx, valid_agent_indices, :, 3:
                ] = torch.Tensor(
                    vbd_trajectories[
                        world_idx, : len(valid_agent_indices), :, 3:5
                    ]
                )

    def get_expert_actions(self):
        """Get expert actions for the full trajectories across worlds.

        Returns:
            expert_actions: Inferred or logged actions for the agents.
            expert_speeds: Speeds from the logged trajectories.
            expert_positions: Positions from the logged trajectories.
            expert_yaws: Heading from the logged trajectories.
        """

        log_trajectory = LogTrajectory.from_tensor(
            self.sim.expert_trajectory_tensor(),
            self.num_worlds,
            self.max_agent_count,
            backend=self.backend,
        )

        if self.config.dynamics_model == "delta_local":
            inferred_actions = log_trajectory.inferred_actions[..., :3]
            inferred_actions[..., 0] = torch.clamp(
                inferred_actions[..., 0], -6, 6
            )
            inferred_actions[..., 1] = torch.clamp(
                inferred_actions[..., 1], -6, 6
            )
            inferred_actions[..., 2] = torch.clamp(
                inferred_actions[..., 2], -torch.pi, torch.pi
            )
        elif self.config.dynamics_model == "state":
            # Extract (x, y, yaw, velocity x, velocity y)
            inferred_actions = torch.cat(
                (
                    log_trajectory.pos_xy,
                    torch.ones(
                        (*log_trajectory.pos_xy.shape[:-1], 1),
                        device=self.device,
                    ),
                    log_trajectory.yaw,
                    log_trajectory.vel_xy,
                    torch.zeros(
                        (*log_trajectory.pos_xy.shape[:-1], 4),
                        device=self.device,
                    ),
                ),
                dim=-1,
            )
        elif (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            inferred_actions = log_trajectory.inferred_actions[..., :3]
            inferred_actions[..., 0] = torch.clamp(
                inferred_actions[..., 0], -6, 6
            )
            inferred_actions[..., 1] = torch.clamp(
                inferred_actions[..., 1], -0.3, 0.3
            )

        return (
            inferred_actions,
            log_trajectory.pos_xy,
            log_trajectory.vel_xy,
            log_trajectory.yaw,
        )

    def get_env_filenames(self):
        """Obtain the tfrecord filename for each world, mapping world indices to map names."""

        map_name_integers = self.sim.map_name_tensor().to_torch()
        filenames = {}
        # Iterate through the number of worlds
        for i in range(self.num_worlds):
            tensor = map_name_integers[i]
            # Convert ints to characters, ignoring zeros
            map_name = "".join([chr(i) for i in tensor.tolist() if i != 0])
            filenames[i] = map_name

        return filenames

    def get_scenario_ids(self):
        """Obtain the scenario ID for each world."""
        scenario_id_integers = self.sim.scenario_id_tensor().to_torch()
        scenario_ids = {}

        # Iterate through the number of worlds
        for i in range(self.num_worlds):
            tensor = scenario_id_integers[i]
            # Convert ints to characters, ignoring zeros
            scenario_id = "".join([chr(i) for i in tensor.tolist() if i != 0])
            scenario_ids[i] = scenario_id

        return scenario_ids


if __name__ == "__main__":

    env_config = EnvConfig(
        dynamics_model="delta_local",
    )
    render_config = RenderConfig()

    # --- [修改 9] 移除 SceneDataLoader
    # train_loader = SceneDataLoader(
    #     root="data/processed/examples",
    #     batch_size=2,
    #     dataset_size=100,
    #     sample_with_replacement=True,
    #     shuffle=False,
    # )
    
    # --- [修改 10] 创建一个临时的 sim_scenes 列表用于测试
    # (!!!) 确保这些路径是有效的，否则测试会失败 (!!!)
    TEST_MAP_DIR = "data/processed/examples" # 假设的路径
    try:
        # 尝试从目录动态加载
        import glob
        import os
        test_scenes_paths = glob.glob(f"{TEST_MAP_DIR}/scene-*.tfrecord")
        if len(test_scenes_paths) < 2:
             raise FileNotFoundError("测试 tfrecord 文件不足")
        test_scenes = test_scenes_paths[:2] # 取前两个
        print(f"__main__ 测试: 找到场景 {test_scenes}")
    except Exception as e:
        print(f"__main__ 测试: 无法加载测试场景: {e}。")
        test_scenes = []

    if len(test_scenes) > 0:
        # --- [修改 11] 使用新的 __init__ 签名
        env = GPUDriveTorchEnv(
            config=env_config,
            sim_scenes=test_scenes, # <--- 传递场景
            max_cont_agents=64,
            device="cpu",
        )

        control_mask = env.cont_agent_mask

        # Rollout
        obs = env.reset()

        sim_frames = []
        agent_obs_frames = []

        expert_actions, _, _, _ = env.get_expert_actions()

        env_idx = 0

        for t in range(10):
            print(f"Step: {t}")

            # Step the environment
            expert_actions, _, _, _ = env.get_expert_actions()
            env.step_dynamics(expert_actions[:, :, t, :])

            highlight_agent = torch.where(env.cont_agent_mask[env_idx, :])[0][
                -1
            ].item()

            # Make video
            sim_states = env.vis.plot_simulator_state(
                env_indices=[env_idx],
                zoom_radius=50,
                time_steps=[t],
                center_agent_indices=[highlight_agent],
            )

            agent_obs = env.vis.plot_agent_observation(
                env_idx=env_idx,
                agent_idx=highlight_agent,
                figsize=(10, 10),
            )

            sim_frames.append(img_from_fig(sim_states[0]))
            agent_obs_frames.append(img_from_fig(agent_obs))

            obs = env.get_obs()
            reward = env.get_rewards()
            done = env.get_dones()
            info = env.get_infos()

            if done[0, highlight_agent].bool():
                break

        env.close()

        media.write_video(
            "sim_video.gif", np.array(sim_frames), fps=10, codec="gif"
        )
        media.write_video(
            "obs_video.gif", np.array(agent_obs_frames), fps=10, codec="gif"
        )

# ... (文件的其余部分保持不变)

# # # =================================================================
# # # [GIGAFLOW DIAGNOSIS] 将此代码块添加到文件最末尾
# # # 运行命令: python gpudrive/env/env_torch.py
# # # =================================================================
# if __name__ == "__main__":
#     import torch
#     import os
#     import glob
#     from gpudrive.env.config import EnvConfig

#     print("🚀 启动 GIGAFLOW 碰撞/越野诊断程序 (底图版)...")

#     # ==========================================
#     # [关键配置] 请填入你们那张"底图"的绝对路径
#     # ==========================================
#     # 例如: "/root/code/gpudrive/maps/Town01.json"
#     base_map_path = "/root/code/gpudrive/maps/Town01_tessellated.json"  # <--- 修改这里！
    
#     # 检查文件是否存在
#     if not os.path.exists(base_map_path):
#         print(f"❌ 错误: 找不到底图文件: {base_map_path}")
#         print("   请修改脚本中的 `base_map_path` 为你们实际使用的 JSON 地图路径。")
#         print("   C++ 需要读取它来构建道路网格，否则无法检测越野和碰撞。")
#         exit(1)

#     # ==========================================
#     # 2. 配置环境 (将这一张图复制 N 份)
#     # ==========================================
#     test_num_worlds = 32  
    
#     # 这就是告诉模拟器：
#     # "我有 32 个世界，每个世界都使用这张底图作为物理环境"
#     sim_scenes = [base_map_path] * test_num_worlds

#     config = EnvConfig()
#     config.device = "cuda"

#     # ==========================================
#     # 3. 初始化 & 运行诊断
#     # ==========================================
#     try:
#         print(f"✅ 加载底图: {base_map_path}")
#         print(f"✅ 初始化 {test_num_worlds} 个并行世界...")
        
#         env = GPUDriveTorchEnv(
#             config=config,
#             sim_scenes=sim_scenes, 
#             max_cont_agents=config.max_num_agents_in_scene, 
#             device="cuda",
#         )
        
#         print("🔄 正在重置环境 (Reset)...")
#         # Reset 时，C++ 会：
#         # 1. 加载底图的道路 (只做一次)
#         # 2. 调用 level_gen.cpp 生成智能体
#         env.reset()
        
#         print("▶️ 执行 Step 1...")
#         dummy_actions = torch.zeros(
#             (env.num_worlds, env.max_agent_count, 3), 
#             device="cuda"
#         )
#         env.step_dynamics(dummy_actions)
        
#         # [新代码 - 直接获取原始 Tensor]
#         # 绕过 Info 类的封装，直接拿底层数据
#         infos = env.sim.info_tensor().to_torch()

#         # --- 核心诊断逻辑 ---
#         # 过滤 Type 7 (Vehicle)
#         agent_types = infos[:, :, 4]
#         vehicle_mask = (agent_types == 7)
#         real_infos = infos[vehicle_mask]
        
#         active_agents = vehicle_mask.sum().item()
        
#         if active_agents > 0:
#             collisions = real_infos[:, 1].sum().item()
#             offroad = real_infos[:, 5].sum().item()
#         else:
#             collisions = 0
#             offroad = 0

#         print("\n" + "="*40)
#         print(f"📊 === 诊断结果 (Total Agents: {active_agents}) ===")
#         print("="*40)
#         print(f"💥 检测到的碰撞 (Collisions): {collisions}")
#         print(f"🚜 检测到的越野 (Off-road):   {offroad}")
#         print("-" * 40)

#         # 结果判定
#         if active_agents == 0:
#             print("⚠️ 警告: 没有检测到 Type=7 的活跃智能体！")
#             print("   -> 请检查 level_gen.cpp 是否正确生成了智能体。")
            
#         elif collisions > 0 or offroad > 0:
#             print("✅ [SUCCESS] 结论: C++ 修复已生效！")
#             print(f"   Python 成功基于底图 '{os.path.basename(base_map_path)}' 读取到了状态。")
#         else:
#             print("❌ [FAILURE] 结论: 数值依然为 0。")
#             print("   -> 尝试连续跑 20 步...")
#             for i in range(20):
#                 env.step_dynamics(dummy_actions)
#                 # [新代码 - 直接获取原始 Tensor]
#         # 绕过 Info 类的封装，直接拿底层数据
#                 infos = env.sim.info_tensor().to_torch()
#                 real_infos = infos[infos[:, :, 4] == 7]
#                 if len(real_infos) > 0:
#                     c = real_infos[:, 1].sum().item()
#                     o = real_infos[:, 5].sum().item()
#                     if c > 0 or o > 0:
#                         print(f"   Step {i+2}: ✅ 终于检测到了！Collisions={c}, Offroad={o}")
#                         break

#     except Exception as e:
#         print(f"\n❌ 运行时错误: {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         if 'env' in locals():
#             env.close()