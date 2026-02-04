
"""
Vectorized environment wrapper for multi-agent environments.
[已清理版本] - 移除了无用的渲染代码和重复的黑匣子系统
"""
import logging
import os
from typing import Optional, Sequence
from collections import defaultdict

import torch
import numpy as np
import gymnasium as gym
import random

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
)

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import RenderConfig, RenderMode
from gpudrive.env import constants

logging.basicConfig(level=logging.INFO)


class SB3MultiAgentEnv(VecEnv):
    """
    Casts multi-agent environments as vectorized environments.
    
    [清理说明]
    - 移除了 GIGAFLOW 黑匣子代码 (使用 visualizer 代替)
    - 移除了未使用的 render_env, log_video_to_wandb 等方法
    - 修复了 step() 的返回值
    - 完善了可视化集成
    """

    def __init__(
        self,
        config,
        exp_config,
        max_cont_agents,
        device,
        sim_scenes: list,
        base_maps: list = None,
        render_mode="rgb_array",
        collision_weight=-0.5,
        goal_achieved_weight=1.0,
        off_road_weight=-0.5,
        log_distance_weight=0.01,
        **kwargs,
    ):
        # =====================================================================
        # 强制使用 RGB_ARRAY 模式 (最轻量级)
        # =====================================================================
        internal_render_config = RenderConfig(render_mode=RenderMode.RGB_ARRAY)
        self.render_mode = render_mode

        # 初始化底层 Torch 环境
        self.base_maps = base_maps if base_maps is not None else list(set(sim_scenes))
        self._env = GPUDriveTorchEnv(
            config=config,
            sim_scenes=sim_scenes,
            max_cont_agents=max_cont_agents,
            device=device,
            render_config=internal_render_config,
        )

        # 初始化父类 VecEnv
        super().__init__(
            self._env.num_worlds,
            self._env.observation_space,
            self._env.action_space,
        )

        # =====================================================================
        # 配置与参数保存
        # =====================================================================
        self.config = config
        self.exp_config = exp_config
        self.device = device
        
        # 场景路径处理
        self.all_scene_paths = [
            os.path.join(self.exp_config.data_dir, scene)
            for scene in sorted(os.listdir(self.exp_config.data_dir))
            if scene.startswith("tfrecord")
        ]
        self.unique_scene_paths = list(set(self.all_scene_paths))
        
        # 环境维度信息
        self.num_worlds = self._env.num_worlds
        self.max_agent_count = self._env.max_agent_count
        self.num_envs = self._env.cont_agent_mask.sum().item()
        self.controlled_agent_mask = self._env.cont_agent_mask.clone()
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        
        # =====================================================================
        # 观测空间定义
        # =====================================================================
        self.compact_obs_dim = 20
        self.full_obs_dim = (
            constants.EGO_FEAT_DIM + 
            (constants.MAX_PARTNER_COUNT * constants.PARTNER_FEAT_DIM) + 
            (constants.MAX_ROAD_OBS_COUNT * constants.ROAD_GRAPH_FEAT_DIM)
        )
        
        self.obs_dim = self.full_obs_dim
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.full_obs_dim,), np.float32
        )
        
        print(f"\n[SB3 WRAPPER] Obs Space Config:")
        print(f"  > Compact Dim (Storage): {self.compact_obs_dim}")
        print(f"  > Full Dim (Network):    {self.full_obs_dim}")
        print(f"  > Observation Space:     {self.observation_space.shape}")

        # =====================================================================
        # 缓冲区与追踪器初始化
        # =====================================================================
        self.info_dim = 6
        self.episode_metrics = defaultdict(list)
        self.iters = 0
        self.num_episodes = 0
        self.info_dict = {}

        # 步数与动作追踪
        self.agent_step = torch.zeros((self.num_worlds, self.max_agent_count)).to(self.device)
        self.actions_tensor = torch.zeros((self.num_worlds, self.max_agent_count)).to(self.device)
        
        # PPO 存储缓冲区 (填充 NaN)
        self.buf_rews = torch.full((self.num_worlds, self.max_agent_count), fill_value=float("nan")).to(self.device)
        self.buf_dones = torch.full((self.num_worlds, self.max_agent_count), fill_value=float("nan")).to(self.device)
        self.buf_obs = torch.full((self.num_envs, self.obs_dim), fill_value=float("nan")).to(self.device)

        # 奖励权重
        self.collision_weight = collision_weight
        self.goal_achieved_weight = goal_achieved_weight
        self.off_road_weight = off_road_weight
        self.log_distance_weight = log_distance_weight

        # Episode 长度追踪
        self.episode_lengths = torch.zeros(self.num_worlds, dtype=torch.int32)

        # =====================================================================
        # 可视化器 (延迟初始化)
        # =====================================================================
        self.visualizer = None
        self._viz_enabled = False
        self._viz_step = 0
        self._global_step = 0

    # =========================================================================
    # 可视化控制方法
    # =========================================================================
    
    def enable_visualization(self, output_dir: str = "training_viz", 
                            map_path: Optional[str] = None):
        """
        启用训练过程可视化
        
        Args:
            output_dir: 视频输出目录
            map_path: 地图 JSON 路径（用于绘制道路）
        """
        try:
            from .training_process_visualizer import TrainingProcessVisualizer
            self.visualizer = TrainingProcessVisualizer(
                env=self,
                output_dir=output_dir,
                view_mode='global',
            )
            self._viz_enabled = True
            print(f"[SB3Wrapper] Visualization enabled. Output: {output_dir}")
        except ImportError as e:
            print(f"[SB3Wrapper] Warning: Could not import TrainingProcessVisualizer: {e}")
            self._viz_enabled = False
    
    def start_recording(self):
        """开始录制当前 episode"""
        if self.visualizer:
            self.visualizer.start_recording()
            self._viz_step = 0
            print(f"[SB3Wrapper] Recording started")
    
    def save_recording(self, tag: str = "") -> Optional[str]:
        """保存当前录制"""
        if self.visualizer and self.visualizer.is_recording:
            return self.visualizer.save_recording(tag)
        return None
    
    def is_recording(self) -> bool:
        """检查是否正在录制"""
        return self.visualizer is not None and self.visualizer.is_recording

    # =========================================================================
    # 观测处理方法
    # =========================================================================

    def reconstruct_observations(self, compact_obs: torch.Tensor):
        """调用 C++ 接口重建数据，并执行归一化"""
        if not compact_obs.is_contiguous():
            compact_obs = compact_obs.contiguous()
            
        ptr = compact_obs.data_ptr()
        rows = compact_obs.shape[0]
        cols = compact_obs.shape[1]
        
        obs_raw = self._env.sim.reconstruct_observations(ptr, rows, cols).to_torch()
        obs_norm = self._env._normalize_reconstructed_obs(obs_raw)
        
        return obs_norm

    def get_compact_obs(self):
        """仅提取核心物理状态用于存储"""
        sim = self._env.sim
        self_obs = sim.self_observation_tensor().to_torch()
        abs_obs = sim.absolute_self_observation_tensor().to_torch()
        
        # [0-2]: Position (x,y,z), [3-6]: Rotation (Quat x,y,z,w)
        pos_rot = abs_obs[..., :7] 
        
        # 拼接数据
        # self_obs (13) + pos_rot (7) = 20 维
        compact_full = torch.cat([self_obs, pos_rot], dim=-1)
        
        flat_mask = self.controlled_agent_mask.view(-1)
        
        # [GIGAFLOW FIX] 关键修改：不要使用 self.compact_obs_dim
        # 直接使用 compact_full.shape[-1] (即 20)，自动适应维度变化
        flat_obs = compact_full.view(-1, compact_full.shape[-1])

        return flat_obs[flat_mask].clone()

    # =========================================================================
    # 核心环境方法
    # =========================================================================

    def _reset_seeds(self) -> None:
        """Reset all environments' seeds."""
        self._seeds = None

    def reset(self, world_idx=None, seed=None):
        """Reset environment and return initial observations."""
        
        # 分支 1: 全局重置
        if world_idx is None:
            self.episode_lengths = torch.zeros(self.num_worlds, dtype=torch.int32)
            self._env.reset()
            
            # 执行一次空动作 (Dummy Step)
            dummy_actions = torch.zeros((self.num_worlds, self.max_agent_count, 3)).to(self.device)
            self._env.step_dynamics(dummy_actions)
            
            # 获取初始观测
            obs = self.get_compact_obs()
            
            # 重置指标追踪器
            self.episode_tracker = torch.zeros(
                (self.num_worlds, self.max_agent_count, 4), 
                device=self.device, 
                dtype=torch.float32
            )
            self.reward_comp_tracker = torch.zeros(
                (self.num_worlds, self.max_agent_count, 2), 
                device=self.device, 
                dtype=torch.float32
            )

            # 重置死亡掩码
            self.dead_agent_mask = ~self.controlled_agent_mask.clone()
            
            return obs

        # 分支 2: 局部重置
        else:
            self._env.sim.reset(world_idx.item())
            self.episode_lengths[world_idx] = 0
            return None

    def step(self, actions) -> VecEnvStepReturn:
        """执行环境步进"""
        self.info_dict = {}
        self.iters += 1
        self._global_step += 1

        # 1. 准备动作 Tensor
        self.actions_tensor[self.controlled_agent_mask] = actions

        # 2. 执行物理步进
        self._env.step_dynamics(self.actions_tensor)

        # 3. 获取 Reward
        reward = self._env.get_rewards(
            collision_weight=self.collision_weight,
            goal_achieved_weight=self.goal_achieved_weight,
            off_road_weight=self.off_road_weight,
            log_distance_weight=self.log_distance_weight
        ).clone()

        # 追踪奖励分量
        if hasattr(self._env, "latest_reward_components"):
            comps = self._env.latest_reward_components
            current_comps = torch.stack([
                comps["rew_speed"], 
                comps["rew_goal_dist"]
            ], dim=-1)
            mask_expanded = self.controlled_agent_mask.unsqueeze(-1)
            self.reward_comp_tracker += (current_comps * mask_expanded)

        # 4. 获取 Done 信号
        done = self._env.get_dones().clone()
        
        # 5. 获取 Info Tensor
        info = self._env.sim.info_tensor().to_torch()
        
        current_info_tensor = torch.stack([
            info[:, :, 5],  # OffRoad
            info[:, :, 1],  # VehCol
            info[:, :, 2],  # NonVehCol
            info[:, :, 3]   # GoalAchieved
        ], dim=-1).float()

        mask = self.controlled_agent_mask.unsqueeze(-1)
        self.episode_tracker += (current_info_tensor * mask)

        # 执行超时截断
        is_timeout = (self.agent_step >= self.config.episode_len - 1)
        is_timeout = is_timeout & self.controlled_agent_mask
        done = torch.logical_or(done, is_timeout)

        # 6. 检查是否有世界结束
        done_worlds = torch.where(
            (done.nan_to_num(0) * self.controlled_agent_mask).sum(dim=1)
            == self.controlled_agent_mask.sum(dim=1)
        )[0]
        
        if len(done_worlds) > 0:
            self._update_info_dict(self.episode_tracker, self.reward_comp_tracker, done_worlds)
            self.num_episodes += len(done_worlds)
            
            self.episode_tracker[done_worlds] = 0.0
            self.reward_comp_tracker[done_worlds] = 0.0
            
            self._env.sim.reset(done_worlds.tolist())
            self.episode_lengths[done_worlds] = -1
        
        self.episode_lengths += 1

        # 7. 更新缓冲区
        self.buf_rews[self.dead_agent_mask] = torch.nan
        self.buf_rews[~self.dead_agent_mask] = reward[~self.dead_agent_mask]
        
        self.buf_dones[self.dead_agent_mask] = torch.nan
        self.buf_dones[~self.dead_agent_mask] = done[~self.dead_agent_mask].float()

        # 8. 更新 Agent 存活状态
        self.agent_step += 1
        self.dead_agent_mask = torch.logical_or(self.dead_agent_mask, done)

        if len(done_worlds) > 0:
            for world_idx in done_worlds:
                self.dead_agent_mask[world_idx, :] = ~self.controlled_agent_mask[world_idx, :].clone()
            self.agent_step[done_worlds] = 0

        # 9. 获取下一步观测
        next_obs = self.get_compact_obs()
        self.obs_alive = next_obs

        # =====================================================================
        # [可视化] 记录训练数据
        # =====================================================================
        if self._viz_enabled and self.visualizer and self.visualizer.is_recording:
            self.visualizer.record_training_step(
                step=self._viz_step,
                observations=next_obs,
                actions=actions,
                rewards=self.buf_rews,
                dones=self.buf_dones,
                global_step=self._global_step,
                world_idx=0,
            )
            self._viz_step += 1

        # 10. 格式化返回给 SB3
        # [修复] 返回正确的变量
        return (
            next_obs, 
            self.buf_rews[self.controlled_agent_mask].reshape(self.num_envs).clone(),
            self.buf_dones[self.controlled_agent_mask].reshape(self.num_envs).clone(),
            info[self.controlled_agent_mask].reshape(self.num_envs, self.info_dim).clone(),
        )

    def close(self) -> None:
        """Close the environment."""
        # 保存未完成的录制
        if self.visualizer and self.visualizer.is_recording:
            self.save_recording("final")
        self._env.close()

    def seed(self, seed=None):
        """Set the random seeds for all environments."""
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def resample_scenario_batch(self):
        """Swap out the dataset."""
        new_sim_scenes = [random.choice(self.base_maps) for _ in range(self.num_worlds)]
        self._env.swap_data_batch(new_sim_scenes)

        self.controlled_agent_mask = self._env.cont_agent_mask.clone()
        self.max_agent_count = self._env.max_agent_count
        self.num_valid_controlled_agents_across_worlds = (
            self._env.num_valid_controlled_agents_across_worlds
        )
        self.num_envs = self.controlled_agent_mask.sum().item()

    def _update_info_dict(self, info, reward_tracker, indices) -> None:
        """Update the info logger with metrics."""
        controlled_agent_info = info[indices][self.controlled_agent_mask[indices]]

        total_off_road = controlled_agent_info[:, 0].sum().item()
        total_veh_collisions = controlled_agent_info[:, 1].sum().item()
        total_non_veh_collisions = controlled_agent_info[:, 2].sum().item()
        total_goal_achieved = controlled_agent_info[:, 3].sum().item()
        finished_rewards = reward_tracker[indices][self.controlled_agent_mask[indices]]
        
        num_agents = self.controlled_agent_mask[indices].sum().item()
        total_speed_rew = finished_rewards[:, 0].sum().item()
        total_dist_rew = finished_rewards[:, 1].sum().item()
        
        current_info = {
            "off_road": total_off_road,
            "veh_collisions": total_veh_collisions,
            "non_veh_collision": total_non_veh_collisions,
            "goal_achieved": total_goal_achieved,
            "num_controlled_agents": num_agents,
        }

        if num_agents > 0:
            self.info_dict["per_vehicle_rew_speed"] = total_speed_rew / num_agents
            self.info_dict["per_vehicle_rew_dist"] = total_dist_rew / num_agents
            current_info["per_vehicle_collision"] = total_veh_collisions / num_agents
            current_info["per_vehicle_off_road"] = total_off_road / num_agents
            current_info["per_vehicle_goal_achieved"] = total_goal_achieved / num_agents
        else:
            self.info_dict["per_vehicle_rew_speed"] = 0.0
            self.info_dict["per_vehicle_rew_dist"] = 0.0
            current_info["per_vehicle_collision"] = 0.0
            current_info["per_vehicle_off_road"] = 0.0
            current_info["per_vehicle_goal_achieved"] = 0.0

        for key, value in current_info.items():
            self.episode_metrics[key].append(value)
        self.info_dict.update(current_info)

        self.info_dict["truncated"] = (
            (
                (self.agent_step[indices] == self.config.episode_len - 1)
                * ~self.dead_agent_mask[indices]
            )
            .sum()
            .item()
        )

    # =========================================================================
    # SB3 VecEnv 接口实现
    # =========================================================================

    def get_attr(self, attr_name, indices=None):
        """获取环境属性"""
        if hasattr(self, attr_name):
            value = getattr(self, attr_name)
        elif hasattr(self, "_env") and hasattr(self._env, attr_name):
            value = getattr(self._env, attr_name)
        else:
            value = None

        if indices is None:
            return [value] * self.num_envs
        if isinstance(indices, int):
            return [value]
        return [value] * len(indices)

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        if hasattr(self, method_name):
            method = getattr(self, method_name)
        elif hasattr(self, "_env") and hasattr(self._env, method_name):
            method = getattr(self._env, method_name)
        else:
            raise AttributeError(f"Method {method_name} not found.")

        result = method(*method_args, **method_kwargs)
        
        if indices is None:
            return [result] * self.num_envs
        elif isinstance(indices, int):
            return [result]
        else:
            return [result] * len(indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environment is wrapped with a given wrapper."""
        if indices is None:
            return [False] * self.num_envs
        elif isinstance(indices, int):
            return [False]
        else:
            return [False] * len(indices)

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()

    def get_images(self, policy=None) -> Sequence[Optional[np.ndarray]]:
        frames = [self._env.render()]
        return frames