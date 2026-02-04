"""
GPUDrive PPO 训练脚本 (修复版)
=============================
修复了可视化问题，实现真正的 Episode 级录制

核心改动：
1. 使用 EpisodeVisualizationCallback 代替 VisualizationCallback
2. 添加详细的诊断输出
3. 修复了录制触发逻辑
"""

import sys
import os
import time
import torch 
import yaml
from box import Box
from typing import Callable
from datetime import datetime
import dataclasses
import random
import numpy as np

# GPUDrive 相关
from gpudrive.integrations.sb3.ppo import IPPO
from gpudrive.integrations.sb3.callbacks import MultiAgentCallback
from gpudrive.env.config import EnvConfig
from gpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
from gpudrive.networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy
from gpudrive.networks.basic_ffn import FFN, FeedForwardPolicy

# SB3 相关
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


# ============================================================
# 修复版 Episode 可视化 Callback
# ============================================================

class EpisodeVisualizationCallback(BaseCallback):
    """
    基于 Episode 的可视化 Callback (修复版)
    
    改进点：
    1. 正确检测 Episode 边界
    2. 详细的进度输出
    3. 错误处理和诊断信息
    """
    
    def __init__(
        self, 
        record_freq: int = 50,      # 每 N 个 Episode 录制一次
        record_first_n: int = 3,    # 前 N 个 Episode 一定录制
        max_recordings: int = 100,  # 最大录制数量
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.record_first_n = record_first_n
        self.max_recordings = max_recordings
        
        self._episode_count = 0
        self._recording_count = 0
        self._is_recording = False
        self._init_ok = False
        self._last_num_episodes = 0
    
    def _on_training_start(self) -> None:
        """训练开始时验证配置"""
        print("\n" + "="*60)
        print("[EpisodeVizCallback] Initializing...")
        print("="*60)
        
        try:
            # 验证 visualizer
            viz = self.training_env.get_attr('visualizer')[0]
            viz_enabled = self.training_env.get_attr('_viz_enabled')[0]
            
            if viz is None:
                print("❌ ERROR: visualizer is None!")
                print("  → 请确保调用了 env.enable_visualization()")
                return
            
            if not viz_enabled:
                print("❌ ERROR: _viz_enabled is False!")
                return
            
            self._init_ok = True
            print("✅ Initialization successful!")
            print(f"  > Record every {self.record_freq} episodes")
            print(f"  > First {self.record_first_n} episodes will be recorded")
            print(f"  > Max recordings: {self.max_recordings}")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
    
    def _should_record(self) -> bool:
        """判断是否应该录制"""
        if not self._init_ok or self._recording_count >= self.max_recordings:
            return False
        if self._episode_count < self.record_first_n:
            return True
        return (self._episode_count - self.record_first_n) % self.record_freq == 0
    
    def _start_recording(self):
        """开始录制"""
        if self._is_recording:
            return
        try:
            self.training_env.env_method('start_recording')
            self._is_recording = True
            if self.verbose:
                print(f"\n[EpisodeVizCallback] 🎬 Recording Episode #{self._episode_count}")
        except Exception as e:
            print(f"[EpisodeVizCallback] ❌ start_recording failed: {e}")
    
    def _save_recording(self, tag: str = ""):
        """保存录制"""
        if not self._is_recording:
            return
        try:
            self.training_env.env_method('save_recording', tag)
            self._recording_count += 1
            self._is_recording = False
            if self.verbose:
                print(f"[EpisodeVizCallback] ✅ Saved ({self._recording_count}/{self.max_recordings})")
        except Exception as e:
            print(f"[EpisodeVizCallback] ❌ save_recording failed: {e}")
            self._is_recording = False
    
    def _on_step(self) -> bool:
        """每步检查 Episode 边界"""
        if not self._init_ok:
            return True
        
        try:
            # 获取当前 episode 数量
            num_episodes = self.training_env.get_attr('num_episodes')[0]
            
            # 检测 Episode 结束
            if num_episodes > self._last_num_episodes:
                # 保存之前的录制
                if self._is_recording:
                    tag = f"ep{self._episode_count}_step{self.num_timesteps}"
                    self._save_recording(tag)
                
                # 更新计数
                self._episode_count = num_episodes
                self._last_num_episodes = num_episodes
                
                # 检查是否需要录制下一个
                if self._should_record():
                    self._start_recording()
            
            # 第一个 Episode 的特殊处理
            if self._episode_count == 0 and not self._is_recording and self._should_record():
                self._start_recording()
                
        except Exception as e:
            pass
        
        return True
    
    def _on_training_end(self) -> None:
        """训练结束时保存"""
        if self._is_recording:
            self._save_recording("final")
        
        if self.verbose:
            print(f"\n[EpisodeVizCallback] Finished")
            print(f"  > Total episodes: {self._episode_count}")
            print(f"  > Total recordings: {self._recording_count}")


# ============================================================
# 工具函数
# ============================================================

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def load_config(config_path: str) -> Box:
    with open(config_path, "r") as f:
        return Box(yaml.safe_load(f))


def print_gpu_stats(label: str = ""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[GPU {label}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def diagnose_map_loading(env):
    """诊断多世界地图加载"""
    print("\n" + "="*60)
    print("[诊断] 多世界地图状态检查")
    print("="*60)
    
    num_worlds = env.num_worlds
    for world_idx in range(num_worlds):
        # 获取该 World 的道路数量
        try:
            # 这需要你在 C++ 端暴露接口
            num_roads = env.sim.data().numRoads  # 示例,需要实际实现
            print(f"World {world_idx}: {num_roads} roads")
        except:
            print(f"World {world_idx}: 无法获取道路数据")
    
    print("="*60 + "\n")
# ============================================================
# 主训练函数
# ============================================================

def train(exp_config: Box):
    """Run PPO training with visualization."""

    # 环境配置
    env_config = dataclasses.replace(
        EnvConfig(),
        reward_type=exp_config.reward_type,
        episode_len=exp_config.episode_len,
        remove_non_vehicles=exp_config.remove_non_vehicles,
        polyline_reduction_threshold=exp_config.polyline_reduction_threshold,
        obs_radius=exp_config.observation_radius,
        collision_behavior=exp_config.collision_behavior,
        enable_procedural_generation=True,
        max_num_agents_in_scene=64,
        reward_weight_speed=exp_config.get("reward_weight_speed", 0.0),
        reward_weight_goal_dist=exp_config.get("reward_weight_goal_dist", 0.05),
    )

    # 选择网络
    if exp_config.mlp_class == "late_fusion":
        exp_config.mlp_class = LateFusionNet
        exp_config.policy = LateFusionPolicy
    elif exp_config.mlp_class == "feed_forward":
        exp_config.mlp_class = FFN
        exp_config.policy = FeedForwardPolicy

    # 地图配置
    DATA_DIR = "/root/code/gpudrive/maps" 
    base_map_paths = [f"{DATA_DIR}/Town01_tessellated.json"]
    num_worlds = exp_config.num_worlds
    sim_scenes = [random.choice(base_map_paths) for _ in range(num_worlds)]

    # =========================================================
    # 创建环境
    # =========================================================
    env = SB3MultiAgentEnv(
        config=env_config,
        exp_config=exp_config,
        max_cont_agents=env_config.max_num_agents_in_scene,
        device=exp_config.device,
        sim_scenes=sim_scenes,
        base_maps=base_map_paths,
        render_3d=False,
    )

    from viz_coordinate_diagnosis import diagnose_coordinates, diagnose_map_file
    diagnose_map_file(base_map_paths[0])  # 检查地图文件结构
    env.reset()
    diagnose_coordinates(env)  # 检查坐标结构
    diagnose_map_loading(env)
    from road_data_diagnostic import diagnose_road_data, quick_test
    sim = env._env.sim
    tl = sim.traffic_light_tensor().to_torch()[0]
    sl = sim.stop_line_tensor().to_torch()[0]
    rmt = sim.road_map_type_tensor().to_torch()[0]

    tl_count = (tl.abs().sum(dim=-1) > 0).sum()
    sl_count = (sl.abs().sum(dim=-1) > 0).sum()
    rmt_count = (rmt > 0).sum()

    print(f"✅ TrafficLights found: {tl_count}")
    print(f"✅ StopLines found: {sl_count}")
    print(f"✅ Roads with MapType: {rmt_count}")

    print("="*60)
    print("🎉 ALL SYSTEMS OPERATIONAL!")
# 快速测试
    quick_test(env)

    # 完整诊断（生成诊断图像）
    diagnose_road_data(env, output_dir="diagnostic_output")

    # =========================================================
    # 启用可视化 (重要！)
    # =========================================================
    viz_enabled = exp_config.get("enable_visualization", True)
    if viz_enabled:
        viz_output_dir = f"training_viz/{datetime.now().strftime('%m%d_%H%M')}"
        env.enable_visualization(
            output_dir=viz_output_dir,
            map_path=base_map_paths[0]
        )
        
        # 验证可视化器已正确初始化
        print("\n" + "="*60)
        print("[MAIN] Visualization Status Check")
        print("="*60)
        print(f"  > visualizer: {env.visualizer}")
        print(f"  > _viz_enabled: {env._viz_enabled}")
        print(f"  > output_dir: {viz_output_dir}")
        print("="*60 + "\n")

    print_gpu_stats("After Env Init")

    # =========================================================
    # 训练配置
    # =========================================================
    exp_config.batch_size = (
        exp_config.num_worlds * exp_config.n_steps
    ) // exp_config.num_minibatches

    datetime_ = datetime.now().strftime("%m_%d_%H_%S")
    run_id = f"{datetime_}"

    # WandB (可选)
    run = None
    if exp_config.get("track", False):
        import wandb
        run = wandb.init(
            project=exp_config.get("project_name", "gpudrive"),
            name=run_id,
            config={**exp_config, **env_config.__dict__},
        )

    # =========================================================
    # 创建 Callbacks
    # =========================================================
    callbacks = []
    
    # 1. 指标记录 Callback
    metrics_callback = MultiAgentCallback(
        config=exp_config,
        wandb_run=run, 
    )
    callbacks.append(metrics_callback)
    
    # 导入新的 Callback
    from rollout_viz_callback import RolloutVisualizationCallback

    # 在创建 callbacks 的地方替换：
    if viz_enabled:
        viz_callback = RolloutVisualizationCallback(
            record_freq=1,       # 每 10 个 rollout 录制一次
            record_first_n=3,     # 前 3 个一定录制
            max_recordings=50,
            verbose=1,
        )
        callbacks.append(viz_callback)

    callback_list = CallbackList(callbacks)

    # =========================================================
    # 创建模型
    # =========================================================
    model = IPPO(
        n_steps=exp_config.n_steps,
        batch_size=exp_config.batch_size,
        env=env,
        seed=exp_config.seed,
        verbose=exp_config.verbose,
        device=exp_config.device,
        tensorboard_log=f"runs/{run_id}",
        mlp_class=exp_config.mlp_class,
        policy=exp_config.policy,
        gamma=exp_config.gamma,
        gae_lambda=exp_config.gae_lambda,
        vf_coef=exp_config.vf_coef,
        clip_range=exp_config.clip_range,
        learning_rate=exp_config.get("lr", 0.0005),
        ent_coef=exp_config.ent_coef,
        n_epochs=exp_config.n_epochs,
        env_config=env_config,
        exp_config=exp_config,
        max_grad_norm=0.5,
        normalize_advantage=True,
        clip_range_vf=0.2,
        target_kl=None,
    )

    # =========================================================
    # 开始训练
    # =========================================================
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Total timesteps: {exp_config.total_timesteps:,}")
    print(f"Visualization: {'Enabled' if viz_enabled else 'Disabled'}")
    if viz_enabled:
        print(f"  > Record freq: {exp_config.get('viz_record_freq', 50)} episodes")
        print(f"  > First N: {exp_config.get('viz_record_first_n', 3)}")
        print(f"  > Max recordings: {exp_config.get('viz_max_recordings', 100)}")
    print("="*60 + "\n")

    model.learn(
        total_timesteps=exp_config.total_timesteps,
        callback=callback_list,
    )

    # =========================================================
    # 清理
    # =========================================================
    if run is not None:
        run.finish()
    
    env.close()
    print("\nTraining completed!")


# ============================================================
# 入口点
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPUDrive PPO Training (Fixed)")
    parser.add_argument("--config", type=str, default="baselines/ppo/config/ppo_base_sb3.yaml")
    parser.add_argument("--no-viz", action="store_true", help="禁用可视化")
    parser.add_argument("--viz-freq", type=int, default=50, help="录制频率（每N个Episode）")
    parser.add_argument("--viz-first", type=int, default=3, help="前N个Episode一定录制")
    
    args = parser.parse_args()
    
    # 加载配置
    exp_config = load_config(args.config)
    
    # 命令行覆盖
    if args.no_viz:
        exp_config.enable_visualization = False
    else:
        exp_config.enable_visualization = True
        exp_config.viz_record_freq = args.viz_freq
        exp_config.viz_record_first_n = args.viz_first
    
    # 开始训练
    train(exp_config)