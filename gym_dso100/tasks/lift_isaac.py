import numpy as np
import torch
import os

from gym_dso100.tasks.base_isaac import BaseIsaac
from isaacgym import gymapi, gymtorch


class LiftIsaac(BaseIsaac):
    """
    Isaac Gym implementation of the lift task.
    The robot needs to lift a box from the table.
    """

    def __init__(
        self,
        obs_type: str = "state",
        render_mode: str = "rgb_array",
        gripper_rotation: list = None,
        observation_width: int = 640,
        observation_height: int = 480,
        visualization_width: int = 640,
        visualization_height: int = 480,
        num_envs: int = 1,
        device: str = "cuda:0",
        **kwargs
    ):
        # Task-specific parameters
        self.lift_height_threshold = 0.1  # Height to lift the box
        self.max_episode_length = 500
        
        # Box parameters
        self.box_size = [0.05, 0.05, 0.05]  # 5cm cube
        self.box_initial_pos = [0.6, 0.0, 0.65]  # On the table
        
        # Target position for the box (lifted position)
        self.target_pos = [0.6, 0.0, 0.8]  # 15cm above table
        
        super().__init__(
            task="lift",
            obs_type=obs_type,
            render_mode=render_mode,
            gripper_rotation=gripper_rotation,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width,
            visualization_height=visualization_height,
            num_envs=num_envs,
            device=device,
            **kwargs
        )
        
        # Task-specific tensors
        self.box_positions = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.box_rotations = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.gripper_positions = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        
        # Success tracking
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lifted_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _load_object_assets(self):
        """Load box asset for lifting task."""
        # Box asset options
        asset_options = gymapi.AssetOptions()
        asset_options.density = 100.0  # Light box for easier lifting
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.armature = 0.01
        
        # Create box asset
        self.box_asset = self.gym.create_box(
            self.sim, 
            self.box_size[0], 
            self.box_size[1], 
            self.box_size[2], 
            asset_options
        )
        
        # Set box material properties
        box_props = self.gym.get_asset_rigid_shape_properties(self.box_asset)
        for p in box_props:
            p.friction = 0.8
            p.restitution = 0.1
        self.gym.set_asset_rigid_shape_properties(self.box_asset, box_props)

    def _add_objects_to_env(self, env, env_idx: int):
        """Add box to the environment."""
        # Box pose
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(
            self.box_initial_pos[0],
            self.box_initial_pos[1], 
            self.box_initial_pos[2]
        )
        box_pose.r = gymapi.Quat(0, 0, 0, 1)
        
        # Create box actor
        box_actor = self.gym.create_actor(
            env, 
            self.box_asset, 
            box_pose, 
            f"box_{env_idx}", 
            env_idx, 
            2  # Collision group
        )
        
        # Set box color (red for visibility)
        self.gym.set_rigid_body_color(
            env, box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION,
            gymapi.Vec3(0.8, 0.2, 0.2)
        )

    def _setup_observation_space(self):
        """Setup observation space for lift task."""
        if self.obs_type == "state":
            # State observations include:
            # - Robot joint positions (7 DOF)
            # - Robot joint velocities (7 DOF) 
            # - Gripper position (3D) 
            # - Gripper rotation (4D quaternion)
            # - Box position (3D)
            # - Box rotation (4D quaternion)
            # - Distance to box (1D)
            # - Box height (1D)
            obs_dim = 7 + 7 + 3 + 4 + 3 + 4 + 1 + 1  # Total: 30
            
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        else:
            super()._setup_observation_space()

    def _setup_action_space(self):
        """Setup action space for lift task."""
        # Actions: 7 joint positions + 1 gripper action
        action_dim = 8
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def _apply_actions(self, actions):
        """Apply actions to the robot."""
        # Scale actions to appropriate ranges
        joint_actions = actions[:, :7]  # First 7 for arm joints
        gripper_action = actions[:, 7:8]  # Last 1 for gripper
        
        # Apply joint position targets
        joint_targets = joint_actions * 3.14159  # Scale to +/- pi radians
        
        # Apply gripper action (open/close)
        gripper_targets = gripper_action * 0.04  # Scale to gripper range
        
        # Combine all targets
        all_targets = torch.cat([joint_targets, gripper_targets, gripper_targets], dim=1)
        
        # Set DOF position targets
        self.gym.set_dof_position_target_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(all_targets)
        )

    def _update_observations(self):
        """Update observations for lift task."""
        # Refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Extract robot state
        robot_dof_pos = self.dof_pos[:, :7]  # First 7 DOF for arm
        robot_dof_vel = self.dof_vel[:, :7]
        
        # Extract gripper position (end-effector)
        # This would typically be computed from forward kinematics
        # For now, use a placeholder
        gripper_pos = self.root_states[:, 0, :3]  # Robot base position as placeholder
        gripper_rot = self.root_states[:, 0, 3:7]  # Robot base rotation as placeholder
        
        # Extract box state
        box_pos = self.root_states[:, 1, :3]  # Box position
        box_rot = self.root_states[:, 1, 3:7]  # Box rotation
        
        # Compute additional features
        distance_to_box = torch.norm(gripper_pos - box_pos, dim=1, keepdim=True)
        box_height = box_pos[:, 2:3] - 0.63625  # Height above table
        
        # Store for reward computation
        self.box_positions = box_pos
        self.box_rotations = box_rot
        self.gripper_positions = gripper_pos
        
        if self.obs_type == "state":
            # Concatenate all observations
            self.obs_buf = torch.cat([
                robot_dof_pos,      # 7
                robot_dof_vel,      # 7
                gripper_pos,        # 3
                gripper_rot,        # 4
                box_pos,            # 3
                box_rot,            # 4
                distance_to_box,    # 1
                box_height,         # 1
            ], dim=1)
        else:
            super()._update_observations()

    def _compute_rewards(self):
        """Compute rewards for lift task."""
        # Distance reward (closer to box is better)
        distance_to_box = torch.norm(
            self.gripper_positions - self.box_positions, dim=1
        )
        distance_reward = -distance_to_box * 2.0
        
        # Height reward (lifting the box)
        box_height = self.box_positions[:, 2] - 0.63625  # Height above table
        height_reward = box_height * 10.0
        
        # Success reward (box lifted above threshold)
        success_reward = torch.where(
            box_height > self.lift_height_threshold,
            torch.tensor(100.0, device=self.device),
            torch.tensor(0.0, device=self.device)
        )
        
        # Stability reward (box not falling)
        stability_reward = torch.where(
            self.box_positions[:, 2] > 0.6,  # Box still on or above table
            torch.tensor(1.0, device=self.device),
            torch.tensor(-10.0, device=self.device)
        )
        
        # Update success tracking
        self.lifted_buf = box_height > self.lift_height_threshold
        self.success_buf = self.lifted_buf
        
        # Total reward
        self.reward_buf = distance_reward + height_reward + success_reward + stability_reward

    def _check_termination(self):
        """Check termination conditions for lift task."""
        # Reset if box falls off table
        box_fell = self.box_positions[:, 2] < 0.5
        
        # Reset if episode is too long
        time_out = self.progress_buf >= self.max_episode_length
        
        # Reset if task is completed (box lifted and held)
        task_completed = self.success_buf
        
        self.reset_buf = box_fell | time_out | task_completed
        
        # Increment progress
        self.progress_buf += 1

    def _reset_envs(self):
        """Reset environments that need resetting."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        
        if len(env_ids) > 0:
            # Reset robot to initial position
            robot_initial_dof_pos = torch.zeros((len(env_ids), 9), device=self.device)
            robot_initial_dof_pos[:, :7] = torch.tensor([0, -0.5, 0, -1.5, 0, 1.0, 0.785], device=self.device)  # Home position
            robot_initial_dof_pos[:, 7:] = 0.04  # Open gripper
            
            # Reset box position with small random variation
            box_initial_pos = torch.zeros((len(env_ids), 3), device=self.device)
            box_initial_pos[:, 0] = self.box_initial_pos[0] + torch.rand(len(env_ids), device=self.device) * 0.1 - 0.05
            box_initial_pos[:, 1] = self.box_initial_pos[1] + torch.rand(len(env_ids), device=self.device) * 0.1 - 0.05
            box_initial_pos[:, 2] = self.box_initial_pos[2]
            
            box_initial_rot = torch.zeros((len(env_ids), 4), device=self.device)
            box_initial_rot[:, 3] = 1.0  # w=1 for identity quaternion
            
            # Apply resets
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(torch.cat([robot_initial_dof_pos, torch.zeros_like(robot_initial_dof_pos)], dim=1)),
                gymtorch.unwrap_tensor(env_ids.int()),
                len(env_ids)
            )
            
            # Reset progress buffer
            self.progress_buf[env_ids] = 0
            self.success_buf[env_ids] = False
            self.lifted_buf[env_ids] = False

    def get_success_rate(self):
        """Get success rate for evaluation."""
        return torch.mean(self.success_buf.float()).item()

    def is_success(self):
        """Check if the task is successful."""
        return self.success_buf.cpu().numpy() 