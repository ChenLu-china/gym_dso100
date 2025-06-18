import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
import torch

import gymnasium as gym
from gymnasium import spaces
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import to_torch, torch_jit_compile


class BaseIsaac(gym.Env):
    """
    Superclass for all gym-dso100 Isaac Gym environments.
    Args:
        task (str): name of the task
        obs_type (str): observation type ("state", "pixels", "depth")
        render_mode (str): render mode
        gripper_rotation (list): initial rotation of the gripper (given as a quaternion)
        num_envs (int): number of parallel environments
        device (str): device to run simulation on
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        task: str,
        obs_type: str = "state",
        render_mode: str = "rgb_array",
        gripper_rotation: Optional[list] = None,
        observation_width: int = 640,
        observation_height: int = 480,
        visualization_width: int = 640,
        visualization_height: int = 480,
        num_envs: int = 1,
        device: str = "cuda:0",
        physics_engine: str = "physx",
        sim_device: str = "cuda:0",
        pipeline: str = "gpu",
        graphics_device_id: int = 0,
        headless: bool = False,
        force_render: bool = True,
    ):
        # Device setup
        self.device = device
        self.sim_device = sim_device
        self.graphics_device_id = graphics_device_id
        self.headless = headless
        self.force_render = force_render
        
        # Environment setup
        self.num_envs = num_envs
        self.task = task
        
        # Coordinates (same as Mujoco version)
        if gripper_rotation is None:
            gripper_rotation = [0, 1, 0, 0]
        self.gripper_rotation = np.array(gripper_rotation, dtype=np.float32)
        self.center_of_table = np.array([1.655, 0.3, 0.63625])
        self.max_z = 1.2
        self.min_z = 0.2
        
        # Observation
        self.obs_type = obs_type
        
        # Rendering
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        
        # Isaac Gym specific setup
        self.cfg = self._create_sim_config()
        self.physics_engine = gymapi.SIM_PHYSX if physics_engine == "physx" else gymapi.SIM_FLEX
        self.pipeline = pipeline
        
        # Initialize Isaac Gym
        self._initialize_isaac_simulation()
        
        # Setup spaces
        self._setup_observation_space()
        self._setup_action_space()
        
        # Initialize buffers
        self._setup_buffers()
        
    def _create_sim_config(self) -> gymapi.SimParams:
        """Create simulation configuration for Isaac Gym."""
        sim_params = gymapi.SimParams()
        
        # Common parameters
        sim_params.dt = 1.0 / 60.0  # 60 FPS
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        
        # PhysX parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        
        return sim_params

    def _initialize_isaac_simulation(self):
        """Initialize Isaac Gym simulation."""
        # Initialize gym
        self.gym = gymapi.acquire_gym()
        
        # Create simulation
        self.sim = self.gym.create_sim(
            self.graphics_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.cfg
        )
        
        if self.sim is None:
            raise Exception("Failed to create Isaac Gym simulation")
        
        # Create viewer if not headless
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        else:
            self.viewer = None
            
        # Setup ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        # Create environments
        self._create_environments()
        
        # Prepare simulation
        self.gym.prepare_sim(self.sim)
        
        # Setup camera
        self._setup_camera()
        
        # Initialize pytorch tensors
        self._init_pytorch_tensors()

    def _create_environments(self):
        """Create parallel environments."""
        # Load assets
        self._load_assets()
        
        # Create environments
        self.envs = []
        self.actor_handles = []
        
        env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        env_upper = gymapi.Vec3(1.0, 1.0, 2.0)
        
        for i in range(self.num_envs):
            # Create environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)
            
            # Add actors to environment
            self._add_actors_to_env(env, i)
    
    def _load_assets(self):
        """Load robot and object assets."""
        # Asset loading options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        
        # Load robot asset (you would need to convert the XML to URDF)
        robot_asset_file = f"{self.task}_robot.urdf"  # Convert from XML
        self.robot_asset = self.gym.load_asset(
            self.sim, 
            os.path.join(os.path.dirname(__file__), "assets", "urdf"),
            robot_asset_file,
            asset_options
        )
        
        # Load object assets if needed
        self._load_object_assets()
    
    def _load_object_assets(self):
        """Load object assets for the task."""
        # This would be implemented based on the specific task
        # For now, create a simple box as an example
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        
        self.box_asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.1, asset_options)
    
    def _add_actors_to_env(self, env, env_idx: int):
        """Add actors (robot, objects) to the environment."""
        # Add robot
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        
        robot_actor = self.gym.create_actor(
            env, self.robot_asset, robot_pose, f"robot_{env_idx}", env_idx, 0
        )
        
        # Store actor handle
        self.actor_handles.append(robot_actor)
        
        # Add objects if needed
        self._add_objects_to_env(env, env_idx)
    
    def _add_objects_to_env(self, env, env_idx: int):
        """Add objects to the environment."""
        # Example: add a box to lift
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(0.5, 0.0, 0.1)
        box_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        
        box_actor = self.gym.create_actor(
            env, self.box_asset, box_pose, f"box_{env_idx}", env_idx, 1
        )
    
    def _setup_camera(self):
        """Setup camera for rendering."""
        if self.viewer is not None:
            cam_pos = gymapi.Vec3(2.0, 2.0, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def _init_pytorch_tensors(self):
        """Initialize PyTorch tensors for state information."""
        # Get actor state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        
        # Wrap in pytorch tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        
        # Split into position and velocity
        self.dof_pos = self.dof_states.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.num_envs, -1, 2)[..., 1]
    
    def _setup_observation_space(self):
        """Setup observation space based on obs_type."""
        if self.obs_type == "state":
            # State-based observations (joint positions, velocities, etc.)
            obs_dim = 20  # Adjust based on your robot's DOF
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        elif self.obs_type == "pixels":
            # Image-based observations
            self.observation_space = spaces.Box(
                low=0, high=255, 
                shape=(self.observation_height, self.observation_width, 3), 
                dtype=np.uint8
            )
        elif self.obs_type == "depth":
            # Depth image observations
            self.observation_space = spaces.Box(
                low=0, high=1, 
                shape=(self.observation_height, self.observation_width, 1), 
                dtype=np.float32
            )
    
    def _setup_action_space(self):
        """Setup action space."""
        # Assuming 7 DOF arm + gripper
        action_dim = 8  # Adjust based on your robot
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
    
    def _setup_buffers(self):
        """Setup buffers for observations, rewards, etc."""
        self.obs_buf = torch.zeros(
            (self.num_envs, self.observation_space.shape[0]), 
            dtype=torch.float32, device=self.device
        )
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    
    def step(self, actions):
        """Step the simulation."""
        # Convert actions to tensor if needed
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        
        # Apply actions
        self._apply_actions(actions)
        
        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Update observations
        self._update_observations()
        
        # Compute rewards
        self._compute_rewards()
        
        # Check for resets
        self._check_termination()
        
        # Return gym interface
        obs = self.obs_buf.cpu().numpy()
        reward = self.reward_buf.cpu().numpy()
        terminated = self.reset_buf.cpu().numpy()
        truncated = np.zeros_like(terminated)
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _apply_actions(self, actions):
        """Apply actions to the robot."""
        # This would be implemented based on your robot's control interface
        # For example, position control:
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions))
        pass
    
    def _update_observations(self):
        """Update observation buffer."""
        # Refresh state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        if self.obs_type == "state":
            # Update state-based observations
            self.obs_buf[:, :] = torch.cat([
                self.dof_pos,
                self.dof_vel,
                # Add other relevant state information
            ], dim=1)
        elif self.obs_type in ["pixels", "depth"]:
            # Update image-based observations
            self._update_camera_observations()
    
    def _update_camera_observations(self):
        """Update camera-based observations."""
        # This would render camera images for each environment
        # Implementation depends on camera setup
        pass
    
    def _compute_rewards(self):
        """Compute rewards for each environment."""
        # This should be implemented based on the specific task
        # For now, return zero rewards
        self.reward_buf.fill_(0.0)
    
    def _check_termination(self):
        """Check if environments should be reset."""
        # This should be implemented based on task-specific termination conditions
        # For now, never terminate
        self.reset_buf.fill_(False)
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Reset all environments
        self.reset_buf.fill_(True)
        self._reset_envs()
        
        # Update observations
        self._update_observations()
        
        obs = self.obs_buf.cpu().numpy()
        info = {}
        
        return obs, info
    
    def _reset_envs(self):
        """Reset environments that need resetting."""
        # Reset robot positions
        # Reset object positions
        # This should be implemented based on the specific task
        pass
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        elif self.render_mode == "rgb_array":
            # Return RGB array
            return self._get_camera_image()
    
    def _get_camera_image(self):
        """Get camera image for rgb_array rendering."""
        # This would capture and return camera image
        # For now, return dummy image
        return np.zeros((self.visualization_height, self.visualization_width, 3), dtype=np.uint8)
    
    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim) 