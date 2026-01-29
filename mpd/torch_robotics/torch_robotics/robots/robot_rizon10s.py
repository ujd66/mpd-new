import os

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robot_tree import convert_link_dict_to_tensor
from torch_robotics.torch_kinematics_tree.utils.files import get_configs_path, get_robot_path
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame
from torch_robotics.environments.primitives import MultiSphereField


class RobotRizon10s(RobotBase):

    link_name_ee = "flange"  # must be in the urdf file

    def __init__(self, gripper=False, grasped_object=None, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):

        urdf_robot_file = os.path.join(get_robot_path(), "rizon10s_description", "flexiv_Rizon10s_kinematics.urdf")

        ##########################################################################################
        super().__init__(
            urdf_robot_file=urdf_robot_file,
            collision_spheres_file_path=os.path.join(get_configs_path(), "rizon10s/rizon10s_sphere_config.yaml"),
            joint_limits_file_path=os.path.join(get_configs_path(), "rizon10s/joint_limits.yaml"),
            link_name_ee=self.link_name_ee,
            gripper_q_dim=0,  # no gripper for Rizon10s
            grasped_object=grasped_object,
            tensor_args=tensor_args,
            **kwargs,
        )

    def render(
        self,
        ax,
        q=None,
        color="blue",
        arrow_length=0.15,
        arrow_alpha=1.0,
        arrow_linewidth=2.0,
        draw_links_spheres=False,
        **kwargs,
    ):
        # Draw skeleton using torchkin
        if q is not None:
            # Forward kinematics
            fks_dict = self.robot_torchkin.get_link_poses(q.unsqueeze(0) if q.ndim == 1 else q)

            # Draw link_collision spheres if requested
            if draw_links_spheres:
                link_tensor = convert_link_dict_to_tensor(fks_dict, self.link_collision_spheres_names)
                link_pos = link_pos_from_link_tensor(link_tensor)
                spheres = MultiSphereField(
                    link_pos.squeeze(0), self.link_collision_spheres_radii.view(-1, 1), tensor_args=self.tensor_args
                )
                spheres.render(ax, color="red", cmap="Reds", **kwargs)

            # Draw EE frame
            if self.link_name_ee in fks_dict:
                frame_EE = fks_dict[self.link_name_ee]
                if frame_EE.ndim == 3:
                    frame_EE = frame_EE[0]
                plot_coordinate_frame(
                    ax,
                    frame_EE,
                    tensor_args=self.tensor_args,
                    arrow_length=arrow_length,
                    arrow_alpha=arrow_alpha,
                    arrow_linewidth=arrow_linewidth,
                )

    def render_trajectories(self, ax, q_pos_trajs=None, start_state=None, goal_state=None, colors=["gray"], **kwargs):
        if q_pos_trajs is not None:
            trajs_pos = self.get_position(q_pos_trajs)
            for traj, color in zip(trajs_pos, colors):
                for t in range(traj.shape[0]):
                    q = traj[t]
                    self.render(ax, q, color, **kwargs, arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.0)
            if start_state is not None:
                self.render(ax, start_state, color="green")
            if goal_state is not None:
                self.render(ax, goal_state, color="purple")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    robot = RobotRizon10s(tensor_args=DEFAULT_TENSOR_ARGS)
    print(f"Robot: {robot.name}")
    print(f"DOF: {robot.q_dim}")
    print(f"End-effector link: {robot.link_name_ee}")
    print(f"Joint limits min: {robot.q_pos_min}")
    print(f"Joint limits max: {robot.q_pos_max}")
