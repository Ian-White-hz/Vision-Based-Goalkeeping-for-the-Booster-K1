# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse as quat_rotate_inverse

class SourceRewards:
    def _reward_eereach(self):

        phase1 = ((self.ball_states[:, 0] - self.env_origins[:, 0] > 1.5) & (self.ball_states[:,7] - self.ball_vel < 2.0)).nonzero(as_tuple=False).flatten()

        taskrew = torch.zeros(self.num_envs, dtype = torch.float, device = self.device)

        end_target_local = self.end_target - self.torso_pos


        asidegoal = torch.clip(end_target_local[:, 1], -1.0, 1.0)
        asidegoal[torch.abs(asidegoal) < 0.3] = 0.

        verticalgoal = torch.clip(self.torso_pos[:, 2] - torch.clip(self.end_target[:,2], 0.3, 1.2), 0.0, 1.0)

        phase1_rew = 1.0 - (verticalgoal + torch.abs(asidegoal)) / 2.0

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        jump_scale =  3.0 + 3.0 * self.curriculumupdate

        vel_sigma = taskrew.clone()
        vel_sigma[self.end_regions == 0] = 1 + 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 0, self.upper_body_index, 8], 0.0, 3.0)
        vel_sigma[self.end_regions == 1] = 1 - 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 1, self.upper_body_index, 8], -3.0, 0.0)
        vel_sigma[self.end_regions == 4] = 1 + 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 4, self.upper_body_index, 8], 0.0, 3.0)
        vel_sigma[self.end_regions == 5] = 1 - 3.0 * torch.clip(self.rigid_body_states[self.end_regions == 5, self.upper_body_index, 8], -3.0, 0.0)

        vel_sigma[self.end_regions == 2] = 1 + jump_scale * torch.clip(self.rigid_body_states[self.end_regions == 2, self.upper_body_index, 9], 0.0, 3.0)
        vel_sigma[self.end_regions == 3] = 1 + jump_scale * torch.clip(self.rigid_body_states[self.end_regions == 3, self.upper_body_index, 9], 0.0, 3.0)

        vel_sigma[behind] = 2.0

        taskrew =(1 - 1 / (1 + torch.exp(-self.curriculumsigma * (self.dist - self.cfg.rewards.reach_th))))


        taskrew *= vel_sigma

        taskrew[phase1] = phase1_rew[phase1]


        return taskrew * (1 - torch.clip(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1), 0., 1.0))

    def _reward_success(self):
        return (self.success_flag + 1.0) * (self.dist < self.cfg.rewards.strict_th)

    def _reward_stopball(self):

        changevel = (self.ball_states[:,7] - self.ball_vel > 2.0) & ((self.ball_states[:, 0] - self.env_origins[:, 0]) > 0.0)
        stopped_ids = (changevel |((self.ball_states[:, 0] - self.env_origins[:, 0]) < 0.0)).nonzero(as_tuple=False).flatten()
        success_ids = ((self.stop_flag == 0) & changevel) .nonzero(as_tuple=False).flatten()
        rew_stop = 1.0 * (self.stop_flag == 0) * changevel
        self.success_flag[success_ids] = 1.0
        self.stop_flag[stopped_ids] = 1.0

        return rew_stop

    def _reward_feetorientaion(self):

        left_quat = self.rigid_body_states[:,  self.contact_feet_indices[0], 3:7]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:,  self.contact_feet_indices[1], 3:7]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        feet_orientation = torch.sum(torch.square(left_gravity[:, :2]), dim=1) + torch.sum(torch.square(right_gravity[:, :2]), dim=1)
        return torch.exp(feet_orientation * -5)

    def _reward_airfeetorientation(self):

        foot_contact_forces_z = self.contact_forces[:, self.contact_feet_indices, 2]
        inair = torch.sum(foot_contact_forces_z, dim=-1) < 1.

        falling = (self.rigid_body_states[:, self.upper_body_index, 9] < 0.0) & inair
        jump_ids = (self.end_regions == 2) | (self.end_regions == 3)

        left_quat = self.rigid_body_states[:,  self.contact_feet_indices[0], 3:7]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.rigid_body_states[:,  self.contact_feet_indices[1], 3:7]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        feet_orientation = torch.sum(torch.square(left_gravity[:, :2]), dim=1) + torch.sum(torch.square(right_gravity[:, :2]), dim=1)

        return torch.exp(feet_orientation * -3) * falling * jump_ids

    def _reward_successland(self):

        foot_contact_forces_z = self.contact_forces[:, self.contact_feet_indices, 2]

        jump = self.root_states[:,2] > 1.0

        self.has_in_air = torch.logical_or(self.has_in_air, jump)

        has_contact = (foot_contact_forces_z[:, 0] > 1.) & (foot_contact_forces_z[:, 1] > 1.)

        one_feet_contact = (((foot_contact_forces_z[:, 0] >  1.) & (foot_contact_forces_z[:, 1] < 1.)) | ((foot_contact_forces_z[:, 0] <  1.) & (foot_contact_forces_z[:, 1] > 1.))) & (self.has_in_air)

        successful_landings = torch.logical_and(has_contact, self.has_in_air)

        air_reward = self.has_in_air.float()
        landing_reward = successful_landings.float() * 5.0

        one_feet_punish = one_feet_contact.float() * -1.0

        jump_ids = (self.end_regions == 2) | (self.end_regions == 3)

        return (air_reward + landing_reward + one_feet_punish) * jump_ids

    def _reward_feet_slippage(self):

        foot_vel = self.rigid_body_states[:,  self.contact_feet_indices, 7:10]
        contactvel = torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1) > 1.), dim=1)
        return torch.exp(contactvel * -10)

    def _reward_penalize_sharpcontact(self):

        return (torch.mean(torch.norm(self.contact_forces[:, self.contact_feet_indices, :], dim=-1), dim = -1) >  self.cfg.rewards.max_contact_force) * 1.0

    def _reward_postorientation(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -3) * behind

    def _reward_penalize_kneeheight(self):
        return (torch.min(self.rigid_body_states[:, self.knee_indices, 2], dim = -1).values < 0.15) * 1.0

    def _reward_postangvel(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -3)* behind

    def _reward_postlinvel(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)

        return torch.exp(torch.sum(torch.square(self.base_lin_vel[:, 0:1]), dim=1) * -3)* behind

    def _reward_postupperdofpos(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        mse = torch.sum(torch.square(self.dof_pos[:, self.upper_body_joint_indices] - self.default_dof_pos[:, self.upper_body_joint_indices]), dim=-1)
        reward = torch.exp(mse * -1)
        return reward * behind

    def _reward_postwaistdofpos(self):

        behind = (self.ball_states[:, 0] - self.env_origins[:, 0] < 0.0) | (self.ball_states[:,7] - self.ball_vel > 2.0)
        mse = torch.sum(torch.square(self.dof_pos[:, self.waist_joint_indices] - self.default_dof_pos[:, self.waist_joint_indices]), dim=-1)
        reward = torch.exp(-3 * mse)
        return reward * behind

    def _reward_stayonline(self):
        distance = torch.clip(torch.abs(self.torso_pos[:,0] - self.env_origins[:, 0]), 0.2, 1.2) - 0.2
        return distance

    def _reward_noretreat(self):
        return -1 * torch.clip(self.base_lin_vel[:, 0], -1.0, 0.0)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_deviation_waist_pitch_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.waist_joint_indices[2]], dim=-1)
