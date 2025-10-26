import torch
from typing import Dict
from einops import reduce, rearrange
from diffusers import DDPMScheduler
from torch.nn import functional as F
from my_diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from my_diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from my_diffusion_policy.policy.base_image_policy import BaseImagePolicy
from my_diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from my_diffusion_policy.model.common.normalizer import LinearNormalizer
from my_diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        #解析动作和观测的维度信息
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        #获取图像编码器的输出特征维度
        obs_feature_dim = obs_encoder.output_shape()[0]

        #构建扩散模型输入维度
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        #若观测作为全局条件，仅动作作为输入，全局条件为观测特征拼接
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim = input_dim,
            local_cond_dim = None,
            global_cond_dim = global_cond_dim,
            diffusion_step_embed_dim = diffusion_step_embed_dim,
            down_dims = down_dims,
            kernel_size = kernel_size,
            n_groups = n_groups,
            cond_predict_scale = cond_predict_scale
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim = self.action_dim,
            obs_dim = 0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps = n_obs_steps,
            fix_obs_steps = True,
            action_visible = False
        )

        self.model = model
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon  #   序列总长度
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps    #   预测的动作步数
        self.n_obs_steps = n_obs_steps  #   观测步数
        self.obs_as_global_cond = obs_as_global_cond    #   观测是否作为全局条件
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property
    def dtype(self):
        return next(self.model.parameters()).dtype
        
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        
        model = self.model
        scheduler = self.noise_scheduler

        #初始化噪声张量
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        #设置采样步数
        scheduler.set_timesteps(self.num_inference_steps)

        #逆扩散过程
        for t in scheduler.timesteps:
            #应用条件掩码：已知条件位置保持不变
            trajectory[condition_mask] = condition_data[condition_mask]

            #预测噪声
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            
            #根据噪声预测更新样本
            trajectory = scheduler.step(
                model_output, t, trajectory, generator = generator, **kwargs).prev_sample

        # 最终确保条件位置严格符合输入
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory
    
    def perdict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:必须包含obs键
        result:包含action键
        """
        assert 'past_action' not in obs_dict, "DiffusionUnetImagePolicy does not use past_action in obs_dict"
        #归一化观测数据
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        #初始化条件掩码
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            # 观测作为全局条件：图像特征拼接后作为全局条件
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs) # 形状：(B*n_obs_steps, Do)
            global_cond = nobs_features.reshape(B, -1)  # 形状：(B, Do*n_obs_steps)
            
            #初始化动作条件数据（全0，掩码全False，因为动作需预测）
            cond_data = torch.zeros(size=(B, T, Da), dtype=self.dtype, device=self.device)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        else:
            # 观测作为全局条件：图像特征拼接后作为全局条件
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs) # 形状：(B*n_obs_steps, Do)
            nobs_features = nobs_features.reshape(B, To, -1)  # 形状：(B, n_obs_steps, Do)
            cond_data = torch.zeros(size=(B, T, Da + Do), dtype=self.dtype, device=self.device)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True  # 观测部分为已知

        # 条件扩散采样生成动作序列
        nsampled = self.conditional_sample(
            cond_data = cond_data,
            cond_mask = cond_mask,
            local_cond = local_cond,
            global_cond = global_cond,
            **self.kwargs)
        
        # 提取动作部分并反归一化
        naction_pred = nsampled[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # 截取需要输出的动作片段（从观测结束位置开始）
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= 训练逻辑：计算扩散损失 =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        #归一化输入数据
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 准备扩散模型的输入和条件
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # 观测作为全局条件
            this_nobs = dict_apply(nobs, 
                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # 观测作为局部条件
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # 生成掩码
        condition_mask = self.mask_generator(trajectory.shape)

        # 随机采样扩散时间步
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=trajectory.device).long()

        #前向扩散：添加噪声到干净序列
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        #重新添加观测条件，应用条件掩码：已知条件位置替换为干净数据
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        #计算损失
        pred = self.model(noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond)
        
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unknown prediction type {pred_type}")
        
        #计算非条件位置损失
        loss_mask = ~condition_mask
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss



