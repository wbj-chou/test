from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from my_diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from my_diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from my_diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class MultiImageObsEncoder(ModuleAttrMixin):
    """
    多图像观测编码器
    - 输入: 包含多种观测的字典 (如 {'cam_front': 图像, 'cam_left': 图像, 'robot_state': 向量})
    - 输出: 一个合并后的高维特征向量 (用于决策或扩散模型)
    """

    def __init__(self,
            shape_meta: dict,                             # 包含各个观测通道的形状与类型 (rgb / low_dim)
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],  # 视觉特征提取 backbone (如 ResNet)
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,  # resize 尺寸
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,    # crop 尺寸
            random_crop: bool=True,                       # 是否随机裁剪
            use_group_norm: bool=False,                   # 是否将 BatchNorm 替换为 GroupNorm
            share_rgb_model: bool=False,                  # 是否所有相机共享同一视觉模型
            imagenet_norm: bool=False                     # 是否对输入图像做 ImageNet 归一化
        ):
        """
        假设：
        - RGB 输入形状为 [B, C, H, W]
        - 低维输入为 [B, D]
        """
        super().__init__()

        #用于保存各类型观测的key
        rgb_keys = list()
        low_dim_keys = list()

        #用于保存各通道对应的子模块
        key_model_map = nn.ModuleDict()                 #每个观测对应的视觉模型
        key_transform_map = nn.ModuleDict               #每个观测对应的预处理操作
        key_shape_map = dict()                          #每个观测对应的输入形状 

        # ============================================================
        # 处理共享视觉 backbone 的情况（例如多个相机共享一个 ResNet）
        # ============================================================

        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            # ==============================
            # 图像类输入（例如相机图像）
            # ==============================
            if type == 'rgb':
                rgb_keys.append(key)
                # 若未共享模型，为每个相机单独配置模型
                if not share_rgb_model:
                    # 用户为每个相机指定了不同模型
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        # 否则复制一份相同模型
                        assert isinstance(rgb_model, nn.Module)
                        this_model = copy.deepcopy(rgb_model)

                # 若启用 group norm，则替换模型中的所有 BatchNorm2d   
                if this_model is not None:
                        if use_group_norm:
                            this_model = replace_submodules(
                                root_module=this_model,
                                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                                func=lambda x: nn.GroupNorm(
                                    num_groups=x.num_features//16, 
                                    num_channels=x.num_features)
                            )
                        key_model_map[key] = this_model

                # ==============================
                # 配置 resize 操作
                # ==============================
                input_shape = shape
                this_resizer = nn.Indentity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize((h,w))
                    input_shape = (shape[0], h, w)
                
                # ==============================
                # 配置随机裁剪或中心裁剪
                # ============================== 
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h,w))

                # ==============================
                # 配置归一化（例如 ImageNet 预训练模型）
                # ==============================      
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    
                # ==============================
                # 将 resize、crop、normalize 组合成一个 transform pipeline
                # ==============================
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform   

            # ==============================
            # 低维输入（例如机器人关节、速度等）
            # ==============================
            elif type == 'low_dim':
                low_dim_keys.append(key)

            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        # 确保 key 顺序固定（防止字典顺序不一致）
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        # 保存配置到对象中
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # ==============================
        # 处理 RGB 图像输入
        # ==============================
        if self.share_rgb_model:
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)

            #拼接所有相机的图像 [T*B, C, H, W]
            imgs = torch.cat(imgs, dim=0)

            #提取视觉特征 [T*B, D]
            feature = self.key_model_map['rgb'](imgs)

            # 恢复为 [N, B, D] → [B, N, D] → [B, N*D]
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)

            features.append(feature)

        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
       
        # ==============================
        # 处理低维输入（状态向量）
        # ==============================
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # ==============================
        # 拼接所有特征为一个向量 [B, total_dim]
        # ==============================
        result = torch.cat(features, dim=-1)
        return result
    
    # ============================================================
    # 自动推理输出 shape（不需要真实数据）
    # ============================================================
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1

        # 构造一个伪造的输入 batch（全零张量）
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs

        # 前向传播一次，取输出 shape
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape



                  