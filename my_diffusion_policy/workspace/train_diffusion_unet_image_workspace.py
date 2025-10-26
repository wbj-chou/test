import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import hydra
import copy
import wandb
import os
import tqdm
import pathlib
from omegaconf import OmegaConf
from my_diffusion_policy.workspace.base_workspace import BaseWorkspace
from my_diffusion_policy.dataset.base_dataset import BaseImageDataset
from my_diffusion_policy.model.common.lr_scheduler import get_scheduler
from my_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from my_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
#from my_diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from my_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from my_diffusion_policy.common.json_logger import JsonLogger
from my_diffusion_policy.model.diffusion.ema_model import EMAModel



class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    #定义需要保存的训练状态
    include_keys = ['global_step', 'epoch', 'optimizer', 'model', 'ema_model']

    def __init__(self, cfg, output_dir = None):
        super().__init__(cfg, output_dir)

        #初始化随机种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        #实例化Diffusion model图像策略模型
        self.model:DiffusionUnetImagePolicy = hydra.utils.instantiate(
            cfg.policy,shape_meta=cfg.shape_meta)
        
        #初始化EMA模型
        self.ema_model:DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        #创建优化器
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,params=self.model.parameters())
        
        #训练状态变量
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        #从checkpoint恢复训练状态
        if cfg.training.resume:
            latest_checkpoint_path = self.get_checkpoint_path()
            if latest_checkpoint_path.is_file():
                self.load_checkpoint(latest_checkpoint_path)
                print(f"从checkpoint恢复训练: {latest_checkpoint_path}")

        #初始化图像任务数据集
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        
        #数据加载
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        
        #获取数据归一化器
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        #验证数据集
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        #配置学习率调度器
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps = cfg.training.lr_warmup_steps,
            num_training_steps = (len(train_dataloader)*cfg.training.num_epochs) \
                // cfg.training.gradient_accumulation_every,
            last_epoch = self.global_step -1
        )

        #配置EMA更新策略
        ema:EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model = self.ema_model
            )

        #初始化PushT任务环境运行器
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)

        #初始化wandb日志记录
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})

        # 配置checkpoint管理器（保存TopK性能模型）
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        #设备配置
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if cfg.training.use_ema:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        #保存一个训练批次用于采样可视化
        train_sampling_batch = None

        #调试模式
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        #================训练主循环==================
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                #训练一个epoch
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                #计算模型损失
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc = f"training_epoch {self.epoch}", leave=False, 
                               mininterval = cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        #数据转移到设备
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        #计算损失
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulation_every
                        loss.backward()

                        #梯度累积更新
                        if self.global_step % cfg.training.gradient_accumulation_every == 0:
                            #优化器步进
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                            #EMA更新
                            if cfg.training.use_ema:
                                ema.step(self.model)

                            #日志记录
                            raw_loss_cpu = raw_loss.item()
                            tepoch.set_postfix(loss = raw_loss_cpu, refresh =False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train/loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }

                            is_last_batch = (batch_idx == len(train_dataloader)-1)
                            #日志记录（跳过最后一个batch，与验证日志合并）
                            if not is_last_batch:                      
                                wandb_run.log(step_log, step = self.global_step)
                                json_logger.log(step_log)
                                self.global_step += 1

                            # 调试模式：提前结束训练步骤
                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >=(cfg.training.max_train_steps -1):
                                break

                        #记录损失
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                #============验证和评估模型=============
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # #执行环境rollout评估
                # if self.epoch % cfg.training.rollout_every ==0:
                #     runner_log = env_runner.run(policy)
                #     #记录所有指标
                #     step_log.update(runner_log)

                #验证集损失计算
                if(self.epoch % cfg.training.val_every ==0):
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader,desc = f"Validation epoch {self.epoch}", leave=False,
                                      mininterval = cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                #数据转移到设备
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                                #计算损失
                                loss = policy.compute_loss(batch)
                                val_losses.append(loss.item())

                                #调试模式：提前结束验证步骤
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >=(cfg.training.max_val_steps -1):
                                    break
                        
                        if len(val_losses) >0:
                            val_loss = np.mean(val_losses)
                            step_log['val_loss'] = val_loss

                #训练集采样可视化（定期执行）
                if (self.epoch % cfg.training.sample_every ==0):
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True))
                        
                        obs_dict = batch['obs']
                        gt_action = batch['action']

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                #保存checkpoint（定期执行）
                if (self.epoch % cfg.training.checkpoint_every ==0):
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                        print(f"保存TopK checkpoint到: {topk_ckpt_path}")

                #结束验证
                policy.train()

                #结束本次epoch
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base = None,
    config_path = str(pathlib.Path(__file__).parent.parent.joinpath('config')),
    config_name = pathlib.Path(__file__).stem
)

def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()


