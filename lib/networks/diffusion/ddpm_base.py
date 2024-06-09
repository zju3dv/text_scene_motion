from typing import Dict, List, Tuple
import math
import torch
import torch.nn as nn

from .make_diffuser import DIFFUSER


def make_schedule_ddpm(timesteps: int, beta: List, beta_schedule: str, s=0.008) -> Dict:
    assert beta[0] < beta[1] < 1.0
    if beta_schedule == 'linear':
        betas = torch.linspace(beta[0], beta[1], timesteps)
    elif beta_schedule == 'cosine':
        x = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    elif beta_schedule == 'sqrt':
        betas = torch.sqrt(torch.linspace(beta[0], beta[1], timesteps))
    else:
        raise Exception('Unsupport beta schedule.')

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])    
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    params_list =  {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
        'log_one_minus_alphas_cumprod': torch.log(1 - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': torch.sqrt(1 / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': torch.sqrt(1 / alphas_cumprod - 1),
        'posterior_variance': posterior_variance,
        'posterior_log_variance_clipped': torch.log(posterior_variance.clamp(min=1e-20)),
        'posterior_mean_coef1': betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
        'posterior_mean_coef2': (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
    }
    params_list = {k: v.float() for k, v in params_list.items()}
    return params_list


@DIFFUSER.register()
class ConditionalDDPM(nn.Module):
    def __init__(self, eps_model: nn.Module, cfg) -> None:
        super(ConditionalDDPM, self).__init__()
        self.eps_model = eps_model
        self._set_cfg(cfg)
        
        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)

        self.optimizer = None
        self.planner = None
    
    @property
    def device(self):
        return self.betas.device
    
    def _set_cfg(self, cfg):
        self.timesteps = cfg.timesteps
        self.schedule_cfg = cfg.schedule_cfg
        self.pred_type = cfg.pred_type
        self.return_steps = False
        self.sample_steps = cfg.get('sample_steps', self.timesteps)
        assert self.sample_steps <= self.timesteps
        self.ddim_sample = self.sample_steps < self.timesteps
        self.ddim_sample_eta = cfg.get('ddim_sample_eta', 0.)
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def set_planner(self, planner):
        self.planner = planner
    
    def forward(self, data):
        """ Reverse diffusion process, sampling with the given data containing condition
        Args:
            data: test data, data['x'] gives the target data, data['y'] gives the condition
        """
        ts, noise = self.rand_ts_noise(data)
        # calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        # predict sth
        output = self.eps_model(x_t, ts, data)
        # get target
        target = self.get_target(noise, data['x'])
        return target, output

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Forward difussion process
        B, *x_shape = x0.shape
        x_t = self.sqrt_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x0 + \
              self.sqrt_one_minus_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * noise
        return x_t
    
    def rand_ts_noise(self, data):
        B = data['x'].shape[0]
        ## randomly sample timesteps
        ts = torch.randint(0, self.timesteps, (B, ), device=self.device).long()
        ## generate Gaussian noise
        noise = torch.randn_like(data['x'], device=self.device)
        return ts, noise
    
    def get_target(self, noise, x0):
        if self.pred_type == 'pred_noise':
            target = noise
        elif self.pred_type == 'pred_x0':
            target = x0
        else:
            raise Exception('Unsupported pred_type.')
        return target

    @torch.no_grad()
    def sample(self, data, k: int=1, init_x=None) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        Args:
            data: test data, data['x'] gives the target data shape
            k: the number of sampled data
        Return:
            Sampled results, the shape is <B, k, T, ...>
        """
        ksamples = []
        for _ in range(k):
            if self.ddim_sample:
                ksamples.append(self.ddim_sample_loop(data, init_x=init_x))
            else:
                ksamples.append(self.p_sample_loop(data, init_x=init_x))
        return torch.stack(ksamples, dim=1)
    
    @torch.no_grad()
    def ddim_sample_loop(self, data, init_x=None):
        B = data['x'].shape[0]
        
        times = torch.linspace(-1, self.timesteps-1, steps=self.sample_steps+1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        ## precompute conditional feature, which will be used in every sampling step
        # self.eps_cond_forward(data)
        # x_t = torch.rand_like(data['x'])
        
        x_t = torch.randn_like(data['x']) if init_x is None else init_x
        all_x_t = [x_t]
        x_0 = None
        
        for time, time_next in time_pairs:
            ts = torch.full((B,), time, device=self.device).long()
            pred_noise, x_0 = self.model_predict(x_t, ts, data, clip_x0=True)
            
            if time_next < 0:
                x_t = x_0
                all_x_t.append(x_t)
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            
            sigma = self.ddim_sample_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(data['x'])
            
            x_t = x_0 * alpha_next.sqrt() + \
                  c * pred_noise + sigma * noise
            all_x_t.append(x_t)
        
        ret = torch.stack(all_x_t, dim=1) if self.return_steps else x_t
        return ret
        
    
    @torch.no_grad()
    def p_sample_loop(self, data, init_x=None) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling
        Args:
            data: test data, data['x'] gives the target data shape
        Return:
            Sampled data, <B, T, ...>
        """
        # x_t = torch.randn_like(data['x'], device=self.device)
        x_t = torch.randn_like(data['x'], device=self.device) if init_x is None else init_x
        ## iteratively sampling
        all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data)
            all_x_t.append(x_t)
        
        ret = torch.stack(all_x_t, dim=1) if self.return_steps else x_t
        return ret
    
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data, init_x=None) -> torch.Tensor:
        """ One step of reverse diffusion process
        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            data: data dict that provides original data and computed conditional feature
        Return:
            Predict data in the previous step, i.e., $x_{t-1}$
        """
        B, *_ = x_t.shape
        batch_timestep = torch.full((B, ), t, device=self.device, dtype=torch.long)

        model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, data, clip_denoised=True, init_x=init_x)
        
        noise = torch.randn_like(x_t) if t > 0 else 0. # no noise if t == 0

        ## sampling with mean updated by optimizer and planner
        if self.optimizer is not None:
            ## openai guided diffusion uses the input x to compute gradient, see
            ## https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L436
            ## But the original formular uses the computed mean?
            gradient = self.optimizer.gradient(model_mean, data, model_variance)
            model_mean = model_mean + gradient
        if self.planner is not None:
            gradient = self.planner.gradient(model_mean, data, model_variance)
            model_mean = model_mean + gradient

        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_x
    
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, data, clip_denoised=True, init_x=None):
        """ Calculate the mean and variance, we adopt the following first equation.
        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        Return:
            (model_mean, posterior_variance, posterior_log_variance)
        """
        B, *x_shape = x_t.shape

        ## predict noise and x0 with model $p_\theta$
        pred_noise, pred_x0 = self.model_predict(x_t, t, data)
        
        # inpainting
        if init_x is not None:
            nf = data['inpaint_nframes']
            inpainting_mask = torch.ones_like(pred_x0)
            inpainting_mask[:, :nf, :] = torch.arange(0, 1, 1/nf, device=self.device).unsqueeze(0).unsqueeze(-1)
            pred_x0 = pred_x0 * inpainting_mask + init_x * (1 - inpainting_mask)
        
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)

        ## calculate mean and variance
        model_mean = self.posterior_mean_coef1[t].reshape(B, *((1, ) * len(x_shape))) * pred_x0 + \
                     self.posterior_mean_coef2[t].reshape(B, *((1, ) * len(x_shape))) * x_t
        posterior_variance = self.posterior_variance[t].reshape(B, *((1, ) * len(x_shape)))
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(B, *((1, ) * len(x_shape))) # clipped variance

        return model_mean, posterior_variance, posterior_log_variance
    
    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, data, clip_x0=False, redervie_noise=False):
        """ Get and process model prediction
        Args:
            x_t: denoised sample at timestep t
            t: denoising timestep
            cond: condition tensor
        Return:
            The predict target `(pred_noise, pred_x0)`, currently we predict the noise, which is as same as DDPM
        """
        pred = self.eps_model(x_t, t, data)
        return self.get_pred_noise_x0(pred, x_t, t, clip_x0, redervie_noise)
    
    def get_pred_noise_x0(self, pred, x_t, t, clip_x0=False, redervie_noise=False):
        B, *x_shape = x_t.shape
        if self.pred_type == 'pred_noise':
            pred_noise = pred
            pred_x0 = self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - \
                    self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * pred_noise
            if clip_x0:
                pred_x0.clamp_(-1., 1.)
                if redervie_noise:
                    pred_noise = (self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - pred_x0) / \
                        self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape)))
        elif self.pred_type == 'pred_x0':
            pred_x0 = pred
            if clip_x0:
                pred_x0.clamp_(-1., 1.)
            pred_noise = (self.sqrt_recip_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape))) * x_t - pred_x0) / \
                        self.sqrt_recipm1_alphas_cumprod[t].reshape(B, *((1, ) * len(x_shape)))
        else:
            Exception('Unsupported pred type.')
        return pred_noise, pred_x0


@DIFFUSER.register()
class ObserConditionalDDPM(ConditionalDDPM):
    def __init__(self, eps_model, cfg) -> None:
        super().__init__(eps_model, cfg)
        self.obser = cfg.get('obser', False)
        self.obser_type = cfg.get('obser_type', 'traj')
    
    def apply_observation(self, x_t, data, init_x=None):
        if not self.obser:
            return x_t
        if self.obser_type == 'traj':
            x_t[..., :9] = data['traj'].clone()
        elif self.obser_type == 'start_traj':
            x_t[..., [0], :] = data['start_pose'][:, :, :9].clone()
        elif self.obser_type == 'start_motion':
            x_t[..., [0], :] = data['start_pose'].clone()
        return x_t

    
    def forward(self, data) -> torch.Tensor:
        """ Reverse diffusion process, sampling with the given data containing condition
        """
        ts, noise = self.rand_ts_noise(data)

        ## calculate x_t, forward diffusion process
        x_t = self.q_sample(x0=data['x'], t=ts, noise=noise)
        ## apply observation before forwarding to eps model
        ## model need to learn the relationship between the observation frames and the rest frames
        x_t = self.apply_observation(x_t, data)
        ## predict sth
        output = self.eps_model(x_t, ts, data)
        ## apply observation after forwarding to eps model
        ## this operation will detach gradient from the loss of the observation tokens
        ## because the target and output of the observation tokens all are constants
        target = self.get_target(noise, data['x'])
        return target, output

    
    @torch.no_grad()
    def p_sample_loop(self, data, init_x=None) -> torch.Tensor:
        """ Reverse diffusion process loop, iteratively sampling
        """
        # x_t = torch.randn_like(data['x'], device=self.device)
        x_t = torch.randn_like(data['x'], device=self.device) if init_x is None else init_x
        ## apply observation to x_t
        x_t = self.apply_observation(x_t, data, init_x=init_x)
        ## iteratively sampling
        # all_x_t = [x_t]
        for t in reversed(range(0, self.timesteps)):
            x_t = self.p_sample(x_t, t, data, init_x=init_x)
            ## apply observation to x_t
            x_t = self.apply_observation(x_t, data)
            # all_x_t.append(x_t)
            
        return x_t