"""
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from climatetranslation.unit.networks import MsImageDis, VAEGen
from climatetranslation.unit.utils import weights_init, get_model_list, get_scheduler
from climatetranslation.unit import ssim
from torch.autograd import Variable
import torch
import torch.nn as nn
import os


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        self.variables = [v for _, varlist in hyperparameters['level_vars'].items() 
                          for v in varlist]
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], 
                            hyperparameters['land_mask_a'],  
                            hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], 
                            hyperparameters['land_mask_b'],  
                            hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], 
                                hyperparameters['land_mask_a'], 
                                hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], 
                                hyperparameters['land_mask_a'], 
                                hyperparameters['dis'])  # discriminator for domain b

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2),
                                        weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        self.recon_func = hyperparameters['recon_loss_func']
    
    def _get_recon_func(self, name):
        if name=='mae':
            def f(input, target):
                return torch.mean(torch.abs(input - target), dim=(0,2,3))
        elif name=='ssim':
            def f(input, target):
                window = ssim.create_window(window_size=11, channel=input.shape[1]).to(target.device)
                return - torch.mean(ssim._ssim(input, target, window, window_size=11, channel=input.shape[1]), dim=(0,2,3))
        elif name=='m4e':
            def f(input, target):
                d = 2
                return torch.mean(torch.abs(input**d - target**d), dim=(0,2,3))**(1/d)
        else:
            raise ValueError("unrecognised loss function {}".format(name))
        return f
    
    def recon_criterion(self, input, target):
        if isinstance(self.recon_func, list):
            loss = torch.zeros(len(self.recon_func))
            for i in range(len(self.recon_func)):
                loss[i] = loss[i] + self._get_recon_func(self.recon_func[i])(input[:, i:i+1], target[:, i:i+1])
        else:
            loss = self._get_recon_func(self.recon_func)(input, target)
        return loss
            

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        
        def set_recon_loss(name_root, recon_loss):
            for i,v in enumerate(self.variables):
                setattr(self, f"{name_root}-{v}", recon_loss[i])
            relative_weights = torch.tensor(hyperparameters['recon_x_multi']).to(recon_loss.device)
            relative_weights = relative_weights/relative_weights.sum()
            setattr(self, name_root, torch.sum(recon_loss*relative_weights))
        
        # reconstruction loss
        set_recon_loss('loss_gen_recon_x_a', self.recon_criterion(x_a_recon, x_a))
        set_recon_loss('loss_gen_recon_x_b',  self.recon_criterion(x_b_recon, x_b))
        set_recon_loss('loss_gen_cyc_x_a', self.recon_criterion(x_aba, x_a))
        set_recon_loss('loss_gen_cyc_x_b', self.recon_criterion(x_bab, x_b))
        # kl loss
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # total loss
        self.loss_gen_total = (
            hyperparameters['gan_w'] * self.loss_gen_adv_a + 
            hyperparameters['gan_w'] * self.loss_gen_adv_b + 
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + 
            hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + 
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + 
            hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + 
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + 
            hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + 
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + 
            hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab
        )
        self.loss_gen_total.backward()
        self.gen_opt.step()


    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_a + self.loss_dis_b)
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

