import networks
from .utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler,clipPatch,generateMaskPatch
from torch.autograd import Variable
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import os

class myNet_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(myNet_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.style_dim = hyperparameters['gen']['style_dim']
        self.enc_c_a = networks.define_VGG_Content()
        self.enc_c_b = networks.define_VGG_Content()
        self.enc_s_a = networks.define_E(input_nc=3, output_nc=self.style_dim, ndf=64)        # encoder for domain a
        self.enc_s_b = networks.define_E(input_nc=3, output_nc=self.style_dim, ndf=64)        # encoder for domain b
        self.gen_a = networks.define_G(input_nc=3, output_nc=3, nz=self.style_dim, ngf=64)  # generator for domain a
        self.gen_b = networks.define_G(input_nc=3, output_nc=3, nz=self.style_dim, ngf=64)  # generator for domain b
        self.dis_a = networks.define_D(input_nc=3, ndf=64,norm='instance', num_Ds=2)     # discriminator for domain a
        self.dis_b = networks.define_D(input_nc=3, ndf=64,norm='instance', num_Ds=2)     # discriminator for domain b
        self.netVGGF = networks.define_VGGF()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)



        # Initiate the criterions or loss functions
        self.criterionGAN = networks.GANLoss(mse_loss=True, tensor=torch.cuda.FloatTensor)                                           # criterion GAN adversarial loss
        self.wGANloss = networks.wGANLoss(tensor=torch.cuda.FloatTensor)                                                             # wGAN adversarial loss
        self.criterionL1 = torch.nn.L1Loss()                                                                              # L1 loss
        self.criterionL2 = networks.L2Loss()                                                                              # L2 loss
        self.criterionZ = torch.nn.L1Loss()                                                                               # L1 loss between code
        self.criterionC = networks.ContentLoss(vgg_features=self.netVGGF)           # content loss
        self.criterionS = networks.StyleLoss(vgg_features=self.netVGGF)               # style loss
        self.criterionC_l = networks.ContentLoss(vgg_features=self.netVGGF)         # local content loss
        self.criterionS_l = networks.StyleLoss(vgg_features=self.netVGGF)             # local style loss
        self.criterionHisogram = networks.HistogramLoss(vgg_features=self.netVGGF)# histogram loss
        self.Feature_map_im = networks.Feature_map_im(vgg_features=self.netVGGF)      # show feature map

        # fix the noise used in sampling
        self.s_a = torch.randn(8, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(8, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)



        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def name(self):
        return 'myNet'

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def set_input(self, x_a, x_b, hyperparameters):
        if  not hyperparameters['whether_mask']:
            self.x_a = x_a
            self.x_b = x_b
        else:
            self.x_b = x_b[:,:,:,0:256]
            self.x_b_patch = x_b[:,:,:,256:512]
            self.x_b_mask_patch = self.x_b
            self.x_a = x_a[:, :, :, 0:256]
            self.x_a_mask = x_a[:, :, :, 256:512]
            self.x_a_mask_patch = generateMaskPatch(self.x_a_mask,self.x_b_patch)

    def forward(self):
        self.eval()
        # random_code
        # s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode input image
        self.c_a = self.enc_c_a(self.x_a)
        self.c_b = self.enc_c_b(self.x_b)
        self.s_a = self.enc_s_a(self.x_a)
        self.s_b = self.enc_s_a(self.x_b)
        # generate image
        self.x_ab = self.gen_a(self.c_b, self.s_a)
        self.x_ba = self.gen_b(self.c_a, self.s_b)
        self.train()

    def gen_update(self, hyperparameters):
        x_a=self.x_a
        x_b=self.x_b
        self.gen_opt.zero_grad()

        # parameter of discriminator are not updated
        self.set_requires_grad(self.dis_a, False)
        self.set_requires_grad(self.dis_b, False)
        s_a_random = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b_random = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # encode
        c_a_prime = self.enc_s_a(x_a)
        c_b_prime = self.enc_s_b(x_b)
        s_a_prime = self.enc_s_a(x_a)
        s_b_prime = self.enc_s_a(x_b)

        # decode (cross domain)
        x_ab = self.gen_a(c_b_prime, s_a_prime)
        x_ba = self.gen_b(c_a_prime, s_b_prime)
        self.x_ab = x_ab
        self.x_ba = x_ba

        # decode (within domain)
        x_a_recon = self.gen_a(c_a_prime, s_a_prime)
        x_b_recon = self.gen_b(c_b_prime, s_b_prime)
        self.x_a_recon = x_a_recon
        self.x_b_recon = x_b_recon

        # random
        x_ba_random = self.gen_a(c_b_prime, s_a_random)
        x_ab_random = self.gen_b(c_a_prime, s_b_random)

        # cross domain encode again
        c_a_recon = self.enc_c_a(x_ab)
        c_b_recon = self.enc_c_b(x_ba)
        s_a_recon = self.enc_s_a(x_ab)
        s_b_recon = self.enc_s_b(x_ba)

        # in domain encode again
        c_a_x_recon = self.enc_c_a(x_a_recon)
        c_b_x_recon = self.enc_c_b(x_b_recon)
        s_a_x_recon = self.enc_s_a(x_a_recon)
        s_b_x_recon = self.enc_s_b(x_b_recon)

        # decode again (if needed)
        x_aba = self.gen_a.decode(x_ba, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_a.decode(x_ab, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_x_recon_c_a = self.recon_criterion(c_a_x_recon, c_a_prime) if hyperparameters['recon_x_c_w'] > 0 else 0
        self.loss_gen_x_recon_c_b = self.recon_criterion(c_b_x_recon, c_b_prime) if hyperparameters['recon_x_c_w'] > 0 else 0
        self.loss_gen_x_recon_s_a = self.recon_criterion(s_a_x_recon, s_a_prime)   if hyperparameters['recon_x_s_w'] > 0 else 0
        self.loss_gen_x_recon_s_b = self.recon_criterion(s_b_x_recon, s_b_prime)  if hyperparameters['recon_x_s_w'] > 0 else 0
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a_prime) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b_prime) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a_prime)  if hyperparameters['recon_c_w'] > 0 else 0
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b_prime) if hyperparameters['recon_c_w'] > 0 else 0
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # GAN loss
        pred_fake_a = self.dis_a.forward(x_ab)
        pred_fake_b = self.dis_b.forward(x_ba)
        self.loss_gen_adv_a, _ = self.criterionGAN(pred_fake_a, True)
        self.loss_gen_adv_b, _ = self.criterionGAN(pred_fake_b, True)

        # content loss
        self.loss_gen_content_a = self.criterionC(x_ab, x_a) if hyperparameters['content_w'] > 0 else 0
        self.loss_gen_content_b = self.criterionC(x_ba, x_b) if hyperparameters['content_w'] > 0 else 0


        # style loss
        self.loss_gen_style_a = self.criterionS(x_ab, x_a) if hyperparameters['style_w'] > 0 else 0
        self.loss_gen_style_b = self.criterionS(x_ba, x_b) if hyperparameters['style_w'] > 0 else 0



        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['content_w'] * self.loss_gen_content_a + \
                              hyperparameters['content_w'] * self.loss_gen_content_b + \
                              hyperparameters['style_w'] * self.loss_gen_style_a + \
                              hyperparameters['style_w'] * self.loss_gen_style_b + \
                              hyperparameters['recon_x_s_w'] * self.loss_gen_x_recon_s_a + \
                              hyperparameters['recon_x_s_w'] * self.loss_gen_x_recon_s_b + \
                              hyperparameters['recon_x_c_w'] * self. loss_gen_x_recon_c_a + \
                              hyperparameters['recon_x_c_w'] * self.loss_gen_x_recon_c_b
            #hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              #hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b
                              #hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              #hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()
        '''
        print('loss_gen_adv_a:', hyperparameters['gan_w'] * self.loss_gen_adv_a.data.cpu().numpy())
        print('loss_gen_adv_b:', hyperparameters['gan_w'] * self.loss_gen_adv_b.data.cpu().numpy())
        print('loss_gen_recon_x_a:', hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a.data.cpu().numpy())
        print('loss_gen_recon_x_b:', hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b.data.cpu().numpy())
        print('loss_gen_recon_s_a:', hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a.data.cpu().numpy())
        print('loss_gen_recon_s_b:', hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b.data.cpu().numpy())
        print('loss_total:',self.loss_gen_total.data.cpu().numpy())
        '''

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, im_a, im_b,hyperparameters):
        if hyperparameters['whether_mask']:
            # initial data
            x_b = im_b[:, :, :, 0:256]
            x_b_patch = im_b[:, :, :, 256:512]
            x_b_mask_patch = x_b
            x_a = im_a[:, :, :, 0:256]
            x_a_mask = im_a[:, :, :, 256:512]
            x_a_mask_patch = generateMaskPatch(x_a_mask, x_b_patch)
            # test
            self.eval()
            x_a.volatile = True
            x_b.volatile = True
            s_a1 = Variable(self.s_a, volatile=True)
            s_b1 = Variable(self.s_b, volatile=True)
            s_a2_rand = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(), volatile=True)
            s_b2_rand = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(), volatile=True)
            x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
            for i in range(x_a.size(0)):
                s_a_fake = self.enc_a(x_a[i].unsqueeze(0))
                s_b_fake = self.enc_b(x_b[i].unsqueeze(0))
                x_a_recon.append(self.gen_a(x_a[i].unsqueeze(0), s_a_fake.unsqueeze(2).unsqueeze(3)))
                x_b_recon.append(self.gen_b(torch.cat((x_b[i].unsqueeze(0), x_b_mask_patch[i].unsqueeze(0)), 1),
                                            s_b_fake.unsqueeze(2).unsqueeze(3)))
                x_ba1.append(self.gen_b(torch.cat((x_a[i].unsqueeze(0), x_a_mask_patch[i].unsqueeze(0)), 1),
                                        s_b_fake.unsqueeze(2).unsqueeze(3)))
                x_ba2.append(self.gen_b(torch.cat((x_a[i].unsqueeze(0), x_a_mask_patch[i].unsqueeze(0)), 1),
                                        s_b2_rand[i].unsqueeze(0)))
                x_ab1.append(self.gen_a(x_b[i].unsqueeze(0), s_a_fake.unsqueeze(2).unsqueeze(3)))
                x_ab2.append(self.gen_a(x_b[i].unsqueeze(0), s_a2_rand[i].unsqueeze(0)))
        else:
            self.eval()
            x_a = im_a
            x_b = im_b
            x_a.volatile = True
            x_b.volatile = True
            x_b_patch = clipPatch(x_b, hyperparameters['new_size'], hyperparameters['patch_size'])
            s_a1 = Variable(self.s_a, volatile=True)
            s_b1 = Variable(self.s_b, volatile=True)
            s_a2_rand = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(), volatile=True)
            s_b2_rand = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(), volatile=True)
            x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
            for i in range(x_a.size(0)):
                s_a_fake = self.enc_a(x_a[i].unsqueeze(0))
                s_b_fake = self.enc_b(x_b[i].unsqueeze(0))
                x_ba1.append(self.gen_a(x_a[i].unsqueeze(0), s_b_fake.unsqueeze(2).unsqueeze(3)))
                x_ba2.append(self.gen_a(x_a[i].unsqueeze(0), s_b_fake.unsqueeze(2).unsqueeze(3)))
                x_ab1.append(self.gen_b(x_b[i].unsqueeze(0), s_a_fake.unsqueeze(2).unsqueeze(3)))
                x_ab2.append(self.gen_b(x_b[i].unsqueeze(0), s_a_fake.unsqueeze(2).unsqueeze(3)))
                x_a_recon.append(self.gen_a(x_ba1[i], s_a_fake.unsqueeze(2).unsqueeze(3)))
                x_b_recon.append(self.gen_b(x_ab1[i], s_b_fake.unsqueeze(2).unsqueeze(3)))

        # aggragate data
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, hyperparameters):
        x_a = self.x_a
        x_b = self.x_b
        if hyperparameters['whether_mask']:
            x_a_patch = self.x_a_mask_patch

        self.dis_opt.zero_grad()
        self.set_requires_grad(self.dis_a, True)
        self.set_requires_grad(self.dis_b, True)
        # encode
        c_a = self.enc_c_a(x_a)
        c_b = self.enc_c_b(x_b)
        s_a = self.enc_s_a(x_a)
        s_b = self.enc_s_b(x_b)
        # decode (cross domain)
        x_ab = self.gen_a(c_b, s_a)
        x_ba = self.gen_b(c_a, s_b)
        #x_ba = self.gen_b(torch.cat((x_a_patch,x_a),1), s_b)
        # save x_ab and x_ba
        self.x_ab = x_ab
        self.x_ba = x_ba


        # loss cause from generator A
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake_a = self.dis_a.forward(x_ab.detach())
        # Real
        pred_real_a = self.dis_a.forward(self.x_a)
        loss_dis_a_fake, _ = self.criterionGAN(pred_fake_a, False)
        loss_dis_a_real, _ = self.criterionGAN(pred_real_a, True)
        self.loss_dis_a = loss_dis_a_fake + loss_dis_a_real


        # loss cause from generator B
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake_b = self.dis_b.forward(x_ba.detach())
        # Real
        pred_real_b = self.dis_b.forward(self.x_b)
        loss_dis_b_fake, _ = self.criterionGAN(pred_fake_b, False)
        loss_dis_b_real, _ = self.criterionGAN(pred_real_b, True)
        self.loss_dis_b = loss_dis_b_fake + loss_dis_b_real

        # step the optimizer
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()
        '''
        print("loss d_a:", loss_dis_a.data.cpu().numpy())
        print("loss d_b:", loss_dis_b.data.cpu().numpy())
        print("total lossD:", self.loss_dis_total.data.cpu().numpy())
        '''

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
        enc_content_name = os.path.join(snapshot_dir, 'enc_content_%08d.pt' % (iterations + 1))
        enc_style_name = os.path.join(snapshot_dir, 'enc_style_%08d.pt' % (iterations + 1))
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.enc_c_a.state_dict(), 'b': self.enc_c_b.state_dict()}, enc_content_name)
        torch.save({'a': self.enc_s_a.state_dict(), 'b': self.enc_s_b.state_dict()}, enc_style_name)
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
    def load_model_dict(self,opts):
        path_E = opts.E_path
        path_G = opts.G_path
        state_dict_E = torch.load(path_E)
        state_dict_G = torch.load(path_G)
        # need discriminator when training
        if opts.phase=="train":
            path_D = opts.D_path
            state_dict_D = torch.load(path_D)
            self.dis_a.load_state_dict(state_dict_D['a'])
            self.dis_b.load_state_dict(state_dict_D['b'])
        # load net by path
        self.enc_a.load_state_dict(state_dict_E['a'])
        self.enc_b.load_state_dict(state_dict_E['b'])
        self.gen_a.load_state_dict(state_dict_G['a'])
        self.gen_b.load_state_dict(state_dict_G['b'])

    def get_current_visuals(self):
        contour_im = tensor2im(self.x_a.data)
        gt_im = tensor2im(self.x_b.data)
        gt_contour_style = tensor2im(self.x_ab.detach())
        contour_gt_style = tensor2im(self.x_ba.detach())
        contour_rec = tensor2im(self.x_a_recon.detach())
        gt_rec = tensor2im(self.x_b_recon.detach())

        ret_dict = OrderedDict([('x_a:contour_im', contour_im),
                                ('x_b:gt_im', gt_im),
                                ('x_ab:gt_contour_style', gt_contour_style),
                                ('x_ba:contour_gt_style', contour_gt_style),
                                ('x_a_rec:contour_rec', contour_rec),
                                ('x_b_rec:gt_rec', gt_rec)])
        return ret_dict

    def get_current_errors(self,hyper):
        ret_dict = OrderedDict([('loss_dis_a', self.loss_dis_a.data * hyper['gan_w']),
                                ('loss_dis_b', self.loss_dis_b.data * hyper['gan_w']),
                                ('loss_dis_total', self.loss_dis_total.data * hyper['gan_w']),
                                ('loss_gen_adv_a', self.loss_gen_adv_a.data * hyper['gan_w']),
                                ('loss_gen_adv_b', self.loss_gen_adv_b.data * hyper['gan_w'])])
        if hyper['recon_x_w'] != 0:
            ret_dict['loss_gen_recon_x_a'] = self.loss_gen_recon_x_a.data * hyper['recon_x_w']
            ret_dict['loss_gen_recon_x_b'] = self.loss_gen_recon_x_b.data * hyper['recon_x_w']
        if hyper['recon_s_w'] != 0:
            ret_dict['loss_gen_recon_s_a'] = self.loss_gen_recon_s_a.data * hyper['recon_s_w']
            ret_dict['loss_gen_recon_s_b'] = self.loss_gen_recon_s_b.data * hyper['recon_s_w']
        if hyper['recon_c_w'] != 0:
            ret_dict['loss_gen_recon_c_a'] = self.loss_gen_recon_c_a.data * hyper['recon_c_w']
            ret_dict['loss_gen_recon_c_b'] = self.loss_gen_recon_c_b.data * hyper['recon_c_w']
        if hyper['recon_x_s_w'] != 0:
            ret_dict['loss_gen_x_recon_s_a'] = self.loss_gen_x_recon_s_a.data * hyper['recon_s_w']
            ret_dict['loss_gen_x_recon_s_b'] = self.loss_gen_x_recon_s_b.data * hyper['recon_s_w']
        if hyper['recon_x_s_w'] != 0:
            ret_dict['loss_gen_x_recon_c_a'] = self.loss_gen_x_recon_c_a.data * hyper['recon_s_w']
            ret_dict['loss_gen_x_recon_c_b'] = self.loss_gen_x_recon_c_b.data * hyper['recon_s_w']
        if hyper['content_w'] != 0:
            ret_dict['loss_gen_content_a'] = self.loss_gen_content_a * hyper['content_w']
            ret_dict['loss_gen_content_b'] = self.loss_gen_content_b * hyper['content_w']
        if hyper['style_w'] != 0:
            ret_dict['loss_gen_style_a'] = self.loss_gen_style_a * hyper['content_w']
            ret_dict['loss_gen_style_b'] = self.loss_gen_style_b * hyper['style_w']


        ret_dict['loss_gen_total'] = self.loss_gen_total.data
        return ret_dict

    def set_requires_grad(self, net, requires_grad=False):
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad  # to avoid computation

def tensor2im(image_tensor, imtype=np.uint8, cvt_rgb=True,initial_mothod='Normal'):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1 and cvt_rgb:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    if initial_mothod == 'VGG':
        image_numpy[0] = ((image_numpy[0] * 0.229) + 0.485) * 255
        image_numpy[1] = ((image_numpy[1] * 0.224) + 0.456) * 255
        image_numpy[2] = ((image_numpy[2] * 0.225) + 0.406) * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


