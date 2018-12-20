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
        self.enc_c_a = networks.define_VGGF()
        self.enc_c_b = networks.define_VGGF()
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
        self.loss_gen_x_recon_s_a = self.recon_criterion(s_a_x_recon, s_a_prime)   if hyperparameters['recon_x_s_w'] > 0 else 0
        self.loss_gen_x_recon_s_b = self.recon_criterion(s_b_x_recon, s_b_prime)  if hyperparameters['recon_x_s_w'] > 0 else 0
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a_prime)  if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b_prime) if hyperparameters['recon_s_w'] > 0 else 0
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
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['content_w'] * self.loss_gen_content_a + \
                              hyperparameters['content_w'] * self.loss_gen_content_b + \
                              hyperparameters['recon_x_s_w'] * self.loss_gen_x_recon_s_a + \
                              hyperparameters['recon_x_s_w'] * self.loss_gen_x_recon_s_b + \
                              hyperparameters['style_w'] * self.loss_gen_style_a + \
                              hyperparameters['style_w'] * self.loss_gen_style_b
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


