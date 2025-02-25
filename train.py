import yaml
from types import SimpleNamespace
import logging
import os
import sys
import shutil
import  math
import torch
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from eval import eval_net
from Networks import WNet,PatchGAN, SwinDiscriminator_U, SwinDiscriminator,AttenDiscriminator, Generator_PRE
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import fastMRIdataset
from loss import netLoss,set_grad
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.FloatTensor)

def train(args):
    # Init PRE network
    PRE_model = Generator_PRE(args)
    print(PRE_model.bilinear)
    logging.info(f'Network:\n'
                 f'\t{"Bilinear" if PRE_model.bilinear else "Transposed conv"} upscaling')
    # PRE_optimizer = torch.optim.SGD(G_model.parameters(), lr=args.lr, momentum=0.9)
    # PRE_optimizer = torch.optim.RMSprop(G_model.parameters(), lr=args.lr, alpha=(0.999))
    PRE_optimizer = torch.optim.Adam(PRE_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.000)
    PRE_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(PRE_optimizer, 'min', patience=5)
    PRE_model.to(device=args.device)
    # Init Generator network
    G_model = WNet(args)
    logging.info(f'Network:\n'
                 f'\t{"Bilinear" if G_model.bilinear else "Transposed conv"} upscaling')
    # G_optimizer = torch.optim.SGD(G_model.parameters(), lr=args.lr, momentum=0.9)
    #G_optimizer = torch.optim.RMSprop(G_model.parameters(), lr=args.lr, alpha=(0.999))
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.000)
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience=5)
    G_model.to(device=args.device)
    # Init Discriminator network
    D_model = PatchGAN(1, crop_center=args.crop_center)
    #D_model = SwinDiscriminator(config=args, num_classes=2, crop_center=args.crop_center, img_size=args.img_size, in_chans=1)
    #D_model = TransDiscriminator(config=args)
    #D_model = AttenDiscriminator(config=args)
    #D_optimizer = torch.optim.SGD(G_model.parameters(), lr=2*args.lr, momentum=0.9)
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=args.lr*2, betas=(0.9, 0.999), weight_decay=0.000)
    #D_optimizer = torch.optim.RMSprop(D_model.parameters(), lr=args.lr, alpha=(0.999))
    D_model.to(device=args.device)

    # Init Dataloaders
    if args.dataset == 'fastMRI':
        train_dataset = fastMRIdataset(args.train_data_dir, args)
        val_dataset = fastMRIdataset(args.val_data_dir, args, validtion_flag=True)
    else:
        logging.error("Data type not supported")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.train_num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.val_num_workers,
                            pin_memory=True, drop_last=True) #Shuffle is true for diffrent images on tensorboard

    # train_list=list(train_loader)


    # Init tensorboard writer
    if args.tb_write_losses or args.tb_write_images :
        writer = SummaryWriter(log_dir=args.output_dir + '/tensorboard')

    # Init loss object
    loss = netLoss(args)

    # Load checkpoint
    if args.load_cp:
        checkpoint = torch.load(args.load_cp, map_location=args.device)
        PRE_model.load_state_dict(checkpoint['G_model_state_dict'])
        G_model.load_state_dict(checkpoint['G_model_state_dict'])
        D_model.load_state_dict(checkpoint['D_model_state_dict'])

        if args.resume_training:
            start_epoch = int(checkpoint['epoch'])
            PRE_optimizer.load_state_dict(checkpoint['PRE_optimizer_state_dict'])
            G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
            PRE_scheduler.load_state_dict(checkpoint['PRE_scheduler_state_dict'])
            G_scheduler.load_state_dict(checkpoint['G_scheduler_state_dict'])
            logging.info(f'Models, optimizer and scheduler loaded from {args.load_cp}')
        else:
            logging.info(f'Models only load from {args.load_cp}')
    else:
        start_epoch = 0
    #Start training

    logging.info(f'''Starting training:
        Epochs:          {args.epochs_n }
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Device:          {args.device}
    ''')


    try: #Catch keyboard interrupt and save state
        for epoch in range(start_epoch, args.epochs_n):
            PRE_model.train()
            G_model.train()
            D_model.train()
            progress = 0

            with tqdm(desc=f'Epoch {epoch + 1}/{args.epochs_n}', unit=' imgs') as pbar:
                #Train loop
                for batch in train_loader:

                    ori_Kspace = batch['ori_Kspace'].to(device=args.device, dtype=torch.float32)
                    masked_Kspaces = batch['masked_Kspaces'].to(device=args.device, dtype=torch.float32)
                    target_Kspace = batch['target_Kspace'].to(device=args.device, dtype=torch.float32)
                    target_img = batch['target_img'].to(device=args.device, dtype=torch.float32)
                    sensitivity_map = batch['sensitivity_map'].to(device=args.device, dtype=torch.float32)
                    # Forward PRE:
                    enh_Kspace, enh_img, ori_img, threshold = PRE_model(masked_Kspaces, sensitivity_map)
                    #Forward G:
                    rec_img, rec_Kspace, rec_mid_image, ORI_img, w1, w2, w3= G_model(enh_Kspace, enh_img, sensitivity_map)

                    #Forward D for G loss:
                    if args.GAN_training:
                        real_D_example = target_img.detach()
                        #fake_D_example = rec_img + (100000 * (real_D_example - rec_img))
                        fake_D_example = rec_img.detach()
                        ############################################experiment
                        # TEMP=fake_D_example.cpu().detach().numpy()
                        # TEMP_real = real_D_example.cpu().detach().numpy()
                        # plt.figure(figsize=(8, 6))
                        # plt.imshow(TEMP[7,0,:,:], cmap='jet', vmin=0, vmax=1)
                        # plt.colorbar(label='Error map')
                        # # plt.title('Error Map (GT)')
                        # plt.axis('off')
                        # plt.show()
                        #
                        # plt.figure(figsize=(8, 6))
                        # plt.imshow(TEMP_real[7,0,:,:], cmap='jet', vmin=0, vmax=1)
                        # plt.colorbar(label='Error map')
                        # # plt.title('Error Map (GT)')
                        # plt.axis('off')
                        # plt.show()
                        #####################################experiment
                        D_real_pred = D_model(real_D_example)
                        D_fake_pred = D_model(fake_D_example)
                        gp = gradient_penalty(D_model, real_D_example, fake_D_example.detach())
                    else:
                        D_fake_pred = None


                    lambda_l1 = args.lambda_l1
                    lambda_l2 = args.lambda_l2
                    #Calc G losses:
                    FullLoss_PRE = loss.calc_PRE_loss(target_img, enh_img, masked_Kspaces)
                    FullLoss, ImL2, ImL1, KspaceL2, advLoss, FFLloss, DCTLoss, DCT_imgLoss, SSIMLoss= loss.calc_gen_loss(ori_Kspace, rec_img, rec_Kspace, target_img, target_Kspace, ORI_img, w1, w2, w3, enh_img, D_fake_pred)
                    # FullLoss += l1_regularization(G_model,lambda_l1) + l2_regularization(G_model, lambda_l2)
                    #Forward D for D loss:
                    if args.GAN_training:
                        D_fake_detach = D_model(fake_D_example.detach())   #Stop backprop to G by detaching
                        D_real_loss,D_fake_loss,DLoss = loss.calc_disc_loss(D_real_pred, D_fake_detach)
                        # DLoss += l1_regularization(D_model,lambda_l1) + l2_regularization(D_model, lambda_l2)
                        if args.GP:
                            DLoss+=0.02*gp
                        # Train/stop Train D criteria
                        train_D = advLoss.item()<D_real_loss.item()*1.25

                    ##################
                    PRE_optimizer.zero_grad()
                    FullLoss_PRE.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(PRE_model.parameters(), max_norm=1.0)
                    PRE_optimizer.step()
                    ##########################
                    #Optimize parameters
                    #Update G
                    if args.GAN_training:
                        set_grad(D_model, False)  # No D update

                    G_optimizer.zero_grad()
                    FullLoss.backward()
                    torch.nn.utils.clip_grad_norm_(G_model.parameters(), max_norm=1.0)
                    G_optimizer.step()
                    #Update D
                    if args.GAN_training:
                        set_grad(D_model, True)  # enable backprop for D
                        if train_D:
                            D_optimizer.zero_grad()
                            DLoss.backward()
                            torch.nn.utils.clip_grad_norm_(D_model.parameters(), max_norm=1.0)
                            D_optimizer.step()
                            # for name, param in D_model.named_parameters():
                            #     if param.grad is not None:
                            #         print(f'Parameter: {name}, Gradient norm: {param.grad.norm()}')
                            #     else:
                            #         print(f'Parameter: {name}, Gradient: None')

                    #Update progress bar
                    progress += 100*target_Kspace.shape[0]/len(train_dataset)
                    if args.GAN_training:
                        pbar.set_postfix(**{'FFLloss':FFLloss.item()*args.loss_weights[4],'FullLoss': FullLoss.item(),'FullLoss_PRE': FullLoss_PRE.item(),'ImL2': ImL2.item()*args.loss_weights[0], 'ImL1': ImL1.item()*args.loss_weights[1],
                                            'KspaceL2': KspaceL2.item()*args.loss_weights[2],'Adv G': advLoss.item()**args.loss_weights[3],'Adv Dloss':DLoss.item(),'Adv D - Real' : D_real_loss.item(),'Adv D - Fake' : D_fake_loss.item(),'Adv-GP':gp.item(), 'DCT': DCTLoss.item()*args.loss_weights[5], 'DCT_imgLoss': DCT_imgLoss.item()*args.loss_weights[6],'SSIMLoss':SSIMLoss.item(), 'weight_k': w1.item(),'weight_Im': w2.item(),'weight_Sparse': w3.item(), 'threshold': threshold.item(),'bias': (w1.item() - w2.item()), 'Prctg of train set': progress})
                    else:
                        pbar.set_postfix(**{'FFLloss':FFLloss.item(),'FullLoss': FullLoss.item(), 'FullLoss_PRE': FullLoss_PRE.item(), 'ImL2': ImL2.item(), 'ImL1': ImL1.item(),'DCT': DCTLoss.item()*args.loss_weights[5], 'DCT_imgLoss': DCT_imgLoss.item()*args.loss_weights[6],
                                            'KspaceL2': KspaceL2.item(),'SSIMLoss':SSIMLoss.item, 'weight_k': w1.item(),'weight_Im': w2.item(),'weight_Sparse': w3.item(), 'threshold': threshold.item(), 'bias': w1.item() - w2.item(), 'Prctg of train set': progress})
                    pbar.update(target_Kspace.shape[0])# current batch size

            # On epoch end
            # Validation
            val_rec_img, val_full_img, val_F_rec_Kspace, val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR, val_SSIM, w1_val, w2_val, w3_val, threshold_val, val_FullLoss_PRE=\
                eval_net(PRE_model, G_model, val_loader, loss, args.device)

            logging.info('Validation full score: {}, ImL2: {}. ImL1: {}, KspaceL2: {}, PSNR: {}, SSIM: {}'
                         .format(val_FullLoss, val_ImL2, val_ImL1, val_KspaceL2, val_PSNR, val_SSIM))
            # Schedular update
            PRE_scheduler.step(val_FullLoss_PRE)
            G_scheduler.step(val_FullLoss)


        # 2. 记录这个epoch的模型的参数和梯度
        #     for tag, value in G_model.named_parameters():
        #         tag = tag.replace('.', '/')
        #         writer.histo_summary(tag, value.data.cpu().numpy(), epoch)
        #         writer.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch)

            #Write to TB:
            if args.tb_write_losses:
                writer.add_scalar('train/FullLoss', FullLoss.item(), epoch)
                writer.add_scalar('train/FullLoss_PRE', FullLoss_PRE.item(), epoch)
                writer.add_scalar('train/ImL2', ImL2.item(), epoch)
                writer.add_scalar('train/ImL1', ImL1.item(), epoch)
                writer.add_scalar('train/KspaceL2', KspaceL2.item(), epoch)
                writer.add_scalar('train/FFL', FFLloss.item(), epoch)
                writer.add_scalar('train/DCTLoss', DCTLoss.item(), epoch)
                writer.add_scalar('train/DCT_imgLoss', DCT_imgLoss.item(), epoch)
                writer.add_scalar('train/SSIMLoss', SSIMLoss.item(), epoch)
                writer.add_scalar('train/weight_k', w1.item(), epoch)
                writer.add_scalar('train/weight_Im', w2.item(), epoch)
                writer.add_scalar('train/weight_Sparse', w3.item(), epoch)
                writer.add_scalar('train/Threshold', threshold.item(), epoch)
                writer.add_scalar('train/bias', w1.item() - w2.item(), epoch)
                writer.add_scalar('train/learning_rate_G', G_optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('train/learning_rate_PRE', PRE_optimizer.param_groups[0]['lr'], epoch)
                if args.GAN_training:
                    writer.add_scalar('train/G_AdvLoss', advLoss.item(), epoch)
                    writer.add_scalar('train/D_AdvLoss', DLoss.item(), epoch)
                writer.add_scalar('validation/FullLoss_PRE', val_FullLoss_PRE, epoch)
                writer.add_scalar('validation/FullLoss', val_FullLoss, epoch)
                writer.add_scalar('validation/ImL2', val_ImL2, epoch)
                writer.add_scalar('validation/ImL1', val_ImL1, epoch)
                writer.add_scalar('validation/KspaceL2', val_KspaceL2, epoch)
                writer.add_scalar('validation/weight_k', w1_val, epoch)
                writer.add_scalar('validation/weight_Im', w2_val, epoch)
                writer.add_scalar('validation/weight_Sparse', w3_val, epoch)
                writer.add_scalar('validation/Threshold', threshold_val.item(), epoch)
                writer.add_scalar('validation/bias', w1_val - w2_val, epoch)
                writer.add_scalar('validation/PSNR', val_PSNR, epoch)
                writer.add_scalar('validation/SSIM', val_SSIM, epoch)

            if args.tb_write_images:
                writer.add_images('train/Fully_sampled_images', target_img, epoch)
                writer.add_images('train/enh_images', enh_img, epoch)
                writer.add_images('train/rec_images', rec_img, epoch)
                writer.add_images('train/ORI_images', ORI_img, epoch)
                writer.add_images('train/Kspace_rec_images', rec_mid_image, epoch)
                writer.add_images('validation/Fully_sampled_images', val_full_img, epoch)
                writer.add_images('validation/rec_images', val_rec_img, epoch)
                writer.add_images('validation/Kspace_rec_images', val_F_rec_Kspace, epoch)

            #Save Checkpoint
            torch.save({
                'epoch': epoch,
                'PRE_model_state_dict': PRE_model.state_dict(),
                'PRE_optimizer_state_dict': PRE_optimizer.state_dict(),
                'PRE_scheduler_state_dict': PRE_scheduler.state_dict(),
                'G_model_state_dict': G_model.state_dict(),
                'G_optimizer_state_dict': G_optimizer.state_dict(),
                'G_scheduler_state_dict': G_scheduler.state_dict(),
                'D_model_state_dict': D_model.state_dict(),
                'D_optimizer_state_dict': D_optimizer.state_dict(),
            }, args.output_dir + f'/CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    except KeyboardInterrupt:
        torch.save({
            'epoch': epoch,
            'PRE_model_state_dict': PRE_model.state_dict(),
            'PRE_optimizer_state_dict': PRE_optimizer.state_dict(),
            'PRE_scheduler_state_dict': PRE_scheduler.state_dict(),
            'G_model_state_dict': G_model.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'G_scheduler_state_dict': G_scheduler.state_dict(),
            'D_model_state_dict': D_model.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
        }, args.output_dir + f'/CP_epoch{epoch + 1}_INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    if args.tb_write_losses or args.tb_write_images:
        writer.close()


def get_args():
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    args = SimpleNamespace(**data)
    args.mask_path = './Masks/{}/mask_{}_{}.pickle'.format(args.mask_type, args.sampling_percentage, args.img_size)

    pprint(data)
    return args

def gradient_penalty(D, xr, xf):
    # [b,1]
    t = torch.rand(320, 1).cuda()
    # [b,1]=>[b,2]
    t = t.expand_as(xr)
    mid = t * xr + (1 - t) * xf
    mid.requires_grad_()

    pred = D(mid)
    grads = torch.autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

    return gp
def l1_regularization(model, lambda_l1):
    l1_reg = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_reg = l1_reg + torch.norm(param, p=1)
    return lambda_l1 * l1_reg

def l2_regularization(model, lambda_l2):
    l2_reg = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg = l2_reg + torch.norm(param, p=2)
    return lambda_l2 * l2_reg

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    # Create output dir
    try:
        os.mkdir(args.output_dir)
        logging.info('Created checkpoint directory')
    except OSError:
        pass

    # Copy configuration file to output directory
    shutil.copyfile('config.yaml',os.path.join(args.output_dir,'config.yaml'))

    # Set device and GPU (currently only single GPU training is supported
    logging.info(f'Using device {args.device}')
    if args.device == 'cuda':
        logging.info(f'Using GPU {args.gpu_id}')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        torch.backends.cudnn.benchmark = True

    train(args)


