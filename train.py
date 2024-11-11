import os
from tqdm import tqdm
import logging
import torch
import torch.backends.cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR

from data import LLIEDataset
from loss import VGGLoss, HDRloss, HistogramMatchingLoss

# from model import enhance_color
from model_cpga import CPGAnet, CPGAnet_blk

from config import get_config

# torch.manual_seed(0)

def load_data(cfg):
    train_data_transform = transforms.Compose([
        # transforms.Resize([400, 600]),
        # transforms.CenterCrop([256, 256]),
        transforms.ToTensor()
    ])
    val_data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_ll_dataset = LLIEDataset(cfg.ori_data_path, cfg.ll_data_path, train_data_transform, dataset_type=cfg.dataset_type, cropsize=cfg.crop_size, istrain=True)
    train_loader = torch.utils.data.DataLoader(train_ll_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    val_ll_dataset = LLIEDataset(cfg.val_ori_data_path, cfg.val_ll_data_path, val_data_transform, False, dataset_type=cfg.dataset_type)
    val_loader = torch.utils.data.DataLoader(val_ll_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                             num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    return train_loader, len(train_loader), val_loader, len(val_loader)

def save_model(epoch, path, net, optimizer, net_name):
    save_path = os.path.join(path, 'weights')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               f=os.path.join(save_path, '{}_{}.pkl'.format('enhance', epoch)))

def load_network(cfg, device):
    if cfg.plus:
        net = CPGAnet_blk(
            n_channels=cfg.n_resch, gamma_n_channel=cfg.n_gammach, n_cpblks=cfg.n_cpblks, 
            n_IAAFch=cfg.n_IAAFch, isdgf=cfg.efficient, iscpgablks=cfg.is_cpgablks
        ).to(device)
    else:
        # net = enhance_color().to(device)
        net = CPGAnet(isdgf=cfg.efficient).to(device)
    if cfg.ckpt:
        net.load_state_dict(torch.load(os.path.join(cfg.ckpt))['state_dict'])
    else:
        print('No pretrain model, train from scratch')
    return net

def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer

def loss_func(device):
    criterion = torch.nn.L1Loss().to(device)
    return criterion

def load_summaries(cfg, model_dir='runs/training'):
    summary_dir = os.path.join(model_dir, 'logs')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    summary = SummaryWriter(log_dir=summary_dir, comment='')
    return summary

def train_step(epoch, train_loader, train_number, device, network, loss_funcs, optimizer, summary, cfg, isSST=False, teacher_network=None):
    train_bar = tqdm(train_loader)
    total_loss = 0.    
    criterion = loss_funcs[0]
    vggloss = loss_funcs[1]
    hdrloss = loss_funcs[2]
    ssimloss = loss_funcs[3]
    mseloss = loss_funcs[4]
    hmloss = loss_funcs[5]
    
    if isSST:
        epochs = cfg.SSTepochs
    else:
        epochs = cfg.epochs
    
    for step, (ori_image, LL_image, _) in enumerate(train_bar):
        count = epoch * train_number + (step + 1)
        ori_image, LL_image = ori_image.to(device), LL_image.to(device)           
        LLIE_image = network(LL_image,)        
        if cfg.knowledge_distillation==0:
            LLIE_image = network(LL_image,)
        else:
            LLIE_image, ln_gamma, ln_intersection, ln_llie= network(LL_image, get_all=True)

        if isSST:            
            recon_loss = criterion(LLIE_image, LL_image) 
            vgg_loss = vggloss(LLIE_image, LL_image)
            hdr_loss = hdrloss(LLIE_image, LL_image)
            ssim_loss = 1. - ssimloss(LLIE_image, LL_image)
            hm_loss = hmloss(LLIE_image, LL_image)
        else:
            recon_loss = criterion(LLIE_image, ori_image) 
            vgg_loss = vggloss(LLIE_image, ori_image)
            hdr_loss = hdrloss(LLIE_image, ori_image)
            ssim_loss = 1. - ssimloss(LLIE_image, ori_image)
            hm_loss = hmloss(LLIE_image, ori_image)
        loss =  cfg.lambda_l1 * recon_loss + cfg.lambda_per * vgg_loss + cfg.lambda_l1 *hdr_loss + cfg.lambda_ssim * ssim_loss + cfg.lambda_hm * hm_loss
        total_loss = total_loss + loss.item()

        if cfg.knowledge_distillation>0:
            if not cfg.knowledge_distillation>1:
                with torch.no_grad():
                    LLIE_teacher, tn_gamma, tn_intersection, tn_llie = teacher_network(LL_image, get_all=True)
                distil_loss = mseloss(tn_gamma, ln_gamma) \
                            + mseloss(tn_intersection, ln_intersection)\
                            + mseloss(tn_llie, ln_llie)             
            loss = loss + cfg.lambda_distill * distil_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        mode = 'SST' if isSST else 'Training'    
        summary.add_scalar(f'{mode}/loss', loss.item(), count)
        summary.add_scalar(f'{mode}/recon_loss', recon_loss.item(), count)
        summary.add_scalar(f'{mode}/vgg_loss', vgg_loss.item(), count)
        summary.add_scalar(f'{mode}/hdr_loss', hdr_loss.item(), count)
        summary.add_scalar(f'{mode}/ssim_loss', ssim_loss.item(), count)
        summary.add_scalar(f'{mode}/hm_loss', hm_loss.item(), count)
        summary.add_scalar(f'{mode}/lr', optimizer.param_groups[0]['lr'], count)
        summary.add_scalar(f'{mode}/distil_loss', distil_loss.item(), count) if cfg.knowledge_distillation>0 and not isSST else None
        train_bar.set_description_str('Epoch: {}/{} | Step: {}/{} | lr: {:.6f} | Loss: {:.6f}-{:.6f}'
                .format(epoch + 1, epochs, step + 1, train_number,
                        optimizer.param_groups[0]['lr'], 
                        total_loss/(step+1), recon_loss.item(), 
                    )
                )
        
def eval(cfg, device, summary, val_loader, network, sample_dir, ssim, psnr, lpips, epoch, isSST=False):
    LLIE_valing_results = {'mse': 0, 'ssims': 0, 'psnrs': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'lpipss': 0, 'lpips': 0,} 
    if isSST:
        sample_dir = sample_dir + '-SST'
        
    val_bar = tqdm(val_loader)
    max_step = 20
    if len(val_loader.dataset) <max_step:
        max_step = len(val_loader.dataset)-1
    save_image = None
    for step, (ori_image, LL_image, _) in enumerate(val_bar):
        ori_image, LL_image = ori_image.to(device), LL_image.to(device)
        if isSST:
            ori_image = LL_image

        with torch.no_grad():
            LLIE_image = network(LL_image)
            
        LLIE_valing_results['batch_sizes'] += cfg.batch_size
        batch_psnr = psnr(LLIE_image, ori_image).item()
        batch_ssim = ssim(LLIE_image, ori_image).item()
        LLIE_valing_results['psnrs'] += batch_psnr * cfg.batch_size
        LLIE_valing_results['ssims'] += batch_ssim * cfg.batch_size
            
        LLIE_valing_results['psnr'] = LLIE_valing_results['psnrs'] / LLIE_valing_results['batch_sizes']
        LLIE_valing_results['ssim'] = LLIE_valing_results['ssims'] / LLIE_valing_results['batch_sizes']
            
        batch_lpips = lpips(LLIE_image, ori_image).item()
        LLIE_valing_results['lpipss'] += batch_lpips * cfg.batch_size
        LLIE_valing_results['lpips'] = LLIE_valing_results['lpipss'] / LLIE_valing_results['batch_sizes']

        if step <= max_step:   # only save image 10 times
            sv_im = torchvision.utils.make_grid(torch.cat((LL_image, LLIE_image, ori_image), 0), nrow=ori_image.shape[0])
            if save_image == None:
                save_image = sv_im
            else:
                save_image = torch.cat((save_image, sv_im), dim=2)
        if step == max_step:   # only save image 15 times
           torchvision.utils.save_image(
                    save_image,
                    os.path.join(sample_dir, '{}.jpg'.format(epoch + 1))
                )
        val_bar.set_description_str('[LLIE] PSNR: %.4f dB SSIM: %.4f LPIPS: %.4f' % (
                        LLIE_valing_results['psnr'], LLIE_valing_results['ssim'], LLIE_valing_results['lpips']))
        
    if isSST:
        summary.add_scalar('SST/PSNR', LLIE_valing_results['psnr'], epoch)
        summary.add_scalar('SST/ssim', LLIE_valing_results['ssim'], epoch)
        summary.add_scalar('SST/lpips', LLIE_valing_results['lpips'], epoch)
    else:            
        summary.add_scalar('Metrics/PSNR', LLIE_valing_results['psnr'], epoch)
        summary.add_scalar('Metrics/ssim', LLIE_valing_results['ssim'], epoch)
        summary.add_scalar('Metrics/lpips', LLIE_valing_results['lpips'], epoch)

def main(cfg):
    # -------------------------------------------------------------------
    # create model dir
    model_dir = os.path.join('runs', 'training', cfg.net_name)
    if not os.path.exists(model_dir):
        os.makedirs(os.path.join('runs', 'training', cfg.net_name))
    # -------------------------------------------------------------------
    logging.basicConfig(filename=os.path.join(model_dir, "training_info.log"), 
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                        filemode='w')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Network name: "+ cfg.net_name)
    # -------------------------------------------------------------------
    # load summaries
    summary = load_summaries(cfg, model_dir)
    # -------------------------------------------------------------------
    # load data
    train_loader, train_number, val_loader, val_number = load_data(cfg)
    logger.info("dataset_type: " + cfg.dataset_type)
    logger.info("Train data size: %s"%str(train_number))
    logger.info("Val data size: %s"%str(val_number))
    # -------------------------------------------------------------------
    # load loss
    criterion = loss_func(device)
    vggloss = VGGLoss(device=device)
    hdrloss = HDRloss()
    ssimloss = SSIM(data_range=1.).to(device=device)
    mseloss = torch.nn.MSELoss()
    hmloss = HistogramMatchingLoss(cfg.hm_bins)
    losses = [criterion, vggloss, hdrloss, ssimloss, mseloss, hmloss]
    # -------------------------------------------------------------------
    # load network
    network = load_network(cfg, device)
    logger.info('Total params: %d'%sum(p.numel() for p in network.parameters() if p.requires_grad))
    logger.info('-efficient: %s'%cfg.efficient)
    logger.info('-plus: %s'%cfg.plus)
    logger.info('-n_cpblks: %d'%cfg.n_cpblks)
    logger.info('-n_channels: %d'%cfg.n_resch)
    logger.info('-gamma_n_channel: %d'%cfg.n_gammach)
    logger.info('-IAAF_n_channel: %d'%cfg.n_IAAFch)
    logger.info('-is_cpgablks: %s'%cfg.is_cpgablks)

    # Knowledge distillation
    if cfg.knowledge_distillation==1:
        if cfg.dataset_type == 'LOL-v1':
            tn_weight = cfg.ckpt_teacher
        teacher_network = CPGAnet(n_channels=16, isdgf=False).to(device)
        teacher_network.load_state_dict(torch.load(tn_weight)['state_dict'])
        teacher_network.eval()
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(network, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.epochs//3+1, T_mult=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.epochs//3+1, T_mult=1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.epochs*2//5, gamma=0.1, last_epoch=-1)
    logger.info("Epochs: %d"%cfg.epochs)
    logger.info("Learning rate: %f"% cfg.lr)
    logger.info("lambda_l1: %f"% cfg.lambda_l1)
    logger.info("lambda_per: %f"% cfg.lambda_per)
    logger.info("lambda_hdr: %f"% cfg.lambda_hdr)
    logger.info("lambda_ssim: %f"% cfg.lambda_ssim)
    logger.info("lambda_distill: %f"% cfg.lambda_distill)
    logger.info("lambda_hm: %f"% cfg.lambda_hm)
    
    logger.info("Batch size: %d"%cfg.batch_size)
    logger.info("Val batch size: %d"%cfg.val_batch_size)
    logger.info("Optimizer: %s"% "Adam")
    logger.info("scheduler:  %s"%"torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.epochs//3+1, T_mult=1)")
    # -------------------------------------------------------------------
    # create sample dir
    sample_dir = os.path.join(model_dir, cfg.sample_output_folder)
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.isdir(sample_dir+'-SST'):
        os.makedirs(sample_dir+'-SST') if not cfg.skipSST else None
    logger.info("Skip Self supervised training: %s"% cfg.skipSST)
    logger.info("SST epochs: %d"% cfg.SSTepochs)
    # -------------------------------------------------------------------

    ssim = SSIM(data_range=1.).to(device=device)
    psnr = PSNR(data_range=1.).to(device=device)
    lpips = LPIPS(net_type='alex').to(device=device)
    
    # start train
    print('-------------------------------------------------------------------')
    print('Start train')
    network.train()
    if not cfg.skipSST :
        print('Start self-supervised training')
        for epoch in range(cfg.SSTepochs):
            # SST train
            train_step(epoch, train_loader, train_number, device, network, losses, optimizer, summary, cfg, isSST=True)   
            network.eval()
            eval(cfg, device, summary, val_loader, network, sample_dir, ssim, psnr, lpips, epoch, isSST=True)
            network.train()
            
            # save per epochs model
            save_model(epoch, model_dir, network, optimizer, cfg.net_name+'-SST')            
        print('End self-supervised training')
    else:
        print('Skip self-supervised training')

    for epoch in range(cfg.epochs):
        train_step(epoch, train_loader, train_number, device, network, losses, optimizer, summary, cfg)   
        scheduler.step()
        # -------------------------------------------------------------------
        # start validation
        print('Epoch: {}/{} | Validation Model Saving Images'.format(epoch + 1, cfg.epochs))
        network.eval()
        eval(cfg, device, summary, val_loader, network, sample_dir, ssim, psnr, lpips, epoch)
        network.train()
        # -------------------------------------------------------------------
        # save per epochs model
        save_model(epoch, model_dir, network, optimizer, cfg.net_name)
    # -------------------------------------------------------------------
    # train finish
    summary.close()



if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
