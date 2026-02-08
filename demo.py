import os
import logging
from shutil import copy
import torch
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm
from config import get_config
# from model import CPGAnet
from model_cpga import CPGAnet, CPGAnet_blk
from data import LLIEDataset
# from thop import profile
try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except Exception:
    FlopCountAnalysis = None
    _HAS_FVCORE = False

try:
    from ptflops import get_model_complexity_info
    _HAS_PTFLOPS = True
except Exception:
    get_model_complexity_info = None
    _HAS_PTFLOPS = False

from skimage import img_as_ubyte
import cv2, time

import warnings
warnings.filterwarnings("ignore")


# @logger
def load_data_eval(cfg):
    data_transform = transforms.Compose([
        # transforms.Resize([400, 600]),
        # transforms.RandomCrop([256, 256]),
        # transforms.CenterCrop([400, 600]),
        transforms.ToTensor()
    ])
    val_ll_dataset = LLIEDataset(cfg.val_ori_data_path, cfg.val_ll_data_path, data_transform, dataset_type = cfg.dataset_type, isdemo=True)
    val_loader = torch.utils.data.DataLoader(val_ll_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                             num_workers=0, drop_last=True, pin_memory=True)

    return val_loader, len(val_loader)

def load_pretrain_network(cfg, device, logger=None):
    if cfg.plus:
        net = CPGAnet_blk(
            n_channels=cfg.n_resch, gamma_n_channel=cfg.n_gammach, n_cpblks=cfg.n_cpblks, 
            n_IAAFch=cfg.n_IAAFch, iscpgablks=cfg.is_cpgablks,
            iaaf_type=cfg.iaaf_type, iaaf_ablation=cfg.iaaf_ablation, iaaf_scoring=[int(cfg.iaaf_scoring[0]), int(cfg.iaaf_scoring[1])],
            conv=cfg.conv_type, block=cfg.block_type, bn=cfg.bn,
            efficient =cfg.efficient, #isdgf=cfg.efficient
        ).to(device)
    else:
        # net = enhance_color().to(device)
        net = CPGAnet(isdgf=cfg.efficient).to(device)
        logger.info('CPGAnet loaded')
    net.load_state_dict(torch.load(os.path.join(cfg.ckpt))['state_dict'])
    return net

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main(cfg):
    # -------------------------------------------------------------------
    # create output directory
    out_dir = os.path.join('runs', 'demo', cfg.net_name)
    os.makedirs(out_dir, exist_ok=True)
    # -------------------------------------------------------------------
    # create log file
    logging.basicConfig(filename=os.path.join(out_dir, "info.log"), 
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
    # -------------------------------------------------------------------
    # load data
    val_loader, val_number = load_data_eval(cfg)
    logger.info('Data loaded')
    logger.info('Number of images: %d' % val_number)
    # -------------------------------------------------------------------
    # load network
    if cfg.ckpt:
        network = load_pretrain_network(cfg, device, logger)
        copy(cfg.ckpt, out_dir)
    else:
        raise ValueError('No checkpoint found')
    

    # start demo
    print('Start demo')
    network.eval()
    demo(device, val_loader, network, out_dir)
    # ---------------------------
    measure_network_efficiency(device, network, logger)
    print('End demo')

def measure_network_efficiency(device, network, logger=None, figsize=(400, 600)):
    """Measure runtime and (optionally) model FLOPs/params.

    This function is resilient: if `fvcore` or `ptflops` are not installed or fail,
    it will still report timing and total parameters and will not raise.
    """
    itr_cnt = int(1e2)
    start_t = time.time_ns()
    with torch.no_grad():
        for _ in range(itr_cnt):
            ll_image = torch.rand(1, 3, figsize[0], figsize[1]).to(device)
            _ = network(ll_image)
    end_t = time.time_ns()
    total_time = end_t - start_t

    input = torch.randn(1, 3, figsize[0], figsize[1]).to(device)

    flops = None
    if FlopCountAnalysis is not None:
        try:
            flops = FlopCountAnalysis(network, (input, ))
        except Exception as e:
            if logger:
                logger.warning(f'fvcore FlopCountAnalysis failed: {e}')
            else:
                print(f'Warning: fvcore FlopCountAnalysis failed: {e}')

    macs, params = None, None
    if get_model_complexity_info is not None:
        try:
            macs, params = get_model_complexity_info(network, (3,  figsize[0], figsize[1]), as_strings=True,
                                                     print_per_layer_stat=False, verbose=False, output_precision=4)
        except Exception as e:
            if logger:
                logger.warning(f'ptflops get_model_complexity_info failed: {e}')
            else:
                print(f'Warning: ptflops get_model_complexity_info failed: {e}')

    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    avg_time_ms = (total_time / 1e9) * 1e3 / itr_cnt
    total_time_ms = ((end_t - start_t) / 1e9) * 1e3 / itr_cnt

    if logger:
        if flops is not None:
            logger.info('FLOPs = ' + str(flops.total()/1000**3) + ' G')
        else:
            logger.info('FLOPs: fvcore not available')
        logger.info('Total params: ' + str(total_params))
        logger.info('Average time (per run) taken by network is : %f ms' % (avg_time_ms))
        logger.info('Average time (total) taken by network is : %f ms' % (total_time_ms))
        if macs is not None:
            logger.info(f'ptflops: Flops: {macs}, Params: {params}')
        else:
            logger.info('ptflops: not available')
    else:
        if flops is not None:
            print('FLOPs = ' + str(flops.total()/1000**3) + ' G')
        else:
            print('FLOPs: fvcore not available')
        print('Total params: ' + str(total_params))
        print('Average time (per run) taken by network is : %f ms' % (avg_time_ms))
        print('Average time (total) taken by network is : %f ms' % (total_time_ms))
        if macs is not None:
            print(f'ptflops: Flops: {macs}, Params: {params}')
        else:
            print('ptflops: not available')

def demo(device, val_loader, network, out_dir):
    with torch.no_grad():
        valloader = tqdm(val_loader)
        valloader.set_description_str('Demo')
        for step, (ori_image, ll_image, _, name) in enumerate(valloader):
            ori_image, ll_image = ori_image.to(device), ll_image.to(device)
            if ll_image.shape[2]%2 != 0:
                ll_image = ll_image[:, :, :-1, :]
            if ll_image.shape[3]%2 != 0:
                ll_image = ll_image[:, :, :, :-1]
            LLIE_image, gamma, out_g, intersection, llie = network(ll_image, get_all=True)
            LLIE_image = torch.clamp(LLIE_image, 0, 1)
            LLIE_image = LLIE_image.permute(0, 2, 3, 1).cpu().detach().numpy()
            LLIE_image = img_as_ubyte(LLIE_image[0])
            save_img((os.path.join(out_dir, name[0])), LLIE_image)

if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
