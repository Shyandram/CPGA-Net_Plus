from utils import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori_data_path', type=str, default='ori',  help='Origin image path')
parser.add_argument('--haze_data_path', type=str, default='haze',  help='Haze image path')

parser.add_argument('--val_ori_data_path', type=str, help='Validation origin image path', default='')
parser.add_argument('--val_haze_data_path', type=str,help='Validation haze image path', default='')

parser.add_argument('--dataset_type', type=str,  help='...', default='LOL-v1')

parser.add_argument('--net_name', type=str, default='nets')

parser.add_argument('--use_gpu', type=str2bool, default=False, help='Use GPU')
parser.add_argument('--gpu', type=int, default=-1, help='GPU id')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--num_workers', type=int, default=0, help='Number of threads for data loader, for window set to 0')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--print_gap', type=int, default=50, help='number of batches to print average loss ')

parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')

parser.add_argument('--output_name', type=str,  help='...')
parser.add_argument('--sample_output_folder', type=str, default='samples',  help='Validation haze image path')
parser.add_argument('--model_dir', type=str, default='models',  help='...')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--ckpt', type=str, default=None,  help='weights/enhance_expe.pkl')
parser.add_argument('--video_dir', type=str,  help='...')

parser.add_argument('--efficient', type=str2bool, default=False, help='Use efficient (DGF) version')
parser.add_argument('--plus', type=str2bool, default=False, help='Use plus version')
parser.add_argument('--n_cpblks', type=int, default=2, help='Number of CP blocks (for plus version)')
parser.add_argument('--n_resch', type=int, default=8, help='Number of channels in res blocks')
parser.add_argument('--n_gammach', type=int, default=16, help='Number of channels in gamma blocks')
parser.add_argument('--n_IAAFch', type=int, default=16, help='Number of channels in IAAF blocks')
parser.add_argument('--is_cpgablks', type=str2bool, default=False, help='Use gamma & IAAF in CP blocks')

parser.add_argument('--knowledge_distillation', type=int, default=0, help='knowledge distillation')
parser.add_argument('--ckpt_teacher', type=str, help='teacher model weight path (knowledge distillation)')

parser.add_argument('--lambda_l1', type=float, default=1, help='L1 loss weight')
parser.add_argument('--lambda_per', type=float, default=1e-2, help='Perceptual loss weight')
parser.add_argument('--lambda_hdr', type=float, default=1, help='HDR loss weight')
parser.add_argument('--lambda_ssim', type=float, default=1e-1, help='SSIM loss weight')
parser.add_argument('--lambda_distill', type=float, default=1e-1, help='Distillation loss weight')
parser.add_argument('--lambda_hm', type=float, default=0, help='Histogram matching loss weight')
parser.add_argument('--hm_bins', type=int, default=256, help='Histogram matching loss bins')

parser.add_argument('--skipSST', type=str2bool, default=False, help='Skip self-supervised training')
parser.add_argument('--SSTepochs', type=int, default=20, help='Number of epochs for self-supervised training')

parser.add_argument('--crop_size', type=int, default=None, help='Crop size for training')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
