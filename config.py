from utils import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori_data_path', type=str, default='ori',  help='Origin image path')
parser.add_argument('--ll_data_path', type=str, default='ll',  help='ll image path')

parser.add_argument('--val_ori_data_path', type=str, help='Validation origin image path', default='')
parser.add_argument('--val_ll_data_path', type=str,help='Validation ll image path', default='')

parser.add_argument('--dataset_type', type=str,  help='...', default='LOL-v1')

parser.add_argument('--net_name', type=str, default='nets')

parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPU')
parser.add_argument('--gpu', type=str, default="auto", help='GPU specification: single ID (e.g., "0"), multiple IDs (e.g., "0,1,2"), "auto" for best GPU, "all" for all GPUs, or -1 for CPU')
parser.add_argument('--multi_gpu', type=str2bool, default=False, help='Enable multi-GPU training with DataParallel')
parser.add_argument('--sync_bn', type=str2bool, default=False, help='Use synchronized batch normalization for multi-GPU training')
parser.add_argument('--gpu_memory_fraction', type=float, default=0.9, help='Fraction of GPU memory to use')
parser.add_argument('--min_gpu_memory', type=float, default=4.0, help='Minimum required GPU memory in GB for auto selection')

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

parser.add_argument('--efficient', action='store_true', default=False, help='Use efficient (DGF) version/ BDSF version')
parser.add_argument('--plus', action='store_true', default=False, help='Use plus version')
parser.add_argument('--n_cpblks', type=int, default=2, help='Number of CP blocks (for plus version)')
parser.add_argument('--n_resch', type=int, default=8, help='Number of channels in res blocks')
parser.add_argument('--n_gammach', type=int, default=16, help='Number of channels in gamma blocks')
parser.add_argument('--n_IAAFch', type=int, default=16, help='Number of channels in IAAF blocks')
parser.add_argument('--is_cpgablks', action='store_true', default=True, help='Use gamma & IAAF in CP blocks')
parser.add_argument('--conv_type', type=str, default='conv', help='Convolution type: conv or ds_conv')
parser.add_argument('--block_type', type=str, default='ResBlock', help="Block type: 'ResBlock', 'InvertedResBlock', or 'ConvNeXtBlock'")

parser.add_argument('--bn', action='store_true', help='Use Batch Normalization')

parser.add_argument('--iaaf_type', type=str, default='IAAF', help='IAAF type: IAAF or IAAF_masking')
parser.add_argument('--iaaf_ablation', type=str, default=None, help='IAAF ablation: none, no_masking, or no_scoring')
parser.add_argument('--iaaf_scoring', type=str, default='22', help='IAAF scoring: 2:2')\

parser.add_argument('--knowledge_distillation', type=int, default=0, help='knowledge distillation')
parser.add_argument('--ckpt_teacher', type=str, help='teacher model weight path (knowledge distillation)')

parser.add_argument('--lambda_l1', type=float, default=1, help='L1 loss weight')
parser.add_argument('--lambda_per', type=float, default=1e-2, help='Perceptual loss weight')
parser.add_argument('--lambda_hdr', type=float, default=1, help='HDR loss weight')
parser.add_argument('--lambda_ssim', type=float, default=1e-1, help='SSIM loss weight')
parser.add_argument('--lambda_distill', type=float, default=1e-1, help='Distillation loss weight')
parser.add_argument('--lambda_hm', type=float, default=0, help='Histogram matching loss weight')
parser.add_argument('--hm_bins', type=int, default=256, help='Histogram matching loss bins')

parser.add_argument('--skipSST', action='store_true', default=False, help='Skip self-supervised training')
parser.add_argument('--SSTepochs', type=int, default=20, help='Number of epochs for self-supervised training')
parser.add_argument('--overview_epoch', type=int, default=0, help='Number of epochs for overview warnup')

parser.add_argument('--crop_size', type=int, default=None, help='Crop size for training')

parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')

parser.add_argument('--wandb_key', type=str, default=None, help='WandB API key for logging')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
