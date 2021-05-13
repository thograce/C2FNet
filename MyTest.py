import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy import misc
from lib.C2FNet import C2FNet
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/C2FNet40/C2FNet-39.pth')

for _data_name in ['CAMO','CHAMELEON','COD10K']: #'CAMO','CHAMELEON','COD10K'
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/C2FNet40/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = C2FNet()
    #model = torch.nn.DataParallel(model)
    #torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)
