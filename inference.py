import numpy as np
import torch
import argparse
import os
import scipy.io as sio
from datasets import HyperX_test, get_dataset
from utils_HSI import seed_worker, test_metrics
from models.vit_pytorch import VisionTransformer as vits
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='DAMS ')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Houston/')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='the name of the source dir')
parser.add_argument('--patch_size', type=int, default=13)
parser.add_argument('--lr', type=float, default=1e-3,
                    help="Learning rate, set by the model if not specified.")
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64,
                    help="Batch size (optional, if absent will be set by the model")

parser.add_argument('--training_sample_ratio', type=float, default=1,
                    help='training sample ratio')

parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--seed', type=int, default=323,
                    help='random seed ')

parser.add_argument('--flip_augmentation', action='store_true', default=False,
                    help="Random flips (if patch_size > 1)")
parser.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
parser.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, bbox_inches = 'tight', pad_inches = 0, dpi = dpi)


def main():

    result_record = {
        'Confusion_matrix': [],
        'OA': 0,
        'TPR': 0,
        'F1scores': 0,
        'kappa': 0
    }


    root = os.path.join(args.save_path, args.target_name +'_results')
    log_dir = os.path.join(root, 'lr_'+ str(args.lr)+
                           '_pt'+str(args.patch_size))
    
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_worker(args.seed) 

    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                            args.data_path)
    hyperparams = vars(args)
    num_classes = gt_tar.max()
    N_BANDS = img_tar.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})
    r = int(hyperparams['patch_size']/2)+1
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))

    hyperparams_train = hyperparams.copy()
    hyperparams_train['radiation_augmentation'] = True

    test_dataset_noise = HyperX_test(img_tar, gt_tar, **hyperparams_train)
    test_dataset = HyperX_test(img_tar, gt_tar, **hyperparams)

    test_loader_noise = torch.utils.data.DataLoader(test_dataset_noise,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    shuffle=False)
    
    model = vits(
        img_size=args.patch_size,
        in_c = N_BANDS,
        patch_size=1,
        embed_dim=512,
        depth=args.layers,
        num_heads=4,
        representation_size=None,
        num_classes=num_classes
    )


    model_path = './results/Houston13_DAMS/lr_0.001_pt13_batch_size64_lambda1_1.0_lambda2_1.0_Layers_2_temperature_0.1/source_classifier.pkl'
    save_weight = torch.load(model_path,map_location='cuda:0' )
    model.load_state_dict(save_weight['model'])

    model = model.to(args.gpu)

    model.eval()
    results = []
    Row, Column = [], []


    for i, (x, y, center_coords) in tqdm(enumerate(test_loader), total=len(test_loader)):
    
        x, y = x.to(args.gpu), y.to(args.gpu)

        with torch.no_grad():

            _, pred_x = model(x)

            result = np.argmax(pred_x.cpu().numpy(), axis=-1) + 1
            results.extend(result)

            Row.extend([coord for coord in center_coords[0]])
            Column.extend([coord for coord in center_coords[1]])


    size = gt_tar.shape

    prediction = np.zeros((size[0],size[1]))
    for i, pred_label in enumerate(results):
        center_x, center_y = Row[i], Column[i]
        prediction[center_x, center_y] = pred_label
    
    prediction = prediction[r:-r, r:-r]
    
    best_G = prediction
    hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
    for i in range(best_G.shape[0]):
        for j in range(best_G.shape[1]):
            if best_G[i][j] == 0:
                hsi_pic[i, j, :] = [0, 0, 0]
            if best_G[i][j] == 1:
                hsi_pic[i, j, :] = [0, 0, 1]
            if best_G[i][j] == 2:
                hsi_pic[i, j, :] = [0, 1, 0]
            if best_G[i][j] == 3:
                hsi_pic[i, j, :] = [0, 1, 1]
            if best_G[i][j] == 4:
                hsi_pic[i, j, :] = [1, 0, 0]
            if best_G[i][j] == 5:
                hsi_pic[i, j, :] = [1, 0, 1]
            if best_G[i][j] == 6:
                hsi_pic[i, j, :] = [1, 1, 0]
            if best_G[i][j] == 7:
                hsi_pic[i, j, :] = [0.5, 0.5, 1]
            if best_G[i][j] == 8:
                hsi_pic[i, j, :] = [1, 0.2, 0.5]
            if best_G[i][j] == 9:
                hsi_pic[i, j, :] = [0.3, 0.4, 1]
            if best_G[i][j] == 10:
                hsi_pic[i, j, :] = [1, 1, 0.6]
            if best_G[i][j] == 11:
                hsi_pic[i, j, :] = [0.1, 0.5, 1]
            if best_G[i][j] == 12:
                hsi_pic[i, j, :] = [0.6, 0.3, 0.8]

    path_prediction = log_dir + '/' + args.target_name + '.mat'
    sio.savemat(path_prediction, {'prediction': prediction})
    
    gt = gt_tar[r:-r, r:-r]
    best_GT = gt
    GT_pic = np.zeros((best_GT.shape[0], best_GT.shape[1], 3))
    for i in range(best_G.shape[0]):
        for j in range(best_G.shape[1]):
            if best_GT[i][j] == 0:
                GT_pic[i, j, :] = [0, 0, 0]
            if best_GT[i][j] == 1:
                GT_pic[i, j, :] = [0, 0, 1]
            if best_GT[i][j] == 2:
                GT_pic[i, j, :] = [0, 1, 0]
            if best_GT[i][j] == 3:
                GT_pic[i, j, :] = [0, 1, 1]
            if best_GT[i][j] == 4:
                GT_pic[i, j, :] = [1, 0, 0]
            if best_GT[i][j] == 5:
                GT_pic[i, j, :] = [1, 0, 1]
            if best_GT[i][j] == 6:
                GT_pic[i, j, :] = [1, 1, 0]
            if best_GT[i][j] == 7:
                GT_pic[i, j, :] = [0.5, 0.5, 1]
            if best_GT[i][j] == 8:
                GT_pic[i, j, :] = [1, 0.2, 0.5]
            if best_GT[i][j] == 9:
                GT_pic[i, j, :] = [0.3, 0.4, 1]
            if best_GT[i][j] == 10:
                GT_pic[i, j, :] = [1, 1, 0.6]
            if best_GT[i][j] == 11:
                GT_pic[i, j, :] = [0.1, 0.5, 1]
            if best_GT[i][j] == 12:
                GT_pic[i, j, :] = [0.6, 0.3, 0.8]

    plt.imshow(prediction, cmap='jet')
    plt.axis('off')
    plt.savefig('./results/' + args.target_name + '_predection' + '.png', bbox_inches = 'tight', pad_inches = 0, dpi = 600)
    plt.show()

    plt.imshow(gt, cmap='jet')
    plt.axis('off')
    plt.savefig('./results/' + args.target_name + '_gt' + '.png', bbox_inches = 'tight', pad_inches = 0, dpi = 600)
    plt.show()

    pred = prediction.reshape(-1)

    gt = gt.reshape(-1)
    results = test_metrics(
            pred,
            gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=num_classes,
        )
    print(results['Confusion_matrix'], '\n', 'TPR:', np.round(results['TPR'] * 100, 2), '\n', 'OA:',
                results['Accuracy'], '\n', 'kappa:', results["Kappa"])
    
    result_record['Confusion_matrix']= '{:}'.format(results['Confusion_matrix'])
    result_record['OA'] = '{:.2f}'.format(results['Accuracy'])
    result_record['TPR'] = '{:}'.format(np.round(results['TPR'] * 100, 2))
    result_record['F1scores'] = '{:}'.format(results["F1_scores"])
    result_record['kappa'] = '{:.4f}'.format(results["Kappa"])

    with open(log_dir + '/results.txt', 'w+') as f:
        for key, value in result_record.items():
            f.write(f"{key}: {value}\n")
    f.close()


if __name__ == '__main__':
    main()