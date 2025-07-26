import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import time
import clip
from datasets import HyperX, HyperX_src, get_dataset, get_dataset_src
from utils_HSI import sample_gt, seed_worker, AvgrageMeter, metrics
from models.Hyperclip import HyperCLIP

parser = argparse.ArgumentParser(description='DAMS')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Houston/')
parser.add_argument('--source_name', type=str, default='Houston13',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='the name of the test dir')
parser.add_argument("--CLIP", default="ViT-B/16", help="CLIP model") # ViT-B/16 ViT-B-32
parser.add_argument('--patch_size', type=int, default=13)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--training_sample_ratio', type=float, default=1.0,
                    help='training sample ratio')
parser.add_argument('--lambda1', default=1.0,  type=float)
parser.add_argument('--lambda2', default=1.0,  type=float)
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
seed_worker(args.seed)

def evaluate(net, val_loader, gpu):
    ps = []
    ys = []
    with torch.no_grad():
        for i,(x1, y1) in enumerate(val_loader):
            x1 = x1.to(gpu)
            y1 = y1.to(gpu)
            y1 = y1 - 1
            _, p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.cpu().numpy())

        ps = np.concatenate(ps)
        ys = np.concatenate(ys)
        acc = np.mean(ys==ps)*100
        results = metrics(ps, ys, n_classes=ys.max() + 1) 
    return acc, results

def main():
    train_res = {
        'best_epoch': 0,
        'best_acc': 0,
        'Confusion_matrix': [],
        'OA': 0,
        'TPR': 0,
        'F1scores': 0,
        'kappa': 0
    }
    root = os.path.join(args.save_path, args.source_name+'_DAMS')
    log_dir = os.path.join(root, 'lr_'+ str(args.lr)+
                           '_pt'+str(args.patch_size) + '_batch_size' + str(args.batch_size) + '_lambda1_' + str(args.lambda1)+ '_lambda2_' + str(args.lambda2)
                           + '_Layers_' + str(args.layers) + '_temperature_' + str(args.temperature))
    
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    hyperparams = vars(args)
    s = ''
    for k, v in args.__dict__.items():
        s += '\t' + k + '\t' + str(v) + '\n'

    f = open(log_dir + '/settings.txt', 'w+')
    f.write(s)
    f.close()
    
    
    img_src, aug_img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, _, _ = get_dataset_src(args.source_name,
                                                            args.data_path)
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})
    
    img_tar, gt_tar, _, IGNORED_LABELS, _, _ = get_dataset(args.target_name,
                                                            args.data_path)
    
    r = int(hyperparams['patch_size']/2)+1
    
    img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
    aug_img_src=np.pad(aug_img_src,((r,r),(r,r),(0,0)),'symmetric')
    gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0)) 
    
    train_gt_src, _, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    
    img_src_con, aug_img_src_con, train_gt_src_con = img_src, aug_img_src, train_gt_src
    
    hyperparams_train = hyperparams.copy()
    hyperparams_train['flip_augmentation'] = True
    hyperparams_train['radiation_augmentation'] = True
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_dataset = HyperX_src(img_src_con, aug_img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True)
    
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                    pin_memory=True,
                                    batch_size=hyperparams['batch_size']) 
    
    clip_model, _ = clip.load(args.CLIP, device=args.gpu)
    clip_model_params =clip_model.state_dict()
    embed_dim = clip_model_params ["text_projection"].shape[1]
    context_length = clip_model_params ["positional_embedding"].shape[0]
    vocab_size = clip_model_params ["token_embedding.weight"].shape[0]
    transformer_width = clip_model_params ["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers=3
    
    model = HyperCLIP(args, embed_dim=embed_dim, clip_model=clip_model, classnames=LABEL_VALUES_src, dtype=clip_model.dtype, img_size=args.patch_size, 
                      inchannel=N_BANDS, layers=args.layers, vision_patch_size=2, num_classes=num_classes, context_length=context_length, vocab_size=vocab_size, 
                      transformer_width=transformer_width, transformer_heads=transformer_heads, transformer_layers=transformer_layers)
    model_dict = model.state_dict()
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in clip_model_params:
            del clip_model_params[key]
    clip_model_params = {k: v for k, v in clip_model_params.items() if k in model_dict and 'visual' not in k.split('.')}
    
    model_dict.update(clip_model_params)
    model.load_state_dict(model_dict)
    model = model.to(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_metric = AvgrageMeter()
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t1 = time.time()
        for it, (data, aug_data, label) in enumerate(train_loader):
            data, label = data.to(args.gpu), label.to(args.gpu)
            aug_data = aug_data.to(args.gpu)
            label = label - 1
            optimizer.zero_grad()
            text = torch.cat([clip.tokenize(f'A hyperspectral image of {LABEL_VALUES_src[k]}.').to(k.device) for k in label])
            sim_loss, loss_scl, loss_ce, loss_trans, loss_e= model(image=data, aug_img=aug_data, text=text, label=label)
            loss = sim_loss + loss_scl + loss_ce + args.lambda1*loss_trans - args.lambda2*loss_e
            loss_metric.update(loss, data.shape[0])
            loss.backward()
            optimizer.step()
        
        t2 = time.time()
        print("[TRAIN EPOCH {}] loss={} Time={:.2f}".format(epoch, loss_metric.get_avg(), t2 - t1))
        
        model.eval()
        taracc, results = evaluate(model.visual, test_loader, args.gpu)
        
        if best_acc < taracc:
            best_acc = taracc
            torch.save({'model':model.visual.state_dict()}, os.path.join(log_dir, f'source_classifier.pkl'))
            train_res['best_epoch'] = epoch
            train_res['best_acc'] = '{:.2f}'.format(best_acc)
            train_res['Confusion_matrix'] = '{:}'.format(results['Confusion_matrix'])
            train_res['OA'] = '{:.2f}'.format(results['Accuracy'])
            train_res['TPR'] = '{:}'.format(np.round(results['TPR'] * 100, 2))
            train_res['F1scores'] = '{:}'.format(results["F1_scores"])
            train_res['kappa'] = '{:.4f}'.format(results["Kappa"])

        print(f'[TRAIN EPOCH {epoch}] taracc {taracc:2.2f} best_taracc {best_acc:2.2f}')
        
    with open(log_dir + '/train_log.txt', 'w+') as f:
        for key, value in train_res.items():
            f.write(f"{key}: {value}\n")
    f.close()
        

if __name__ == "__main__":

    main()