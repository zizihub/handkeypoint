import torch


def main(name):
    ori_ckpts = torch.load(name, map_location='cpu')
    if 'net' in ori_ckpts:
        ckpts = ori_ckpts['net']
    else:
        ckpts = ori_ckpts
    ################# modify ##################
    new_ckpts = {}
    for k, v in ckpts.items():
        # k = k.replace('model.', '')
        if not 'head' in k:
            new_ckpts[k] = v
        else:
            if 'segmentation_head.' in k:
                nk = k.replace('segmentation_head.', 'head.')
            else:
                nk = k
            new_ckpts[nk] = v
            print('#### {} ====> {}: {}'.format(k, nk, v.shape))
    # print(new_ckpts.keys())
    ###########################################
    if 'net' in ori_ckpts:
        ori_ckpts['net'] = new_ckpts
    else:
        ori_ckpts = new_ckpts
    torch.save(ori_ckpts, name)
    print('##### new checkpoint saved')


def redirect():
    import os
    from collections import defaultdict
    import shutil
    src = '../log/Hair Seg'
    move_list = defaultdict(list)
    for folder in os.listdir(src):
        for f in os.listdir(os.path.join(src, folder)):
            if f.endswith('.log') and not 'inference' in f:
                date = f[-23:-13]
                os.makedirs(os.path.join(
                    os.path.dirname(src), date), exist_ok=True)
                move_list[date].append(os.path.join(src, folder))
                break
    for date, values in move_list.items():
        for v in values:
            base = os.path.basename(v)
            dst = os.path.join(os.path.dirname(src), date, base)
            print(v, '===>', dst)
            shutil.move(v, dst)


if __name__ == '__main__':
    main('../log/SKY/2021-11-08/resnet18__Unetx16+oc(256, 128, 64, 32, 16)__sz512_dcfl/resnet18__Unetx16+oc(256, 128, 64, 32, 16)__sz512_dcfl_best.pth')
