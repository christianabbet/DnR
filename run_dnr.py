import os
import argparse

from tqdm import tqdm
from dataset import ApplyOnKey, DnRDataset
from torch.utils.data import DataLoader
from dnr import CAE_DNR, NonParametricClassifier, ANsDiscovery, Criterion

from torchvision import transforms
import torch.optim as optim
import torch


def model_func(model, optimizer, data_loader, npc, ANs_discovery, criterion, round, n_samples):

    model.train()

    loss = None
    tqdm_iterator = tqdm(data_loader, desc='train')
    for batch_idx, data in enumerate(tqdm_iterator):

        data_in = data['image_he'].cuda().float()
        data_out = data['image'].cuda().float()
        index = data['idx_overall'].cuda().long()

        if 'image_pairs' in data:
            data_in_p = data['image_pairs_he'].cuda().float()
            data_out_p = data['image_pairs'].cuda().float()
            index_p = data['idx_overall'].cuda().long() + n_samples

            data_in = torch.cat((data_in, data_in_p), 0)
            data_out = torch.cat((data_out, data_out_p), 0)
            index = torch.cat((index, index_p), 0)

        optimizer.zero_grad()

        # calculate loss and metrics
        res = model.calculate_objective(data_in, data_out, index, npc, ANs_discovery, criterion, round)

        # Parse new loss and add to old one
        _loss = dict([(k, v.item()) for k, v in res.items()])
        loss = dict([(k, loss[k]+_loss[k]) for k in loss]) if loss is not None else _loss

        tqdm_iterator.set_postfix(dict([(k, v/(batch_idx+1)) for k, v in loss.items()]))

        # backward pass
        res['loss'].backward()
        # step
        optimizer.step()


def main():

    parser = argparse.ArgumentParser(description='Run CNN classifier on Kather 19')
    parser.add_argument('--output', dest='output', type=str,
                        default='.', help='Output path')
    parser.add_argument('--db', dest='db', type=str,
                        default='samples.npy', help='Path to database')
    parser.add_argument('--device', dest='device', type=str,
                        default='cuda', choices=["cpu", "cuda"], help='Which device to use')
    parser.add_argument('--pretrained', dest='pretrained', type=str,
                        default='dnr_model_state', help='Path to pretrained model base name (.pth, _ans.pth, _npc.pth)')
    args = parser.parse_args()

    batch_size = 32
    n_channels = 2
    max_round = 4
    max_epoch = 20

    # Create dataset and sampler$
    ds_train = DnRDataset(
        filename=args.db,
        transform=transforms.Compose([
            ApplyOnKey(on_key='image_he', func=transforms.ToTensor()),
            ApplyOnKey(on_key='image', func=transforms.ToTensor()),
            ApplyOnKey(on_key='image_pairs_he', func=transforms.ToTensor()),
            ApplyOnKey(on_key='image_pairs', func=transforms.ToTensor()),
        ]),
    )

    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0)

    print('Build model with n_channels: {} ...'.format(n_channels))
    model = CAE_DNR(pretrained=True, n_channels=n_channels, hidden_dimension=512).cpu()
    npc = NonParametricClassifier(512, 2*len(ds_train)).cpu()
    ANs_discovery = ANsDiscovery(2*len(ds_train)).cpu()
    criterion = Criterion()
    optimizer = optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.999))
    model_save = os.path.join(args.output, '{}_model'.format(model.__class__.__name__))

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    start_round = 0  # start for iter 0 or last checkpoint iter
    round = start_round

    if os.path.exists(args.pretrained + ".pth"):
        print('Loading pretrained', args.pretrained + ".pth")
        model.load_state_dict(torch.load(args.pretrained + ".pth"))
        npc.load_state_dict(torch.load(args.pretrained + "_npc.pth"))
        ANs_discovery.load_state_dict(torch.load(args.pretrained + "_ans.pth"))

    # At each round we increase the entropy threshold to select NN
    while round < max_round:

        # variables are initialized to different value in the first round
        is_first_round = True if round == start_round else False

        if not is_first_round:
            ANs_discovery.update(round, npc, None)

        # start to train for an epoch
        epoch = start_epoch if is_first_round else 0
        while epoch < max_epoch:
            print('Round: {}/{}, epoch: {}/{}'.format(round+1, max_round, epoch+1, max_epoch))

            # 1. Train model (1 epoch)
            model_func(model=model, optimizer=optimizer, data_loader=dl_train,
                       npc=npc, ANs_discovery=ANs_discovery, criterion=criterion,
                       round=round, n_samples=len(ds_train))

            torch.save(model, model_save+"_{}_{}.pth".format(round, epoch))
            torch.save(npc, model_save+"_npc_{}_{}.pth".format(round, epoch))
            torch.save(ANs_discovery, model_save+"_ans_{}_{}.pth".format(round, epoch))

            epoch += 1

        # log best accuracy after each iteration
        round += 1


if __name__ == '__main__':
    main()

