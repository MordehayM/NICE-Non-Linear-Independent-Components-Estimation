"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
import nice
import time
import pickle
import matplotlib.pyplot as plt

def train(flow, trainloader, optimizer, epoch, device, sample_shape, filename, train_loss):

        running_loss = 0
        batches = 0

        start = time.time()
        flow.train()  # set to training mode
        for batch_idx, (inputs, _) in enumerate(trainloader):
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[
                3])  # change  shape from BxCxHxW to Bx(C*H*W)
            # TODO Fill in
            optimizer.zero_grad()
            inputs = inputs.to(device)

            loss = -flow(inputs).mean()  # over batch, minus for minimize instead maximize the log_porb
            loss.backward()
            optimizer.step()
            #print(f"batch: {loss}")
            # print statistics
            running_loss += loss.item()
            batches += 1
        end = time.time()
        print(f" Train:    epoch: {epoch},\t | time:  {end - start} \n "
              f"loss: {running_loss/batches}")
        train_loss.append(running_loss/batches)


        flow.eval()  # set to inference mode
        with torch.no_grad():
            #reconstruction
            z, _ = flow.f(inputs)
            recont = flow.f_inverse(z).cpu()
            recont = recont.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])  # convet to BxCxHxW
            torchvision.utils.save_image(torchvision.utils.make_grid(recont),
                                         './reconstruction/' + filename + 'epoch%d.png' % epoch)


def test(flow, testloader, filename, epoch, sample_shape, device, test_loss):
    flow.eval()  # set to inference mode

    with torch.no_grad():
        samples = flow.sample(100).cpu()
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])  # convet to BxCxHxW
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
    # TODO full in
        running_loss = 0
        batches = 0
        for batch_idx, (inputs, _) in enumerate(testloader):
            batches += 1
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[
                3])  # change  shape from BxCxHxW to Bx(C*H*W)
            inputs = inputs.to(device)
            loss = -flow(inputs).mean()  # over batch, minus for minimize instead maximize the log_porb
            # print statistics
            running_loss += loss.item()
        print(f" Test:  epoch - {epoch} \n  loss: {running_loss / batches}")
        test_loss.append(running_loss / batches)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1. / 256.))  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
        full_dim = 1 * 28 * 28  # CxHxW
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
        full_dim = 1 * 28 * 28  # CxHxW
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%s_' % args.coupling_name \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden

    flow = nice.NICE(
        prior=args.prior,
        coupling_name=args.coupling_name,
        num_coupling=args.num_coupling,
        in_out_dim=full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,

        device=device).to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr) #, betas=[args.beta1, args.beta2], eps=args.ep, weight_decay=1)
    #my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.94)

    train_loss = []
    test_loss = []
    for epoch_idx in range(args.epochs):

        train(flow=flow, trainloader=trainloader,
                             optimizer=optimizer, epoch=epoch_idx,
                             device=device, sample_shape=sample_shape,
                            filename=model_save_filename, train_loss=train_loss)
        #my_lr_scheduler.step()
        test(flow=flow, testloader=testloader, filename=model_save_filename, epoch=epoch_idx,
         sample_shape=sample_shape, device=device, test_loss=test_loss)


    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    np.savez(f"train_loss_{args.dataset}_{args.coupling_name}", train=train_loss)
    np.savez(f"test_loss_{args.dataset}_{args.coupling_name}", test=test_loss)

    torch.save({
        'num_epoch': args.epochs,
        'model_state_dict': flow.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'prior': args.prior,
        'coupling_name': args.coupling_name,
        'mid_dim': args.mid_dim,
        'hidden': args.hidden, },
        './models/' + model_save_filename + '.tar')


    print('Checkpoint Saved')


    #save plot
    plt.plot(np.arange(1, 51), test_loss, np.arange(1, 51), train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss = - Log-likelihood')
    plt.title(f'Loss_Vs_Epoch - {args.dataset}_{args.coupling_name}')
    plt.grid(True)
    plt.legend(['Test', 'Train'], loc='upper right')
    plt.savefig(f'loss_vs_epoch_{args.dataset}_{args.coupling_name}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--num_coupling',
                        help='num of coupling layers',
                        type=int,
                        default='4')
    parser.add_argument('--coupling_name',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)



    args = parser.parse_args()
    main(args)
