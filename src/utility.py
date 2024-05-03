import torch
import torch.optim as optim
import os
import json
import itertools
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from NAPE import Position_encode
from pos_encode_loss import Position_encode_loss


def save_as_json(dict_list_obj, filename):
    path = os.path.join(os.getcwd(), 'pos_n_neg_neigh')
    if not os.path.exists(path):
        os.mkdir(path)

    file_path = os.path.join(path,filename)
    with open(file_path, 'w') as f:
        json.dump(dict_list_obj, f)

    print(f"Saved file: {filename}")


def read_as_json(filename, path=None):
    if path is None:
        path = os.path.join(os.getcwd(), 'pos_n_neg_neigh')
    file_path = os.path.join(path,filename)
    with open(file_path, 'r') as f:
        json_obj = json.load(f)

    return json_obj


def use_cuda(use_cpu,many=False,verbose=True):
    """
        Imports GPU if available, else imports CPU.
    """
    if use_cpu: #overrides GPU and use CPU
        device = torch.device("cpu")
        USE_CUDA = False
        if verbose:
            print("CPU used")
        return device,USE_CUDA

    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        device = torch.device("cuda" if many else "cuda:0")
        if verbose:
            print("GPU is available")
    else:
        device = torch.device("cpu")
        if verbose:
            print("GPU not available, CPU used")
    return device,USE_CUDA


def toNumpy(v,device): #This function converts the entry 'v' back to a numpy (float) data type
    if device.type == 'cuda':
        return v.detach().cpu().numpy()
    return v.detach().numpy()



def parse_args():
    '''
    Parses the scaleNAPE arguments.
    '''
    parser = argparse.ArgumentParser(description="Run scaleNAPE.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                    help='Input graph path')

    parser.add_argument('--d', dest='dimensions', type=int, default=16,
                    help='Number of dimensions. Default is 16.')

    parser.add_argument('--walk-length', type=int, default=60,
                    help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=2,
                    help='Number of walks per source. Default is 10.')

    parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')

    parser.add_argument('--p', type=float, default=1,
                    help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                    help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                    help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                    help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--filename', type=str, default="WikiCSDataset_adj.npy",
                    help='name for adjacency matrix to be converted to edgelist (default: WikiCSDataset_adj.npy)')

    parser.add_argument('-u','--use-saved-model', dest='use_saved_model', action='store_true', default=False,
                    help='use existing trained model (default: False)')

    parser.add_argument('--seed', type=int, default=10,
                    help='random seed for splitting the dataset into train and test (default: 0)')

    parser.add_argument('-i','--use-cpu', dest='use_cpu', action='store_true', default=False,
                    help='overrides GPU and use CPU unstead (default: False)')

    parser.add_argument('--many-gpu', dest='many_gpu', action='store_true', default=False,
                    help='decides whether to use many GPUs (default: False)')

    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate (default: 0.001)')

    parser.add_argument('--k1', type=float, default=1.0, help='hyperparameters for NAPE loss function (default: 1.0)')

    parser.add_argument('--k2', type=float, default=1.0, help='hyperparameters for NAPE loss function (default: 1.0)')

    parser.add_argument('--k3', type=float, default=1.0, help='hyperparameters for NAPE loss function (default: 1.0)')

    parser.add_argument('--k', type=int, default=10, help='size of negative samples to be drawn (default: 10)')

    parser.add_argument('--k_deg', type=int, default=20,
                    help='size of negative samples to be drawn for degree distribution (default: 20)')

    parser.add_argument('--checkpoint', type=str, default="checkpoint_scaledNAPE.pt",
                    help='file name of the saved model (default: checkpoint_scaledNAPE.pt)')

    parser.add_argument('--show-result', dest='show_result', action='store_true', default=False,
                    help='Skips training (default: False)')

    parser.add_argument('--no-scale', dest='scale', action='store_false', default=True,
                    help='decide if we use the scaled version of the NAPE algorithm (default: True)')

    return parser.parse_args()


def run(args, G, nodes, pos_neigh, neg_samples, deg_pos_neigh, deg_neg_samples, deg_vec):

    save_path = os.path.join(os.getcwd(), 'Saved_models')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    many_gpu = args.many_gpu #Decide if we use multiple GPUs or not
    device,USE_CUDA = use_cuda(args.use_cpu,many=many_gpu)

    #load model
    d = args.dimensions
    N = len(nodes)
    old_similar = N

    old_loss = np.inf
    model = Position_encode(G, N, d, pos_neigh, neg_samples,
                            deg_pos_neigh, deg_neg_samples, deg_vec,
                            init=True, seed=args.seed, scale=args.scale)

    if args.use_saved_model:
        # To load model
        model.load_state_dict(torch.load(os.path.join(save_path,f'd={d}_{args.checkpoint}'),map_location=device))
        print("USING SAVED MODEL!")


    if USE_CUDA: #To set it up for parallel usage of both GPUs (speeds up training)
        torch.cuda.manual_seed_all(args.seed)
        model = torch.nn.DataParallel(model) if many_gpu else model #use all free GPUs if needed
    model.to(device)

    criterion = Position_encode_loss(k1=args.k1, k2=args.k2, k3=args.k3)

    params = list(model.parameters())
    Num_Param = sum(p.numel() for p in params if p.requires_grad)
    print("Number of Trainable Parameters is about %d" % (Num_Param))

    optimizer = optim.Adam(params, lr= args.lr)

    print("Pre-training Result:")
    numbering, similar = check_result(model, device, G, nodes)
    print(f"***{similar} nodes are similar***\n")

    filename = save_as_txt(numbering, nodes, device, args.seed, args.scale, filename=args.checkpoint[:-3], name_return=True)

    # for epoch in range(args.epochs):
    if not args.show_result:
        with tqdm(range(args.epochs)) as t:
            try:
                for epoch in t:

                    t.set_description('Epoch %d' % (epoch+1))
                    model.train()
                    total_loss = 0
                    out1, out2, out3 = model(torch.tensor(nodes))
                    loss = criterion(out1.sum(), out2.sum(), out3.sum())

                    if optimizer is not None:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss = toNumpy(loss,device)


                    with torch.no_grad():
                        print(f"Epoch {epoch+1}:")
                        numbering, similar = check_result(model, device, G, nodes)
                        print(f"Total loss= {total_loss}")
                        print(f"***{similar} nodes are similar***")
                        if determine_save(total_loss, old_loss, similar, old_similar, epoch, min_runs=0):
                            old_similar = similar
                            old_loss = total_loss

                            if USE_CUDA and many_gpu:
                                torch.save(model.module.state_dict(), os.path.join(save_path,f'd={d}_{args.checkpoint}'))
                            else:
                                torch.save(model.state_dict(), os.path.join(save_path,f'd={d}_{args.checkpoint}'))

            except KeyboardInterrupt as e:
                del model
                print("\nGetting result from SAVED MODEL:")
                model = Position_encode(G, N, d, pos_neigh, neg_samples, deg_pos_neigh, deg_neg_samples, deg_vec, scale=args.scale)
                model.load_state_dict(torch.load(os.path.join(save_path,f'd={d}_{args.checkpoint}'),map_location=device))
                model.to(device)

                output = model(torch.tensor(nodes))
                loss = criterion(*output)
                total_loss = toNumpy(loss,device)

                print("\nBest Result is:\n")
                numbering, similar = check_result(model, device, G, nodes, final=True)
                print(f"Total loss= {total_loss}")
                print(f"***{similar} nodes are similar***")
                print(e)
                raise

    # Load best result
    del model
    print("\nGetting result from SAVED MODEL:")
    model = Position_encode(G, N, d, pos_neigh, neg_samples, deg_pos_neigh, deg_neg_samples, deg_vec, scale=args.scale)
    model.load_state_dict(torch.load(os.path.join(save_path,f'd={d}_{args.checkpoint}'),map_location=device))
    model.to(device)

    output = model(torch.tensor(nodes))
    loss = criterion(*output)
    total_loss = toNumpy(loss,device)

    print("\nBest Result is:\n")
    numbering, similar = check_result(model, device, G, nodes, final=True)
    print(f"Total loss= {total_loss}")
    print(f"***{similar} nodes are similar***")

    similar_nodes = None
    if args.show_result:
        similar_nodes = unique_count(numbering, get_similar_nodes=True)

    with open(filename,'w+') as f:
        f.write(f"\nFinal loss={total_loss:.2f}\n")
        f.write(f"\nFinal similar={similar}")
        f.write("\n")

    save_as_txt(numbering, nodes, device, args.seed, args.scale, filename=args.checkpoint[:-3])

    return similar_nodes

    # END algo


def determine_save(loss, old_loss, new_similar, old_similar, epoch, min_runs=15):
    """Decide if the model paramter should be saved"""
    # return (loss <= old_loss) and (new_similar <= old_similar) and (epoch > min_runs)
    if (new_similar <= old_similar) and (epoch > min_runs):
        return (loss <= old_loss) or (new_similar <= old_similar)
    else:
        return False


def check_result(model, device, G=None, nodes=[], final=False):
    """Returns the numbering of the skeletal graph"""
    model.eval()

    node_tensor = torch.tensor(nodes).to(device)
    Z, L_adj, L_deg_dist, L_deg = model(node_tensor,test=True,deg=final)
    N,d = Z.shape
    # This is because we use multiple GPU
    if not final:
        try:
            N = N//torch.cuda.device_count()
        except ZeroDivisionError:
            N = N//1
    Z = Z[:N]
    err = 0.0001
    Z = Z + err
    Z = torch.round(Z)
    two_vec = torch.zeros(d).to(device)
    for i in range(d):
        two_vec[i] = pow(2,d-1-i)
    numbers = (Z * two_vec).sum(dim=1) #shape: N
    numbers = toNumpy(numbers, device)
    # We expect the count for each
    identical = unique_count(numbers)

    if final:
        print(f"Degree Loss:{L_deg}, Node Loss:{L_adj},\nDeg dist Loss:{L_deg_dist}")

    return numbers, identical


def unique_count(numbers, get_similar_nodes=False):
    """Returns count of identical numbers."""
    num_list = numbers.tolist()
    num_set, count = np.unique(num_list, return_counts=True)
    gt_1 = np.not_equal(count,1)
    identical = count[gt_1].sum()

    if get_similar_nodes:
        iden_ID = num_set[gt_1]
        similar_nodes = [np.argwhere(numbers==id).flatten().tolist() for id in iden_ID]
        similar_nodes = list(itertools.chain(*similar_nodes))
        return set(similar_nodes)

    return identical



def save_as_txt(number, nodes, device, seed, scale, filename='betaNAPE_numbers', name_return=False):

    ts = 'scale_' if scale else ''
    filename = filename + '_' + ts + 'seed_' + str(seed) + '_device_' + str(device) + '.txt'
    if name_return:
        return filename

    print("Saving Final Result...")
    pairs = zip(nodes , number)
    with open(filename,'a') as f:
        print(*pairs, sep='\n', file=f)

    print("Final Result saved!")
