import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from zero_one import Sig_Linear2 as Sigmoid_alt



class Position_encode(nn.Module):
    """This helps learn the positional encoding for our graph."""

    def __init__(self, G=None, N=None, d=None,
                pos_neigh={}, neg_samples=[], deg_pos_neigh={},
                deg_neg_samples={}, deg_vec=None,
                init=False, seed=10, scale=False, p_neg=0.6):
        super(Position_encode, self).__init__()

        self.p_neg = p_neg # the percentage of negative neighbours to use
        self.scale = scale
        self.my_sigmoid = Sigmoid_alt()
        self.N = len(G.nodes()) if N is None else N

        self.deg_loss = nn.MSELoss()

        if d is None:
            d = int(np.ceil(np.log2(N)))
            print(f"Dimension d={d}")

        self.d = d
        self.P = nn.Parameter(torch.randn(self.N,d))
        # if init:
        #     self.visualize_positive_entries(self.P, d, seed)
        self.W_d = nn.Parameter(torch.randn(d))

        self.G = G
        self.deg_vec = deg_vec
        self.pos_neigh = pos_neigh
        self.neg_samples = neg_samples
        self.deg_pos_neigh = deg_pos_neigh
        self.deg_neg_samples = deg_neg_samples


    def visualize_positive_entries(self, A, d, seed):
        # Ensure A is a PyTorch tensor
        if not torch.is_tensor(A):
            raise ValueError("Input must be a PyTorch tensor")

        # Create B with -1 where A is negative and +1 where A is positive
        B = torch.where(A < 0, torch.tensor([-1.0]), torch.tensor([1.0]))

        # Sum the last dimension of B to get an N by 1 vector
        sum_B = B.sum(dim=-1, keepdim=True)

        # Convert the result to a NumPy array for scatter plotting
        result_np = sum_B.numpy()

        # Scatter plot
        plt.plot(range(result_np.shape[0]), result_np[:, 0])#, marker='o', s=20)
        plt.xlabel("Node")
        plt.ylabel("Sum of parity")
        plt.title("Plot of Vector parity")

        plt.savefig(f'blacknwhite_d={d}_seed={seed}.jpg')
        print("\nParameter initialization sign is saved...\n")


    def degree_loss(self, Z):
        """Difference between degree vectors"""
        deg_prime = torch.matmul(Z, self.W_d) #shape: N x 1
        return self.deg_loss(deg_prime , torch.from_numpy(self.deg_vec).to(deg_prime.device).float())
        # return torch.norm(deg_prime - torch.from_numpy(self.deg_vec).to(deg_prime.device))


    def hamming_dist(self, zi, zp):
        """Computes hamming distance using absolute value"""
        return (zi - zp).abs().sum(dim=-1)


    def softmax_formula(self, Z, i, pos_neigh, neg_samples, deg=False):
        """
            This is an implementation of the softmax-based equation
            used in supervised contrastive learning.
        """
        if len(pos_neigh) == 0:
            pos_neigh = [i]
            if not deg:
                self.pos_neigh[i] = pos_neigh
            else:
                self.deg_pos_neigh[i] = pos_neigh
        neg_samples = list(neg_samples)
        num_unique_pos_neigh = len(pos_neigh) * 1.0 #make it a float
        sum_neg = torch.exp(self.hamming_dist(Z[i], Z[neg_samples])).sum()
        log_pos = self.hamming_dist(Z[i], Z[pos_neigh])
        log_sum_neg = torch.log(sum_neg)
        pos_min_neg = log_pos.sum()/num_unique_pos_neigh - log_sum_neg

        return -pos_min_neg


    def softmax_approx(self, Z, i, pos_neigh, neg_samples):
        """This equation approximates the softmax using sigmoid and regression."""

        sum_neg = torch.log( torch.sigmoid( self.hamming_dist(Z[i], Z[neg_samples]))).sum() #constant
        log_pos = torch.log( torch.sigmoid( self.hamming_dist(Z[i], Z[pos_neigh]))) # len(pos_neigh) x 1
        sum_of_logs = (log_pos - sum_neg).sum()
        num_unique_pos_neigh = len(set(pos_neigh)) * 1.0 #make it a float
        avg_pos_node = -sum_of_logs/num_unique_pos_neigh

        return avg_pos_node


    def get_non_intersecting_neg_neigh(self, i):
        """
            We want that the negative samples for the degree distribution
            does not contain nodes that are direct neighbors.
        """
        free_negatives = self.deg_neg_samples#.difference(self.G.neighbors(i))
        return list(free_negatives)


    def contrast_adj_n_degDist(self, Z, nodes):
        """
            Learn embeddings based on hamming distance and supervised contrastive loss.
            The purpose of this is to use the adjacency graph to learn the embeddings
            of vectors so they are close together as needed.
        """

        total = 0
        deg_dist_total = 0

        for i in nodes:
            if self.scale:
                # node distribution
                total += self.softmax_approx(Z, i, pos_neigh=self.pos_neigh[i],
                                                neg_samples=self.neg_samples)
                # Degree distribution
                deg_dist_total += self.softmax_approx(Z, i, pos_neigh=self.deg_pos_neigh[i],
                                                neg_samples=self.get_non_intersecting_neg_neigh(i))
            else:
                # node distribution
                if self.pos_neigh is None:
                    # Do this if when computing all neighbours in advance exhausts memory
                    pos_neigh = list(self.G.neighbors(i))
                    if len(pos_neigh) == 0: # this is done for solitary nodes
                        pos_neigh = [i]
                    neg_samples = set(self.G.nodes()).difference(list(self.G.neighbors(i)) + [i])
                    neg_samples = random.sample(neg_samples, int(len(neg_samples)*self.p_neg))
                else:
                    pos_neigh = self.pos_neigh[i]
                    neg_samples = self.neg_samples[i]

                total += self.softmax_formula(Z, i, pos_neigh=pos_neigh,
                                                neg_samples=neg_samples)
                del pos_neigh, neg_samples # save memory

                # Degree distribution
                if self.deg_neg_samples is None:
                    nodes_with_degX = self.deg_pos_neigh
                    deg_pos_neigh = nodes_with_degX[self.G.degree[i]]
                    if len(deg_pos_neigh) == 0:
                        deg_pos_neigh = [i] # For isolated nodes
                    deg_neg_samples = set(self.G.nodes()).difference(nodes_with_degX[self.G.degree[i]])
                    deg_neg_samples = random.sample(deg_neg_samples, int(len(deg_neg_samples)*self.p_neg))
                else:
                    deg_pos_neigh = self.deg_pos_neigh[i]
                    deg_neg_samples = self.deg_neg_samples[i]
                    
                    
                deg_dist_total += self.softmax_formula(Z, i, pos_neigh=deg_pos_neigh,
                                                neg_samples=deg_neg_samples, deg=True)
                del deg_pos_neigh, deg_neg_samples, nodes_with_degX # save memory


        return total, deg_dist_total



    def forward(self, selected_nodes=None, test=False, deg=False):
        """Implements the proposed algorithm"""

        if selected_nodes is not None:
            nodes = selected_nodes.cpu().numpy()

        if test:
            Z = self.my_sigmoid(self.P)

            if deg:
                L_deg = self.degree_loss(Z)
                L_adj, L_deg_dist = self.contrast_adj_n_degDist(Z, nodes)
                return Z, L_adj, L_deg_dist, L_deg

            return Z, None, None, None

        Z = self.my_sigmoid(self.P)

        L_deg = self.degree_loss(Z)

        L_adj, L_deg_dist = self.contrast_adj_n_degDist(Z, nodes)

        return L_adj, L_deg_dist , L_deg
