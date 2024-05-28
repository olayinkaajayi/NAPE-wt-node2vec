'''
Reference implementation of NAPE algorithm.

Author: Olayinka Ajayi

Date: 23rd January 2024.
'''
import os
import json
import itertools
import random
import numpy as np
import torch
import networkx as nx
import node2vec
from utility import parse_args, run, save_as_json, read_as_json

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	# Because the graph may have nodes not in the connected component
	# we wish to capture that using the complement of the range of the maximum node ID.
	singleton_nodes = set(range(max(G.nodes())+1)).difference(G.nodes())
	G.add_nodes_from(singleton_nodes)

	return G, singleton_nodes



def convert_to_edgelist(adj_file='WikiCSDataset_adj.npy', folder='/dcs/large/u2034358/'):

	filename, ext = adj_file.split('.')
	print(f"\nWorking on the {filename} Graph dataset")

	if ext == 'npy':
		print('\nInput is Numpy array.')
		with open(folder+adj_file,'rb') as f:
			A = np.load(f,allow_pickle=True)

		print(f"A: shape= {A.shape}")
		G = nx.from_numpy_array(A)
		nx.write_edgelist(G, folder+filename+'.edgelist')

		return folder+filename+'.edgelist'

	elif ext == 'json':
		print('\nInput is json file.')
		with open('WikiCS/'+adj_file) as json_file:
			graph_data = json.load(json_file)


		G = nx.Graph()
		G.add_nodes_from(list(range(len(graph_data['links']))))
		edge_list = list(itertools.chain(
								*[[(i, nb) for nb in nbs]
									for i,nbs in enumerate(graph_data['links'])
								]
								)
							)
		G.add_edges_from(edge_list)
		nx.write_edgelist(G, folder+filename+'.edgelist')

		return folder+filename+'.edgelist'

	elif ext == 'edgelist':
		print('\nInput is edge list.')
		return folder+adj_file

	else:
		assert ext in ['npy', 'edgelist', 'json'], 'Input file should be edgelist, json or numpy array.'


def get_negative_samples(args, G, sorted_nodes=[]):
	"""Generate negative samples for approximating the softmax with logistic regression."""
	sorted_lst = sorted( list(G.degree()) )
	degree_vec = np.array( list( dict(sorted_lst).values() ) , dtype=float)
	if args.scale:
		degree_prob = degree_vec/degree_vec.sum()
		neg_samples = np.random.choice(a=sorted_nodes, size= args.k, replace=False, p=degree_prob)
	else:
		neg_samples = None if args.memory_issue else [set(sorted_nodes).difference(list(G.neighbors(node)) + [node]) for node in sorted_nodes]

	return neg_samples, degree_vec


def get_nodes_with_degX(degree_vec):
	unique_deg, count_deg = np.unique(degree_vec, return_counts=True)
	nodes_with_degX = {deg: np.where(degree_vec==deg)[0].tolist() for deg in unique_deg}
	return nodes_with_degX

def get_neg_samples_deg_dist(G, args, nodes=[], nodes_with_degX={}, scale=False):
	if scale:
		return set(random.sample(nodes,args.k_deg))
	else:
		return None if args.memory_issue else {node:set(nodes).difference(nodes_with_degX[G.degree[node]]) for node in nodes}


def get_necessary_objects(args, nx_G, nodes_with_degX, nodes, scale=False):
	"""To avoid recomputing the random walks, we save them once for each seed."""

	if scale:

		attachment = f'_wl-{args.walk_length}-nw-{args.num_walks}-p-{args.p}-q-{args.q}-seed-{args.seed}'
		dataset = args.filename.split('.')[0]
		filename_pos_neigh = f'{dataset}_pos_neigh{attachment}.json'
		filename_deg_pos_neigh = f'{dataset}_deg_pos_neigh{attachment}.json'

		path_pos_neigh = os.path.join(os.getcwd(),'pos_n_neg_neigh',filename_pos_neigh)
		path_deg_pos_neigh = os.path.join(os.getcwd(),'pos_n_neg_neigh',filename_deg_pos_neigh)

		if os.path.exists(path_pos_neigh) and os.path.exists(path_deg_pos_neigh):
			print("\nReading saved Simulated walks...")
			pos_neigh = read_as_json(filename_pos_neigh)
			deg_pos_neigh = read_as_json(filename_deg_pos_neigh)

			pos_neigh = {int(k):v for k,v in pos_neigh.items()}
			deg_pos_neigh = {int(k):v for k,v in deg_pos_neigh.items()}
		else:
			G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
			print("\nPreprocessing Transition probabilities...")
			G.preprocess_transition_probs()
			print("\nSimulating Walks...")
			pos_neigh, deg_pos_neigh = G.simulate_walks(args.num_walks, args.walk_length, nodes_with_degX)
			save_as_json(pos_neigh,filename_pos_neigh)
			save_as_json(deg_pos_neigh,filename_deg_pos_neigh)
	else:
		pos_neigh = None if args.memory_issue else {node:list(nx_G.neighbors(node)) for node in nodes}
		deg_pos_neigh = nodes_with_degX if args.memory_issue else {node:nodes_with_degX[nx_G.degree[node]] for node in nodes}

	return pos_neigh, deg_pos_neigh



def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	graph_name = convert_to_edgelist(adj_file=args.filename)
	args.input = graph_name

	nx_G, singleton_nodes = read_graph()
	nodes = sorted(list(nx_G.nodes()))
	print(f"Node len: {len(nodes)}, max node: {max(nodes)}")

	neg_samples, degree_vec = get_negative_samples(args,nx_G,nodes)

	# deg_pos_neigh: Dict(keys: degree, values: list of nodes having said degree)
	nodes_with_degX = get_nodes_with_degX(degree_vec)

	# list of negative samples (for degree graph)
	deg_neg_samples = get_neg_samples_deg_dist(nx_G, args, nodes, nodes_with_degX, args.scale)

	# pos_neigh: Dict(keys: nodes, values: merged elements of random walk)
	pos_neigh, deg_pos_neigh = get_necessary_objects(args, nx_G, nodes_with_degX, nodes, args.scale)

	print("\nRunning NAPE Algorithm...")
	similar_nodes = run(args, nx_G, nodes, pos_neigh, neg_samples, deg_pos_neigh, deg_neg_samples, degree_vec)

	if args.show_result:
		print(f"Length of singleton nodes: {len(singleton_nodes)}")
		print(f"Length of similar nodes: {len(similar_nodes)}")
		print(f"Length of similar_nodes - singleton_nodes: {len(similar_nodes.difference(singleton_nodes))}")
		print("Similar nodes that are not singleton:\n",similar_nodes.difference(singleton_nodes))

		degree_of_similar = {key: degree_vec[key] for key in similar_nodes.difference(singleton_nodes)}
		with open('degree_of_similar.json', 'w') as f:
			json.dump(degree_of_similar, f, indent=4)
		print("Last file save!")


if __name__ == "__main__":
	args = parse_args()
	#Initialize seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	main(args)
