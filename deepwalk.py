from cogdl import options, experiment
from cogdl.data import Graph
from cogdl.datasets import NodeDataset
import numpy as np
import torch
from torch_geometric.datasets import NELL, Coauthor

NUM_TRAIN_DATA_PER_LABEL_CLASS = 20
NUM_VAL_DATA = 500
NUM_TEST_DATA = 1000

class NellNodeDataset(NodeDataset):
    def __init__(self, path="./data/nell_cogdl"):
        self.path = path
        super(NellNodeDataset, self).__init__(path, scale_feat=False, metric="accuracy")

    def process(self):
        """You need to load your dataset and transform to `Graph`"""
        dataset = NELL(root='./data/nell')


        data = Graph(
            x=dataset.data.x.to_dense(),
            edge_index=dataset.data.edge_index,
            y=dataset.data.y,
            train_mask=dataset.data.train_mask,
            val_mask=dataset.data.val_mask,
            test_mask=dataset.data.test_mask)

        return data

class CoauthorNodeDataset(NodeDataset):
    def __init__(self, path="./data/coauthor_new"):
        self.path = path
        super(CoauthorNodeDataset, self).__init__(path, scale_feat=False, metric="accuracy")

    def process(self):
        """You need to load your dataset and transform to `Graph`"""
        dataset = Coauthor(root='./data/coauthor', name='CS')

        num_nodes = dataset.data.y.shape[0]

        idx_train = np.array([], dtype=np.int32)
        labels_copy = np.copy(dataset.data.y.numpy())
        label_set = set(labels_copy)
        # sample 20 training dataset for each label class
        for label in label_set:
            sample = (np.where(labels_copy == label)[0]
                      [0:NUM_TRAIN_DATA_PER_LABEL_CLASS])
            idx_train = np.concatenate([idx_train, sample])
            labels_copy[sample] = -1
        # construct validation dataset
        idx_val = np.where(labels_copy != -1)[0][0: NUM_VAL_DATA]
        labels_copy[idx_val] = -1
        # construct testing dataset
        idx_test = np.where(labels_copy != -1)[0][0: NUM_TEST_DATA]

        # add masks to dataset
        train_mask = np.zeros(num_nodes, dtype=np.bool8)
        val_mask = np.zeros(num_nodes, dtype=np.bool8)
        test_mask = np.zeros(num_nodes, dtype=np.bool8)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_val] = True

        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)

        # import pdb; pdb.set_trace()

        data = Graph(
            x=dataset.data.x,
            edge_index=dataset.data.edge_index,
            y=dataset.data.y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask)

        return data


if __name__ == "__main__":
    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)

    if args.dataset[0] == 'nell':
        experiment(dataset=NellNodeDataset(), model='deepwalk', args=args)
    elif args.dataset[0] == 'coauthor':
        experiment(dataset=CoauthorNodeDataset(), model='deepwalk', args=args)
    else:
        experiment(dataset=args.dataset[0], model='deepwalk', args=args)
