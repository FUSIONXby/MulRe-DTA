import torch
from torch_geometric.data import Data, Dataset,InMemoryDataset
from tqdm import tqdm
from rdkit import Chem
from torch_geometric import data as DATA
from Aload2.agumentation.graph_augmentation import *

class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, drug_ids=None, target_ids=None, y=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.process(drug_ids, target_ids, y)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, drug_ids, target_ids, y):
        data_list = []
        for i in range(len(drug_ids)):
            DTA = DATA.Data(drug_id=torch.IntTensor([drug_ids[i]]), target_id=torch.IntTensor([target_ids[i]]), y=torch.FloatTensor([y[i]]))
            data_list.append(DTA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class GraphDatasetDrug(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDatasetDrug, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, features, edge_index,lg= graphs_dict[key]
            mol = Chem.MolFromSmiles(lg)
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            ##
            # 增强图1 - 只删非环非极性键，删边概率0.2，删特征概率0.1
            edge_index_1 = selective_dropout_adj(mol, GCNData.edge_index, drop_prob=0.1,min_drop_ratio=0.1)
            x_1 = drop_feature(GCNData.x,mol,GCNData.edge_index, 0.1,0.5)
            GCNData.g1 = DATA.Data(x=x_1, edge_index=edge_index_1, batch=GCNData.batch)
            #GCNData.g1 = DATA.Data(x=x_1, edge_index=edge_index_1)

            # 增强图2 - 删边概率0.4，删特征概率0.3
            edge_index_2 = selective_dropout_adj(mol, GCNData.edge_index, drop_prob=0.3,min_drop_ratio=0.3)
            x_2 = drop_feature(GCNData.x,mol,GCNData.edge_index, 0.3,0.5)
            GCNData.g2 = DATA.Data(x=x_2, edge_index=edge_index_2, batch=GCNData.batch)
            #GCNData.g2 = DATA.Data(x=x_2, edge_index=edge_index_2)
            ##
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GraphDatasetTarget(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDatasetTarget, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, features, edge_index = graphs_dict[key]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
