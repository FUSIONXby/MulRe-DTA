# preprocess_graphs.py
import torch
import json
from collections import OrderedDict
from data_processing.graph_builder import get_drug_molecule_graph, get_target_molecule_graph
from utils.datasets import GraphDatasetDrug,GraphDatasetTarget
def preprocess_and_save(dataset):
    # 药物
    # print("Processing drug graphs...")
    # drug_graphs_dict = get_drug_molecule_graph(
    #     json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    # drug_graphs_Data = DTADataset(graphs_dict=drug_graphs_dict, dttype="drug")
    # torch.save(drug_graphs_Data, f"data/{dataset}/pt/drug_graphs.pt")

    print("Processing g1g2drug graphs...")
    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    drug_graphs_Data = GraphDatasetDrug(graphs_dict=drug_graphs_dict, dttype="g1g2drug")
    torch.save(drug_graphs_Data, f"data/{dataset}/pt/g1g2drug_graphs.pt")
    # 靶点
    print("Processing target graphs...")
    target_graphs_dict = get_target_molecule_graph(
        json.load(open(f'data/{dataset}/targets.txt'), object_pairs_hook=OrderedDict), dataset)
    target_graphs_Data = GraphDatasetTarget(graphs_dict=target_graphs_dict, dttype="target")
    torch.save(target_graphs_Data, f"data/{dataset}/pt/target_graphs.pt")

    print(f"Graphs for {dataset} saved successfully!")

if __name__ == "__main__":
    preprocess_and_save("davis")  # 或者你的数据集名
