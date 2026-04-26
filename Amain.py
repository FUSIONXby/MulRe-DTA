import yaml
from torch_geometric.loader import DataLoader
from training.trainer import Trainer, EarlyStopping
from models.main_model import DTA_GCN
from models.prediction_head import PredictModule
from data_processing.data_loader import *
from utils.metrics import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # 1. 加载配置 (需要安装 PyYAML: pip install pyyaml)
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['env']['seed'])
    device = torch.device(f"cuda:{cfg['env']['cuda']}" if torch.cuda.is_available() else "cpu")

    # 2. 数据加载
    ds = cfg['paths']['dataset']
    affinity_mat = load_data(ds)
    train_set, test_set, aff_graph, soft_pos = process_data(affinity_mat, ds, cfg['train']['pos_threshold'])

    # 动态加载图数据
    # drug_graphs_data = GraphDatasetDrug(graphs_dict=get_drug_molecule_graph(
    #     json.load(open(f'../data/{ds}/drugs.txt'), object_pairs_hook=OrderedDict)), dttype="drug")
    drug_graphs_data = torch.load(f'data/{ds}/pt/g1g2drug_graphs.pt')
    target_graphs_data = torch.load(f'data/{ds}/pt/target_graphs.pt')

    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, collate_fn=collate)
    #val_loader = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=cfg['train']['batch_size'], shuffle=False, collate_fn=collate)

    drug_ldr = DataLoader(drug_graphs_data, batch_size=aff_graph.num_drug, collate_fn=collate)
    target_ldr = DataLoader(target_graphs_data, batch_size=aff_graph.num_target, collate_fn=collate)

    # 3. 初始化模型 (从配置读取维度)
    m_cfg = cfg['model']
    m_cfg['ns_dims'][0] = aff_graph.num_drug + aff_graph.num_target + 2  # 动态补齐

    model = DTA_GCN(
        tau=m_cfg['tau'], lam=m_cfg['lam'], ns_dims=m_cfg['ns_dims'],
        d_ms_dims=m_cfg['drug_ms_dims'], t_ms_dims=m_cfg['target_ms_dims'],
        embedding_dim=m_cfg['embedding_dim'], dropout_rate=m_cfg['edge_dropout']
    ).to(device)
    predictor = PredictModule().to(device)
    soft_pos = torch.from_numpy(soft_pos).float().to(device)

    # 4. 训练流程
    trainer = Trainer(model, predictor, cfg, device)
    stopper = EarlyStopping(patience=cfg['train']['patience'])
    best_mse = float('inf')
    device = torch.device('cuda:{}'.format(m_cfg['cuda']) if torch.cuda.is_available() else 'cpu')

    for epoch in range(1, cfg['train']['epochs'] + 1):
        trainer.train_one_epoch(device,epoch, train_loader, drug_ldr, target_ldr, aff_graph, soft_pos)

        if epoch % cfg['train']['val_interval'] == 0:
            y_true, y_pred = trainer.test(test_loader, drug_ldr, target_ldr, aff_graph, soft_pos)
            metrics = model_evaluate(y_true, y_pred)
            mse = metrics[0]
            print(f"Epoch {epoch} Val MSE: {mse:.6f}")

            if stopper(mse):  # 如果是当前最佳
                save_dir = cfg['paths']['save_dir']
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    print(f"Created directory: {save_dir}")
                best_mse = mse
                torch.save(model.state_dict(), os.path.join(cfg['paths']['save_dir'], "best_model.pt"))
                torch.save(predictor.state_dict(), os.path.join(cfg['paths']['save_dir'], "best_pred.pt"))
                print(">>> Best Model Saved")

            if stopper.early_stop:
                print("Early Stopping Triggered")
                break

    # 5. 最终测试
    model.load_state_dict(torch.load(os.path.join(cfg['paths']['save_dir'], "best_model.pt")))
    test_true, test_pred = trainer.test(test_loader, drug_ldr, target_ldr, aff_graph, soft_pos)
    print("Final Test Results:", model_evaluate(test_true, test_pred))


if __name__ == '__main__':
    main()