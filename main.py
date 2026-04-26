import os
import yaml
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from types import SimpleNamespace
from torch_geometric.loader import DataLoader

# 假设这些是你项目内部的模块
from training.trainer import Trainer, EarlyStopping
from models.main_model import DTA_GCN
from models.prediction_head import PredictModule
from data_processing.data_loader import *
from utils.metrics import model_evaluate
# 【关键修改】导入你的辅助函数
from utils.helpers import save_result_grouped_by_epoch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 1. 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['env']['seed'])

    # 统一设备定义
    device = torch.device(f"cuda:{cfg['env']['cuda']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 【关键修改】构建 args 对象以兼容 save_result_grouped_by_epoch
    # 将 yaml 配置扁平化或映射到需要的字段
    train_cfg = cfg.get('train', {})
    model_cfg = cfg.get('model', {})

    args = SimpleNamespace(
        lr=train_cfg.get('lr', 0.001),  # 确保 config.yaml 里有 lr，或者给默认值
        batch_size=train_cfg.get('batch_size', 32),
        epochs=train_cfg.get('epochs', 100),
        tau=model_cfg.get('tau', 0.5),
        lam=model_cfg.get('lam', 0.5),
        # 如果有其他需要的参数，继续在这里添加
    )

    # 【关键修改】生成唯一的 Run ID (例如基于时间戳)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Run ID: {run_id}")

    # 2. 数据加载
    ds = cfg['paths']['dataset']
    print(f"Loading dataset: {ds}...")
    affinity_mat = load_data(ds)
    train_set,test_set, aff_graph, soft_pos = process_data(affinity_mat, ds, cfg['train']['pos_threshold'])

    drug_graphs_data = torch.load(f'data/{ds}/pt/g1g2drug_graphs.pt')
    target_graphs_data = torch.load(f'data/{ds}/pt/target_graphs.pt')

    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, collate_fn=collate)
    #val_loader = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=cfg['train']['batch_size'], shuffle=False, collate_fn=collate)

    drug_ldr = DataLoader(drug_graphs_data, batch_size=aff_graph.num_drug, collate_fn=collate)
    target_ldr = DataLoader(target_graphs_data, batch_size=aff_graph.num_target, collate_fn=collate)

    # 3. 初始化模型
    m_cfg = cfg['model']
    m_cfg['ns_dims'][0] = aff_graph.num_drug + aff_graph.num_target + 2

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

    VAL_INTERVAL = 25
    # 定义保存文件的路径 (可以根据需要修改文件名)
    #RESULT_FILE_PATH = "training_results.xlsx"
    RESULT_FILE_PATH = cfg['paths']['log_xlsx']
    print(f"Starting training for {cfg['train']['epochs']} epochs. Validation every {VAL_INTERVAL} epochs.")
    print(f"Results will be saved to: {RESULT_FILE_PATH}")

    for epoch in range(1, cfg['train']['epochs'] + 1):
        train_loss = trainer.train_one_epoch(device, epoch, train_loader, drug_ldr, target_ldr, aff_graph, soft_pos)

        if train_loss is not None:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f}", end="")
        else:
            print(f"Epoch {epoch:3d} | Train Loss: N/A", end="")

        # --- 验证与保存逻辑 ---
        if epoch % VAL_INTERVAL == 0:
            y_true, y_pred = trainer.test(test_loader, drug_ldr, target_ldr, aff_graph, soft_pos)
            metrics_list = model_evaluate(y_true, y_pred)

            # 【关键修改】解析指标
            # 假设 model_evaluate 返回 [MSE, R2, CI, ...] 顺序，请根据你 utils.metrics 的实际返回顺序调整索引
            # 通常顺序可能是：MSE, RMSE, MAE, R2, CI, Pearson 等
            # 这里假设：index 0=MSE, index 1=R2 (或类似), index 2=CI
            # 为了安全，建议打印一下 metrics_list 确认顺序，或者修改 model_evaluate 返回字典
            # 如果 model_evaluate 返回的是列表，这里做一个假设性的映射：
            mse_val = metrics_list[0]
            r2_val = metrics_list[1] if len(metrics_list) > 1 else 0.0
            ci_val = metrics_list[2] if len(metrics_list) > 2 else 0.0

            # 如果 model_evaluate 返回的是字典，则直接用：
            # metrics_dict = model_evaluate(...)
            # mse_val = metrics_dict['MSE'] ...

            print(f" | Val MSE: {mse_val:.6f}, R2: {r2_val:.6f}, CI: {ci_val:.6f}")

            # 构造用于保存的数据字典
            epoch_result = {
                "epoch": epoch,
                "MSE": mse_val,
                "R2": r2_val,
                "CI": ci_val,
                "Split": "Validation",  # 标记是验证集
                "Train_Loss": train_loss if train_loss is not None else 0.0
            }

            # 调用保存函数
            save_result_grouped_by_epoch(epoch_result, args, run_id, file_path=RESULT_FILE_PATH)

            if stopper(mse_val):
                save_dir = cfg['paths']['save_dir']
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                torch.save(predictor.state_dict(), os.path.join(save_dir, "best_pred.pt"))
                print(">>> Best Model Saved")

            if stopper.early_stop:
                print("Early Stopping Triggered")
                break
        else:
            print("")

    # 5. 最终测试
    print("\n--- Final Testing ---")
    best_model_path = os.path.join(cfg['paths']['save_dir'], "best_model.pt")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for final test.")
    else:
        print("No best model saved, using current weights.")

    test_true, test_pred = trainer.test(test_loader, drug_ldr, target_ldr, aff_graph, soft_pos)
    final_metrics_list = model_evaluate(test_true, test_pred)

    # 解析最终测试指标
    mse_test = final_metrics_list[0]
    r2_test = final_metrics_list[1] if len(final_metrics_list) > 1 else 0.0
    ci_test = final_metrics_list[2] if len(final_metrics_list) > 2 else 0.0

    print(f"Final Test Results -> MSE: {mse_test:.6f}, R2: {r2_test:.6f}, CI: {ci_test:.6f}")

    # 【关键修改】保存最终测试结果
    final_result = {
        "epoch": cfg['train']['epochs'],  # 或者标记为 -1 表示最终
        "MSE": mse_test,
        "R2": r2_test,
        "CI": ci_test,
        "Split": "Test",  # 标记是测试集
        "Train_Loss": 0.0  # 最终测试没有当轮train loss
    }

    # 追加保存到同一个文件，这样一次运行的所有记录都在一个表里
    save_result_grouped_by_epoch(final_result, args, run_id, file_path=RESULT_FILE_PATH)
    print(f"Final test results appended to {RESULT_FILE_PATH}")


if __name__ == '__main__':
    main()