import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch import nn
from .SGCN import SignedGCN
import wandb

from lightgcn.utils import get_logger, logging_conf 
from Sgcn.utils import edge_split_by_sign


logger = get_logger(logger_conf=logging_conf)


def build(n_node: int, weight: str = None, **kwargs):
    model = SignedGCN(**kwargs)
    if weight:
        if not os.path.isfile(path=weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info(" model")
        state = torch.load(f=weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def run(
    model: nn.Module,
    num_nodes : int,
    train_data: dict,
    test_data : dict,   
    valid_data: dict = None,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    model_dir: str = None,
    patience: int = 10
):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    os.makedirs(name=model_dir, exist_ok=True)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_auc, best_epoch = 0, -1
    early_stopping_counter = 0
    #print(num_nodes)
    input_feature = torch.rand((num_nodes,32),device="cuda")
    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        # optimizer.zero_grad()
        node_embedding, loss = train(model,train_data,optimizer,input_feature)
        
        
        # VALID
        auc = validate(valid_data,model,node_embedding)
        
        
        wandb.log(dict(valid_auc_epoch=auc))
        
        
        if auc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, auc)
            best_auc, best_epoch = auc, e
            torch.save(obj= {"model": model.state_dict(), "epoch": e + 1},
                       f=os.path.join(model_dir, f"best_model.pt")) 
            
            with torch.no_grad():
                print("t")
                pred = model.discriminate(node_embedding,edge_index=test_data["edge"])
                pred = pred.flatten().detach().cpu().numpy()
                os.makedirs(name="./submit/", exist_ok=True)
                write_path = os.path.join("./submit/", "submission_t.csv")
                pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=write_path, index_label="id")
                
           
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                break
    torch.save(obj={"model": model.state_dict(), "epoch": e + 1},
               f=os.path.join(model_dir, f"last_model.pt"))
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def train(model: nn.Module, train_data: dict, optimizer: torch.optim.Optimizer, embedding:torch.Tensor):
    model.train()
    optimizer.zero_grad()
    pos_edge, neg_edge = edge_split_by_sign(train_data)
    next_embedding = model(embedding,pos_edge,neg_edge)
    loss = model.loss_bce(next_embedding,train_data)
    # backward
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        auc,acc = model.test(next_embedding,train_data)
    
    logger.info("TRAIN LOSS : %.4f, Train AUC : %.4f", loss.item(), auc)
    return next_embedding, loss


def validate(valid_data: dict, model: nn.Module, embedding:torch.Tensor):
    model.eval()
    with torch.no_grad():
        pos_edge,neg_edge = edge_split_by_sign(valid_data)
        auc,acc = model.test(embedding,valid_data)
        
    logger.info("VALID AUC : %.4f", auc)
    return auc





'''
def inference(model: nn.Module, train_data: dict, test_data:dict, output_dir: str, n_nodes:int):
    model.eval()
    embedding = torch.rand((n_nodes,32),device="cuda")
    with torch.no_grad():
        pos_edge, neg_edge = edge_split_by_sign(train_data)
        next_embedding = model(embedding,pos_edge,neg_edge)
        pred = model.discriminate(next_embedding,edge_index=test_data["edge"])
    
    logger.info("Saving Result ...")
    pred = pred.flatten().detach().cpu().numpy()
    print(pred.flatten())
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=write_path, index_label="id")
    logger.info("Successfully saved submission as %s", write_path)
'''