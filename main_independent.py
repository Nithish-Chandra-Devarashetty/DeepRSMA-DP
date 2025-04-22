import torch
from torch.utils.data import Dataset, DataLoader
import os
from data import RNA_dataset, Molecule_dataset, RNA_dataset_independent, Molecule_dataset_independent, WordVocab
from model import RNA_feature_extraction, GNN_molecule, mole_seq_model, cross_attention
from torch_geometric.loader import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import random
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set print options for debugging tensors
torch.set_printoptions(profile="full")

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Hyperparameters
hidden_dim = 128
EPOCH = 200
RNA_type = 'Viral_RNA_independent'
seed = 1

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)

set_seed(seed)

# Load datasets
try:
    rna_dataset = RNA_dataset(RNA_type)
    molecule_dataset = Molecule_dataset(RNA_type)
    rna_dataset_in = RNA_dataset_independent()
    molecule_dataset_in = Molecule_dataset_independent()
    logging.info("Datasets loaded successfully")
except Exception as e:
    logging.error(f"Error loading datasets: {e}")
    raise

# Custom dataset for paired RNA and molecule data
class CustomDualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        if len(self.dataset1) != len(self.dataset2):
            raise ValueError(f"Dataset lengths do not match: {len(self.dataset1)} vs {len(self.dataset2)}")

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)

# Utility function to average multiple lists
def average_multiple_lists(lists):
    return [sum(item)/len(lists) for item in zip(*lists)]

# DeepRSMA model
class DeepRSMA(nn.Module):
    def __init__(self):
        super(DeepRSMA, self).__init__()
        self.rna_graph_model = RNA_feature_extraction(hidden_dim)
        self.mole_graph_model = GNN_molecule(hidden_dim)
        self.mole_seq_model = mole_seq_model(hidden_dim)
        self.cross_attention = cross_attention(hidden_dim)

        self.line1 = nn.Linear(hidden_dim*2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)

        self.rna1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.mole1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.rna2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.mole2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.gelu = nn.GELU()

    def forward(self, rna_batch, mole_batch):
        try:
            # Get RNA features
            rna_out_seq, rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = self.rna_graph_model(rna_batch, device)
            rna_out_seq = rna_out_seq.to(device)
            rna_out_graph = rna_out_graph.to(device)
            rna_mask_seq = rna_mask_seq.to(device)
            rna_mask_graph = rna_mask_graph.to(device)
            rna_seq_final = rna_seq_final.to(device)
            rna_graph_final = rna_graph_final.to(device)

            # Get molecule features
            mole_graph_emb, mole_graph_final = self.mole_graph_model(mole_batch)
            mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, device)
            mole_graph_emb = mole_graph_emb.to(device)
            mole_graph_final = mole_graph_final.to(device)
            mole_seq_emb = [emb.to(device) for emb in mole_seq_emb]
            mole_mask_seq = mole_mask_seq.to(device)

            # Compute molecule sequence final representation
            if mole_seq_emb[-1].size(1) != mole_mask_seq.size(1):
                raise ValueError("Mismatch in mole_seq_emb and mole_mask_seq dimensions")
            mole_seq_final = (mole_seq_emb[-1] * mole_mask_seq.unsqueeze(dim=2)).mean(dim=1)
            mole_seq_final = mole_seq_final.to(device)

            # Process molecule graph embeddings
            flag = 0  # Initialize flag to 0 (removed erroneous 'ascended')
            mole_out_graph = []
            mask = []
            for i in mole_batch.graph_len:
                count_i = int(i)
                if flag + count_i > mole_graph_emb.size(0):
                    raise ValueError(f"Index out of bounds: flag={flag}, count_i={count_i}, mole_graph_emb size={mole_graph_emb.size(0)}")
                x = mole_graph_emb[flag:flag+count_i]
                if x.size(0) > 128:
                    raise ValueError(f"Graph embedding size {x.size(0)} exceeds maximum length 128")
                temp = torch.zeros((128 - x.size(0), hidden_dim), device=device)
                x = torch.cat((x, temp), 0)
                mole_out_graph.append(x)
                mask.append([1]*count_i + [0]*(128 - count_i))
                flag += count_i
            mole_out_graph = torch.stack(mole_out_graph).to(device)
            mole_mask_graph = torch.tensor(mask, dtype=torch.float, device=device)

            # Cross-attention
            context_layer, _ = self.cross_attention(
                [rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph],
                [rna_mask_seq, rna_mask_graph, mole_mask_seq, mole_mask_graph],
                device
            )

            out_rna = context_layer[-1][0].to(device)
            out_mole = context_layer[-1][1].to(device)

            # Compute lengths
            rna_seq_len = rna_out_seq.shape[1]
            rna_graph_len = rna_out_graph.shape[1]
            mole_seq_len = mole_seq_emb[-1].shape[1]
            mole_graph_len = mole_out_graph.shape[1]

            rna_len = out_rna.shape[1]
            mole_len = out_mole.shape[1]
            if rna_len != rna_seq_len + rna_graph_len:
                raise ValueError(f"Mismatch in RNA length: expected {rna_seq_len + rna_graph_len}, got {rna_len}")
            if mole_len != mole_seq_len + mole_graph_len:
                raise ValueError(f"Mismatch in molecule length: expected {mole_seq_len + mole_graph_len}, got {mole_len}")

            # Compute RNA cross representations
            if rna_mask_seq.sum(dim=1).eq(0).any():
                raise ValueError("RNA mask sequence contains zero sums")
            rna_cross_seq = ((out_rna[:, :rna_seq_len] * rna_mask_seq.unsqueeze(2)).sum(dim=1) /
                            rna_mask_seq.sum(dim=1, keepdim=True) + rna_seq_final) / 2
            rna_cross_stru = ((out_rna[:, rna_seq_len:] * rna_mask_graph.unsqueeze(2)).sum(dim=1) /
                            rna_mask_graph.sum(dim=1, keepdim=True) + rna_graph_final) / 2
            rna_cross = (rna_cross_seq + rna_cross_stru) / 2
            rna_cross = self.rna2(self.dropout(self.gelu(self.rna1(rna_cross))))

            # Compute molecule cross representations
            if mole_mask_seq.sum(dim=1).eq(0).any():
                raise ValueError("Molecule mask sequence contains zero sums")
            mole_cross_seq = ((out_mole[:, :mole_seq_len] * mole_mask_seq.unsqueeze(2)).sum(dim=1) /
                            mole_mask_seq.sum(dim=1, keepdim=True) + mole_seq_final) / 2
            mole_cross_stru = ((out_mole[:, mole_seq_len:] * mole_mask_graph.unsqueeze(2)).sum(dim=1) /
                            mole_mask_graph.sum(dim=1, keepdim=True) + mole_graph_final) / 2
            mole_cross = (mole_cross_seq + mole_cross_stru) / 2
            mole_cross = self.mole2(self.dropout(self.gelu(self.mole1(mole_cross))))

            # Final layers
            out = torch.cat((rna_cross, mole_cross), 1)
            out = self.line1(out)
            out = self.dropout(self.gelu(out))
            out = self.line2(out)
            out = self.dropout(self.gelu(out))
            out = self.line3(out)
            return out

        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            raise

# Create datasets and dataloaders
try:
    train_dataset = CustomDualDataset(rna_dataset, molecule_dataset)
    test_dataset = CustomDualDataset(rna_dataset_in, molecule_dataset_in)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=1, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False)
    logging.info("Dataloaders created successfully")
except Exception as e:
    logging.error(f"Error creating datasets or dataloaders: {e}")
    raise

# Initialize model and optimizer
try:
    model = DeepRSMA()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=6e-5, weight_decay=1e-5)
    loss_fct = torch.nn.MSELoss()
    logging.info("Model and optimizer initialized")
except Exception as e:
    logging.error(f"Error initializing model or optimizer: {e}")
    raise

# Training loop
y_pred_all = []
max_p = -1

for epoch in range(EPOCH):
    try:
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pre = model(batch[0].to(device), batch[1].to(device))
            y = batch[0].y.to(device)
            if pre.shape[0] != y.shape[0]:
                raise ValueError(f"Prediction and label batch sizes do not match: {pre.shape[0]} vs {y.shape[0]}")
            loss = loss_fct(pre.squeeze(dim=1), y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation
        with torch.no_grad():
            model.eval()
            y_label = []
            y_pred = []
            for batch_v in test_loader:
                label = Variable(torch.from_numpy(np.array(batch_v[0].y))).float().to(device)
                score = model(batch_v[0].to(device), batch_v[1].to(device))
                logits = torch.squeeze(score).cpu().numpy()
                label_ids = label.cpu().numpy()
                y_label.extend(label_ids.flatten().tolist())
                y_pred.extend(logits.flatten().tolist())

            p = pearsonr(y_label, y_pred)
            s = spearmanr(y_label, y_pred)
            rmse = np.sqrt(mean_squared_error(y_label, y_pred))
            logging.info(f'Epoch: {epoch}, PCC: {p[0]:.4f}, SCC: {s[0]:.4f}, RMSE: {rmse:.4f}')

            if max_p < p[0]:
                max_p = p[0]
                logging.info(f'Best: Epoch: {epoch}, PCC: {p[0]:.4f}, SCC: {s[0]:.4f}, RMSE: {rmse:.4f}')
                os.makedirs('save', exist_ok=True)
                torch.save(model.state_dict(), f'save/model_independent_{seed}.pth')

        model.train()

    except Exception as e:
        logging.error(f"Error during epoch {epoch}: {e}")
        raise

logging.info("Training completed successfully")