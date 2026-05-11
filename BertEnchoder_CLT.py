import torch
import torch.nn as nn
import math
class BERTmbeadings(nn.Module):
    def __init__(self, vocab_size, embedDim, numPosEmbeading, padIdx = None):
        super().__init__()
        self.Embead = nn.Embedding(vocab_size, embedDim, padIdx)
        self.PosEmbead = nn.Embedding(numPosEmbeading, embedDim)
        self.Norm = nn.LayerNorm(embedDim)
    def forward(self, tokens, pos = None):
        if pos == None :
            pos = torch.arange(tokens.shape[1], device=tokens.device)
            pos = pos.unsqueeze(0)
            pos = pos.expand(tokens.shape[0], -1)

        res = self.Embead(tokens) + self.PosEmbead(pos)
        return self.Norm(res)

class EncoderBLock(nn.Module):
    def __init__(self, embedDim, num_heads, dropout = 0.1):
        super().__init__()
        self.embedDim = int(int(embedDim / num_heads) * num_heads)
        self.Attantion = nn.MultiheadAttention(self.embedDim, num_heads, dropout= dropout, batch_first= True)
        self.MLP = nn.Sequential(
            nn.Linear(self.embedDim, self.embedDim * 4),
            nn.GELU(),
            nn.Linear(self.embedDim * 4, self.embedDim)
        )
        self.Norm1 = nn.LayerNorm(self.embedDim)
        self.Norm2 = nn.LayerNorm(self.embedDim)
        self.Drop1 = nn.Dropout(dropout,inplace=True)
        self.Drop2 = nn.Dropout(dropout,inplace=True)
    def forward(self, inp, mask = None, scale = (1.0,1.0)):
        a_o, _ = self.Attantion(inp, inp , inp, key_padding_mask=mask)
        l1 = self.Norm1(inp + self.Drop1(a_o).mul(scale[0]))
        m_o = self.MLP(l1)
        op = self.Norm2(l1 + self.Drop2(m_o).mul(scale[1]))
        return op

class MaskedLangModeling(nn.Module):
    def __init__(self,vocabSize : int, embedDim : int, embeddings = None,mul_fac = 0):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim),
        )
        self.Pread = nn.Linear(embedDim, vocabSize)
        # mul_fac != mul_fac is a nan check
        if mul_fac == 0 or math.isnan(mul_fac):
            self.mul_fac = 1 / math.sqrt(embedDim)
        else:
            self.mul_fac = mul_fac
        if embeddings is not None:
            self.Pread.weight = embeddings.weight
    def forward(self, seq):
        out = self.MLP(seq)
        return self.Pread(out * self.mul_fac)

class NextSentPred(nn.Module):
    def __init__(self, embedDim : int, idx = 0):
        super().__init__()
        self.idx = idx
        self.MLP = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.LayerNorm(embedDim),
            nn.GELU(),
            nn.Linear(embedDim, 1)
        )
    def forward(self, seq):
        return self.MLP(seq[:, self.idx, :]).view(-1)


class EncoderOnly(nn.Module):
    def __init__(self, vocabSize=123, embedDim = 192, numHeads = 12,
                 numLayers = 12, numPosEmbeading=128, padIdx = 0,
                 Same_Weights_Out_EMbed = True,cfgDict = None,
                 mlm_mul_fac = 0, gamma = 1):
        super().__init__()
        if cfgDict == None:
            embedDim = int(int(embedDim / numHeads) * numHeads)
            self.cfgDict = {"vocabSize" : vocabSize,
                            "embedDim" : embedDim,
                            "numHeads" : numHeads,
                            'numLayers' : numLayers,
                            "numPosEmbeading" : numPosEmbeading,
                            "padIdx" : padIdx,
                            "Same_Weights_Out_EMbed" : Same_Weights_Out_EMbed,
                            "mlm_mul_fac" : mlm_mul_fac,
                            "gamma": gamma
                            }
        else:
            self.cfgDict = cfgDict
            vocabSize = self.cfgDict["vocabSize"]
            embedDim = self.cfgDict["embedDim"]
            numHeads = self.cfgDict["numHeads"]
            numPosEmbeading = self.cfgDict["numPosEmbeading"]
            padIdx = self.cfgDict["padIdx"]
            numLayers = self.cfgDict["numLayers"]
            Same_Weights_Out_EMbed = bool(self.cfgDict["Same_Weights_Out_EMbed"])
            gamma = self.cfgDict["gamma"]
            if "mlm_mul_fac" in self.cfgDict:
                mlm_mul_fac = self.cfgDict["mlm_mul_fac"]
            else:
                mlm_mul_fac = 0
                self.cfgDict["mlm_mul_fac"] = mlm_mul_fac
        self.Embead = BERTmbeadings(vocabSize, embedDim, numPosEmbeading, padIdx)
        self.Blocks = nn.ModuleList()
        for i in range(numLayers):
            self.Blocks.append(EncoderBLock(embedDim, numHeads))
        if Same_Weights_Out_EMbed:
            self.MLM = MaskedLangModeling(vocabSize, embedDim,self.Embead.Embead,mul_fac=mlm_mul_fac)
        else:
            self.MLM = MaskedLangModeling(vocabSize, embedDim,mul_fac=mlm_mul_fac)

        self.Update_Out_Scale_List_No_Check(gamma)

    def Update_Out_Scale_List(self, gamma):
        if gamma == self.cfgDict["gamma"]:
            return
        self.Update_Out_Scale_List_No_Check(gamma)
        
    def Update_Out_Scale_List_No_Check(self, gamma):
        self.cfgDict['gamma'] = gamma
        device = next(self.parameters()).device
        l = torch.arange(1,self.cfgDict["numLayers"] * 2 + 1,step=1,
                         dtype=torch.float,device=device)
        l = l ** (-float(gamma))
        l = l.reshape(-1, 2)
        self.register_buffer("Out_Scale_List", l)

    def forward(self, tokens, mask = None):
        x = self.Embead(tokens)
        for Block, scale in zip(self.Blocks, self.Out_Scale_List):
            x = Block(x, mask, scale)

        MLMOut = self.MLM(x)
        return MLMOut

    def addBlock(self,num_blocks = 1,freeze_rest = True, copy_prev = True, noise_fac = 0.15):
        if freeze_rest:
            self.requires_grad_(False)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        prev_block_data = []
        if copy_prev:
            for block in self.Blocks:
                state_dict = block.state_dict()
                std_dict = {}
                for key in state_dict.keys():
                    if state_dict[key].is_floating_point():
                        std_dict[key] = torch.std(state_dict[key], unbiased=False).item()
                    else:
                        std_dict[key] = 0.0
                prev_block_data.append((state_dict, std_dict))

        prev_len = len(prev_block_data)
        for i in range(num_blocks):
            block = EncoderBLock(self.cfgDict["embedDim"],self.cfgDict['numHeads'],0.1)
            if copy_prev and prev_len > 0:
                new_state_dict = {}
                idx = i % prev_len
                for key in prev_block_data[idx][1].keys():
                    if prev_block_data[idx][0][key].is_floating_point():
                        new_state_dict[key] = torch.randn_like(prev_block_data[idx][0][key]) * noise_fac
                        new_state_dict[key] *= prev_block_data[idx][1][key]
                        new_state_dict[key] += prev_block_data[idx][0][key].clone().detach()
                    else:
                        new_state_dict[key] = prev_block_data[idx][0][key].clone().detach()
                block.load_state_dict(new_state_dict)
            block = block.to(device)
            block.requires_grad_(True)
            self.Blocks.append(block)
        self.cfgDict["numLayers"] += num_blocks
        self.Update_Out_Scale_List_No_Check(self.cfgDict["gamma"])