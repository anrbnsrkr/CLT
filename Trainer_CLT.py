import torch
import tqdm
import pickle

from sklearn.utils import shuffle
from torch.nn import MSELoss


def Load_Data(pth):
    with open(pth, 'rb') as f:
        d = pickle.load(f)
    return d

def Dict_Shuffle(dic):
    keys = list(dic.keys())
    arrays = [dic[k] for k in keys]
    arrays = shuffle(*arrays)
    return {key: array for key, array in zip(keys, arrays)}

def Cvt_Dict_To_TensorDataset(dictonary, batch_size = 20, lebels = None, shuffle = True, drop_last = False,r = (0, float('inf'))):
    if shuffle:
        dictonary = Dict_Shuffle(dictonary)
    if lebels == None:
        lebels = list(dictonary.keys())
    lis = [dictonary[k][r[0]: int(min(len(dictonary[k]), r[1]))] for k in lebels]
    dataset = torch.utils.data.TensorDataset(*lis)
    data_loder = torch.utils.data.DataLoader(dataset=dataset, batch_size= batch_size,shuffle=shuffle,drop_last=drop_last)
    return data_loder

class Trainer():
    def get_model(self):
        return self.model
    def __init__(self, model : torch.nn.Module, forward_func, loss_class, optimizer_class, scheduler_class = None,
                 load_data_func = None, accuracy_func = None, save_func = None,
                 user_metrics_class = None):
        self.model = model

        self.forward = forward_func

        self.loss_calc = loss_class
        self.optimizer = optimizer_class
        self.scheduler = scheduler_class
        self.calc_accu = accuracy_func
        self.load_data = load_data_func
        self.save = save_func
        self.user_metrics_class = user_metrics_class


    def new_dict(self, dic:dict, val:float = 0.00000001):
        d = dic.copy()
        for k in d.keys():
            d[k] = val
        return d

    def generate_return_dict(self, data,init_loss = 10000000.0, init_accu = 0):
        x = next(iter(data))
        device = next(self.model.parameters()).device
        with torch.no_grad():
            forward_out = self.forward(self.model, x, device)
            batch_on_device = forward_out['item_on_device']
            out = forward_out['out']
            loss = self.loss_calc(out, batch_on_device)
            accu = {}
            if self.calc_accu is not None:
                accu = self.calc_accu(out, batch_on_device)

            m_dict = {}
            for k in loss:
                m_dict[k] = init_loss
            for k in accu:
                m_dict[k] = init_accu
        return m_dict

    def run_epoch(self, data, train:bool, epoch_no, tot_epochs, disable_pbar = False, d_set_no = 0, enable_bf16 = False):
        # torch.autograd.set_detect_anomaly(True)
        if self.load_data is not None:
            data = self.load_data(data)
        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()
        return_dict = {}
        pbar = tqdm.tqdm(data, leave=True, disable=disable_pbar)
        num_mini_batchs = len(data)
        print_len = int(num_mini_batchs * 0.05)
        accu = {}
        device = next(self.model.parameters()).device
        for i, batch in enumerate(pbar):
            with torch.amp.autocast(device_type=device.type, enabled=enable_bf16, dtype=torch.bfloat16):
                forward_out = self.forward(self.model, batch, device)
                batch_on_device = forward_out['item_on_device']
                out = forward_out['out']
                loss = self.loss_calc(out, batch_on_device)
            if train:
                self.optimizer(self.model, loss,i,num_mini_batchs,epoch_no,tot_epochs, d_set_no)
                if self.scheduler is not None:
                    self.scheduler(self.model, i, num_mini_batchs, epoch_no, tot_epochs, d_set_no)
            if self.calc_accu is not None:
                accu = self.calc_accu(out, batch_on_device)

            m_dict = {**loss, **accu}
            p_fix = {}
            for k, v in m_dict.items():
                if k[0] == '_':
                    continue
                tmp = return_dict.get(k, 0.0) * i + v.item()
                tmp /= (i + 1)
                return_dict[k] = tmp
                p_fix[k] = f'{tmp:4f}'
            pbar.set_postfix(p_fix)
            if train and self.user_metrics_class is not None:
                self.user_metrics_class(self.model, return_dict, i, num_mini_batchs,epoch_no, tot_epochs)
            if disable_pbar and ((i + 1) % print_len == 0 or (i + 1) == num_mini_batchs):
                print(f"Step {i},Epoch {epoch_no}: {p_fix}")
        torch.cuda.empty_cache()
        return return_dict

    def test(self, test_data, disable_pbar = False,enable_bf16=False):
        with torch.no_grad():
            res = self.run_epoch(test_data, False, -1, -1, disable_pbar,enable_bf16=enable_bf16)
        return res

    def train(self, train_data, test_data = None, epochs = 1, disable_pbar = False,enable_bf16=False, enable_validation_bf16 = False, save_name = "Model"):
        perv_metrices_train = self.generate_return_dict(train_data)
        perv_metrices_test = None
        if test_data is not None:
            perv_metrices_test = self.generate_return_dict(test_data)
        train_history = []
        test_history = []

        for e in range(1, epochs + 1):
            if epochs > 1:
                print(f"Epoch Number: {e} / {epochs}")
            if test_data is not None:
                print('Training')
            self.optimizer.zero_grad()
            cur_metrices_train = self.run_epoch(train_data, train=True, epoch_no = e,
                                    tot_epochs = epochs ,disable_pbar = disable_pbar,
                                        enable_bf16=enable_bf16)
            train_history.append(cur_metrices_train)
            cur_metrices_test = None
            if test_data is not None:
                print("Validation")
                with torch.no_grad():
                    cur_metrices_test = self.run_epoch(test_data, train=False, epoch_no = e,
                                            tot_epochs = epochs ,disable_pbar = disable_pbar,enable_bf16 = enable_validation_bf16)
                    test_history.append(cur_metrices_test)

            if self.save is not None:
                self.save(save_name,self.model, e, epochs,cur_metrices_train, perv_metrices_train,
                          cur_metrices_test, perv_metrices_test)

            perv_metrices_train = cur_metrices_train.copy()
            if cur_metrices_test is not None:
                perv_metrices_test = cur_metrices_test.copy()

        return train_history, test_history


#########################
"Templates"
#########################

class Optimizer_Class():
    def __init__(self, optim, step_every = 1):
        self.optim = optim
        self.step_every = step_every
    def __call__(self, model, loss, i, num_mini_batchs,epoch_no,tot_epochs, d_set_no):
        l = loss["MLM_Loss"] / self.step_every
        l.backward()
        i += 1
        if i % self.step_every == 0 or i==num_mini_batchs:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optim.step()
            self.optim.zero_grad()
    def zero_grad(self):
        self.optim.zero_grad()


# class Optimizer_Class_With_Grad_Scaleing():
#     def __init__(self, optim, step_every = 1):
#         self.optim = optim
#         self.step_every = step_every
#     def __call__(self, model, loss, i, num_mini_batchs,epoch_no,tot_epochs, d_set_no):
#         l = loss["MLM_Loss"] / self.step_every
#         l.backward()
#         i += 1
#         if i % self.step_every == 0 or i==num_mini_batchs:
        # grad_norm = torch.tensor(0.0, device=l.weight.device)
        # for p in self.optim.param_groups:
        #     for q in p['params']:
        #         w_norm += q.norm()
                # # grad_norm = q.grad.norm().item()
                # if grad_norm > 0 and w_norm > 0.01:
                #     scale_fac = ((self.target * w_norm) / grad_norm)
                #     q.grad.mul_(scale_fac)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             self.optim.step()
#             self.optim.zero_grad()
#     def zero_grad(self):
#         self.optim.zero_grad()

class Optimizer_Class_L2_Level_Wise():
    def __init__(self, optim, step_every = 1,target = 1e-4):
        self.optim = optim
        self.step_every = step_every
        self.target = target
    def __call__(self, model, loss, i, num_mini_batchs,epoch_no,tot_epochs, d_set_no):
        l = loss["MLM_Loss"] / self.step_every
        i += 1

        l.backward()
        if i % self.step_every == 0 or i==num_mini_batchs:
            w_norm = torch.tensor(0.0, device=l.device)
            num_blocks = len(model.Blocks)
            for block_idx, Block in enumerate(model.Blocks):
                for q in Block.MLP.parameters():
                    w_norm += (self.target * float(block_idx + 1) / num_blocks) * q.norm()**2
                for q in Block.Attantion.parameters():
                    w_norm += (self.target * float(block_idx + 1) / num_blocks) * q.norm()**2
            w_norm.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optim.step()
            self.optim.zero_grad()
    def zero_grad(self):
        self.optim.zero_grad()

class Optimizer_Class_L2_Uniform():
    def __init__(self, optim, step_every = 1,target = 1e-4):
        self.optim = optim
        self.step_every = step_every
        self.target = target
    def __call__(self, model, loss, i, num_mini_batchs,epoch_no,tot_epochs, d_set_no):
        l = loss["MLM_Loss"] / self.step_every
        i += 1

        l.backward()
        if i % self.step_every == 0 or i==num_mini_batchs:
            w_norm = torch.tensor(0.0, device=l.device)
            for block_idx, Block in enumerate(model.Blocks):
                for q in Block.MLP.parameters():
                    w_norm = w_norm + self.target * q.norm() ** 2
                for q in Block.Attantion.parameters():
                    w_norm = w_norm + self.target * q.norm() ** 2
            w_norm.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optim.step()
            self.optim.zero_grad()
    def zero_grad(self):
        self.optim.zero_grad()

class Optimizer_Class_L2_Uniform_No_Q_K():
    def __init__(self, optim, step_every = 1,target = 1e-4):
        self.optim = optim
        self.step_every = step_every
        self.target = target
    def __call__(self, model, loss, i, num_mini_batchs,epoch_no,tot_epochs, d_set_no):
        l = loss["MLM_Loss"] / self.step_every
        i += 1
        l.backward()

        if i % self.step_every == 0 or i==num_mini_batchs:
            w_norm = torch.tensor(0.0, device=l.device)
            mul = 1.0
            for block_idx, Block in enumerate(model.Blocks):
                q_w, k_w, v_w = torch.chunk(Block.Attantion.in_proj_weight, 3, dim=0)
                q_b, k_b, v_b = torch.chunk(Block.Attantion.in_proj_bias, 3, dim=0)
                if Block.Attantion.in_proj_weight.grad is not None:
                    Block.Attantion.in_proj_weight.grad.mul_(mul)
                if Block.Attantion.in_proj_bias.grad is not None:
                    Block.Attantion.in_proj_bias.grad.mul_(mul)
                mul += 1.0
                w_norm = w_norm + self.target * (v_w.norm() ** 2)
                w_norm = w_norm + self.target * (v_b.norm() ** 2)
                for p in Block.Attantion.out_proj.parameters():
                    w_norm = w_norm + self.target * (p.norm() ** 2)
                    if p.grad is not None:
                        p.grad.mul_(mul)
                for p in Block.MLP.parameters():
                    w_norm = w_norm + self.target * (p.norm() ** 2)
                    if p.grad is not None:
                        p.grad.mul_(mul)
                mul += 1.0
            w_norm.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optim.step()
            self.optim.zero_grad()
    def zero_grad(self):
        self.optim.zero_grad()

class Loss_Class():
    def __init__(self, criterion_mlm):
        self.criterion = criterion_mlm

    def __call__(self, output, values):
        mask = output['mask']
        if mask.sum() == 0:
            return {"MLM_Loss" : torch.tensor(0.0, device=mask.device, requires_grad=True)}
        MLM = self.criterion(output['model'][mask], values[0][mask])
        return {"MLM_Loss" : MLM}

class Scheduler_Class():
    def __init__(self, scheduler, step_every = 1, call_functions = []):
        self.sched = scheduler
        self.step_every = step_every
        self.call_fuctions = call_functions
        self.call_fuctions_idx = 0
        self.call_fuctions_len = len(call_functions)
        self.total_steps = 0
    def __call__(self, model,i, num_mini_batchs,epoch_no, tot_epochs, d_set_no):
        i += 1
        if i % self.step_every == 0 or i == num_mini_batchs:
            self.sched.step()
            while self.call_fuctions_len > self.call_fuctions_idx and self.total_steps == self.call_fuctions[self.call_fuctions_idx][0]:
                self.call_fuctions[self.call_fuctions_idx][1](**self.call_fuctions[self.call_fuctions_idx][2])
                self.call_fuctions_idx += 1
                print(f"Functiction{self.call_fuctions_idx} Called")

            self.total_steps += 1

class Scheduler_Class_CLT():
    def __init__(self, scheduler, gamma_max, gamma_min, gama_scheduler_steps,
                 step_every = 1, call_functions = []):
        self.sched = scheduler
        self.step_every = step_every
        self.call_fuctions = call_functions
        self.call_fuctions_idx = 0
        self.call_fuctions_len = len(call_functions)
        self.total_steps = 0
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.gama_scheduler_steps = gama_scheduler_steps
        self.gamma_delta = float(gamma_max - gamma_min)

    def Get_New_Gamma_LinerLR(self):
        if self.total_steps >= self.gama_scheduler_steps:
            return self.gamma_min
        x = self.gamma_max - self.gamma_delta * max(0, self.total_steps) / max(1, self.gama_scheduler_steps)
        return x

    def __call__(self, model,i, num_mini_batchs,epoch_no, tot_epochs, d_set_no):
        i += 1
        if i % self.step_every == 0 or i == num_mini_batchs:
            self.sched.step()
            while self.call_fuctions_len > self.call_fuctions_idx and self.total_steps == self.call_fuctions[self.call_fuctions_idx][0]:
                self.call_fuctions[self.call_fuctions_idx][1](**self.call_fuctions[self.call_fuctions_idx][2])
                self.call_fuctions_idx += 1
                print(f"Functiction{self.call_fuctions_idx} Called")

            self.total_steps += 1
            new_gamma = self.Get_New_Gamma_LinerLR()
            model.Update_Out_Scale_List(new_gamma)



def Mask(inp:torch.tensor, mask_id:int, vocab_size : int,change_prob = 0.2,p_rand = 0.1, p_unchanged = 0.1)->(torch.tensor, torch.tensor) :
    ret = inp.clone()
    prob_mat = torch.rand(inp.shape, device=inp.device)
    un_changed = prob_mat < change_prob * p_unchanged
    p_rand = p_rand + p_unchanged
    x = prob_mat < (change_prob * p_rand)
    random = (~un_changed)  & x
    y = prob_mat < change_prob
    mask = ~x & y
    ret[mask] = mask_id
    num_random = int(random.sum().item())
    ret[random] = torch.randint(0, vocab_size, (num_random,), device=inp.device)
    return ret, y

def accuracy_func(output, values):
    mask = output['mask']
    x = ~values[1]
    is_eqal = torch.argmax(output['model'], dim= -1) ==  values[0]
    MLM = torch.sum(is_eqal[mask]) / max(1.0, torch.sum(mask))
    Overall = torch.sum(is_eqal[x]) / max(1, torch.sum(x))
    return {"MLM_Accu" : MLM, "Overall_Accu": Overall}

def save(save_name, model, e, epochs,cur_metrices_train, perv_metrices_train, cur_metrices_test, perv_metrices_test):
    torch.save({'state_dict': model.state_dict(), 'cfgDict': model.cfgDict}, f"{save_name}_{e}.pth")

def save2(save_name, model, e, epochs,cur_metrices_train, perv_metrices_train, cur_metrices_test, perv_metrices_test):
    torch.save({'state_dict': model.state_dict(), 'cfgDict': model.cfgDict}, f"{save_name}_{e}.pth")
    for b in model.Blocks:
        for p in b.parameters():
            print(p.shape,p.norm().item())
        print("_______________")

class User_Metrices:
    def __init__(self, save_duration = 0.05):
        self.save_duration = save_duration
        self.checkpoint_metrices = []
    def __call__(self, model, moving_accu, i, num_mini_batchs, epoch_no, tot_epochs):
        i+= 1
        modulo_val = int(num_mini_batchs * self.save_duration)
        if i % modulo_val == 0 or i == num_mini_batchs:
            self.checkpoint_metrices.append((i, epoch_no, dict(moving_accu)))
    def save(self,name):
        with open(f'{name}_data_logs.plk', 'wb') as f:
            pickle.dump(self.checkpoint_metrices, f)

    def accu_over_steps(self):
        prev_epech = -1
        prev_batch_idx = -1
        return_lis = []
        for m in self.checkpoint_metrices:
            if prev_epech !=  m[1] or prev_batch_idx > m[0]:
                return_lis.append(tuple(m))
                prev_epech = m[1]
                prev_m = m
            else:
                i1 = prev_m[0]
                i2 = m[0]
                prev_batch_idx = i2
                setps_m = (i2,m[1],{})
                for keys in m[2]:
                    setps_m[2][keys] = (m[2][keys] * i2 - prev_m[2][keys] * i1) / max(1, (i2 - i1))
                prev_m = m
                return_lis.append(setps_m)
        return return_lis

class Forward_Class:
    def __init__(self,mask_id = 4, change_prob = 0.2, p_rand = 0.1, p_unchanged = 0.1):
        self.change_prob = change_prob
        self.p_rand = p_rand
        self.p_unchanged = p_unchanged
        self.mask_id = mask_id
    def __call__(self, model, item, device):
        item_on_device = []
        item_on_device.append(item[0].to(device=device, dtype=torch.int64))
        item_on_device.append(item[1].to(device=device, dtype=torch.bool))

        inp, mask = Mask(item_on_device[0], self.mask_id, model.cfgDict['vocabSize'], self.change_prob, self.p_rand,self.p_unchanged)
        mask = mask & (~item_on_device[1])
        out = model.forward(inp, item_on_device[1])
        return {"out" : {"model" : out, "mask": mask}, 'item_on_device' : item_on_device}


class Forward_Class_Out_scale:
    def __init__(self,mask_id = 4, change_prob = 0.2, p_rand = 0.1, p_unchanged = 0.1):
        self.change_prob = change_prob
        self.p_rand = p_rand
        self.p_unchanged = p_unchanged
        self.mask_id = mask_id
    def __call__(self, model, item, device):
        item_on_device = []
        item_on_device.append(item[0].to(device=device, dtype=torch.int64))
        item_on_device.append(item[1].to(device=device, dtype=torch.bool))

        inp, mask = Mask(item_on_device[0], self.mask_id, model.cfgDict['vocabSize'], self.change_prob, self.p_rand,self.p_unchanged)
        mask = mask & (~item_on_device[1])
        out = model.forward_out_scale(inp, item_on_device[1])
        return {"out" : {"model" : out, "mask": mask}, 'item_on_device' : item_on_device}


class Forward_Class_vec_norm():
    def __init__(self,mask_id = 4, change_prob = 0.2, p_rand = 0.1, p_unchanged = 0.1):
        self.change_prob = change_prob
        self.p_rand = p_rand
        self.p_unchanged = p_unchanged
        self.mask_id = mask_id
    def __call__(self, model, item, device):
        item_on_device = []
        item_on_device.append(item[0].to(device=device, dtype=torch.int64))
        item_on_device.append(item[1].to(device=device, dtype=torch.bool))

        inp, mask = Mask(item_on_device[0], self.mask_id, model.cfgDict['vocabSize'], self.change_prob, self.p_rand,self.p_unchanged)
        mask = mask & (~item_on_device[1])
        out, norm = model.forward_vec_norm(inp, item_on_device[1])
        return {"out" : {"model" : out, "mask": mask, 'norm': norm}, 'item_on_device' : item_on_device}

class Loss_Class_vec_norm():
    def __init__(self, criterion_mlm):
        self.criterion = criterion_mlm
        self.criterion2 = torch.nn.MSELoss()

    def __call__(self, output, values):
        mask = output['mask']
        if mask.sum() == 0:
            return {"MLM_Loss" : torch.tensor(0.0, device=mask.device, requires_grad=True)}
        MLM = self.criterion(output['model'][mask], values[0][mask])
        i = 0.0
        norm_loss = torch.tensor(0.0, device=mask.device, requires_grad=True)
        for norm in output['norm']:
            for vals in norm:
                i += 1.0
                norm_loss = norm_loss + self.criterion2(vals[1], vals[0].mul(1 / i).detach())
        return {"MLM_Loss" : MLM, "_Norm_Loss" : norm_loss}


class Optimizer_Class_vec_norm():
    def __init__(self, optim, step_every = 1, norm_loss_fac = 0.01):
        self.optim = optim
        self.step_every = step_every
        self.norm_loss_fac = norm_loss_fac
    def __call__(self, model, loss, i, num_mini_batchs,epoch_no,tot_epochs, d_set_no):
        l = (loss["MLM_Loss"] + self.norm_loss_fac * loss['_Norm_Loss'])/ self.step_every
        l.backward()
        i += 1
        if i % self.step_every == 0 or i==num_mini_batchs:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optim.step()
            self.optim.zero_grad()
    def zero_grad(self):
        self.optim.zero_grad()
