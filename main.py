from cProfile import label
from copy import deepcopy
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import embedding
from adjustText import adjust_text
from constants.constant import MAP_DICT, CIV_DICT, CIV_DICT_REVERSE
from models.model import win_predictor
from models.dataloader import load_data
import torch

def train_rt(model, data_train, epoch, optimizer, criterion):
    losses = 0.
    acc = 0
    model.train()
    num = 0
    for map, draft1, draft2, target in data_train:
        target = target.float()
        # if torch.cuda.is_available():
        #     map, draft1, draft2, target = map.cuda(), draft1.cuda(), draft2.cuda(), target.cuda()
        pred_rt = model.forward(map, draft1, draft2)
        pred_label = torch.argmax(pred_rt, dim=1)
        target_label = torch.argmax(target, dim=1)
        acc += torch.sum(pred_label==target_label)
        num+=map.shape[0]
        target.float()
        loss = criterion(pred_rt, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print('epoch : ', epoch, ',train losses : ', losses / len(data_train), 'accuracy :',(acc/num).item())
    return losses / len(data_train), (acc/num).item()

def eval_rt(model, data_val, epoch, criterion):
    losses = 0.
    model.eval()
    num = 0
    acc = 0
    for map, draft1, draft2, target in data_val:
        num += map.shape[0]
        target = target.float()
        # if torch.cuda.is_available():
        #     map, draft1, draft2, target = map.cuda(), draft1.cuda(), draft2.cuda(), target.cuda()
        pred_rt = model(map, draft1, draft2)
        pred_label = torch.argmax(pred_rt, dim=1)
        target_label = torch.argmax(target, dim=1)
        acc += torch.sum(pred_label==target_label)
        loss = criterion(pred_rt, target)
        losses += loss.item()
    print('epoch : ', epoch, ',eval losses : ', losses / len(data_val), 'accuracy :',(acc/num).item())

    return losses / len(data_val), (acc/num).item()

def run_rt(epochs, eval_inter, model, data_train, data_val, optimizer, criterion):
    train_loss=[]
    train_acc=[]
    val_acc=[]
    val_loss=[]
    for e in range(1, epochs + 1):
        loss, acc = train_rt(model, data_train, e, optimizer, criterion)
        torch.save(deepcopy(model.state_dict()), 'checkpoints/model_{}'.format(e))
        train_loss.append(loss)
        train_acc.append(acc)
        if e % eval_inter == 0:
            loss,acc = eval_rt(model, data_val, e, criterion)
            val_loss.append(loss)
            val_acc.append(acc)
    plt.clf()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['train loss','val loss','train acc','val acc'])
    plt.savefig('fig/loss.png')

def main_train():
    dataset_train, dataset_test = load_data('data_S14_07_10.csv',[0.8,0.2],16)
    model = win_predictor(drop_rate=0.5)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    run_rt(500,1,model, dataset_train, dataset_test, optimizer, torch.nn.BCELoss())


def load_model(e):
    device = torch.device('cpu')
    model = win_predictor(drop_rate=0.5)
    model.load_state_dict(torch.load('checkpoints/model_{}'.format(e), map_location=device, weights_only=True))
    return model



def complete_draft(model, map_played, pickA1, pickA2, pickA3, pickB1, pickB2, pickB3, pickB4,bans):
    civ_id = list(range(1,78))
    res={}
    for id in civ_id :
        map, draft1, draft2 = (torch.tensor(MAP_DICT[map_played]),
                               torch.tensor([CIV_DICT[pickA1], CIV_DICT[pickA2], CIV_DICT[pickA3], id]),
                               torch.tensor([CIV_DICT[pickB1], CIV_DICT[pickB2], CIV_DICT[pickB3], CIV_DICT[pickB3]]),)
        output = model(map, draft1, draft2)
        res[id]=output[0].item()
    res[pickA1]=None
    res[pickA2] = None
    res[pickA3] = None
    res[pickB4] = None
    res[pickB1] = None
    res[pickB3] = None
    res[pickB2] = None

    return res

def main_pred():
    device = torch.device('cpu')
    model = win_predictor(drop_rate=0.5)
    model.load_state_dict(torch.load('models/best_model', map_location=device, weights_only=True))
    get_embedding(model)
    res = complete_draft(model,'pangee standard','shaka','ambiorix','basil ii', 'cyrus', 'nader shah', 'gilgamesh', 'trajan',[])
    return res

def dict_to_sorted_table(dico):
    res = []
    for k in dico.keys():
        if dico[k] is not None:
            res.append((dico[k],CIV_DICT_REVERSE[k]))
    return res

def get_embedding(model):
    font0 = {}
    font0['size']='xx-small'
    civ_id = torch.tensor(list(range(1, 78)))
    emb = model.forward_embedding(civ_id)
    arr = emb.detach().numpy()
    texts = []
    for i in range(len(arr)):
        texts.append(plt.text(arr[i,0],arr[i,1],CIV_DICT_REVERSE[i+1],font0))
    plt.scatter(arr[:,0],arr[:,1])
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    plt.savefig('emb.png')
    plt.close()


if __name__ =='__main__':
    # main_train()
    res = main_pred()
    tab = dict_to_sorted_table(res)
    tab.sort(reverse=True)
