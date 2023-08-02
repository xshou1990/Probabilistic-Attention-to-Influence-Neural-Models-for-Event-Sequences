import sys
sys.path.append("../")
import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import sklearn
import sklearn.metrics

import transformer.Constants as Constants


from preprocess.Dataset import get_dataloader
from transformer.Models_ll import Transformer
from tqdm import tqdm

    


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
#     print('[Info] Loading test data...')
#     test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    devloader = get_dataloader(dev_data, opt.batch_size, shuffle=True)
#     testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, devloader,  num_types

def load_prior(opt):
    print('[Info] Loading prior ...')
    with open(opt.prior + 'prior.pkl', 'rb') as f:
        prior = pickle.load(f)
    
    return prior



def train_epoch(model, training_data, optimizer, opt, prior):
    """ Epoch operation in training phase. """

    model.train()

    pri = torch.flatten(prior[opt.event_interest-1]).to(opt.device)
    binpri = torch.stack([1-pri,pri])
    
    num_iter = 0 # number of batches per epoch

    
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):

        num_iter += 1
        """ prepare data """
        _,_, event_type = map(lambda x: x.to(opt.device), batch)
        
        event_type_0 = torch.hstack([torch.zeros(event_type.shape[0],1).int().to('cuda'),event_type])

        """ forward """
        optimizer.zero_grad()

        output, _, relation = model(event_type_0, opt.num_samples, opt.event_interest, opt.threshold)
        

        rel = torch.flatten(relation[opt.event_interest-1]) 
        binrel = torch.stack([1-rel,rel])
        
        """ backward """
        # negative log-likelihood given influence sample 
        event_loss_ave = 0
        
        for i in range(len(output)):
            event_ll = log_likelihood(model, output[i,:,:-1,:], event_type )
            event_loss = -event_ll
            event_loss_ave += event_loss
            
        event_loss_ave = event_loss_ave/len(output)

  #     KL divergence of approx. posterior and prior
        kldiv = torch.sum(binrel.T * torch.log(binrel.T +1e-15) - binrel.T *torch.log(binpri.T +1e-15) )


#  negative ELBO loss
        loss =   event_loss_ave + kldiv 
        loss.backward()

        """ update parameters """
        optimizer.step()

    

    return kldiv, -event_loss_ave 


def eval_epoch(model, validation_data, opt, prior):
    """ Epoch operation in evaluation phase. """

    model.eval()

    pri = torch.flatten(prior[opt.event_interest-1]) 
    binpri = torch.stack([1-pri,pri])
    

    total_event_ll =0 # total loglikelihood
    num_iter = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            num_iter +=1

            """ prepare data """
            _,_, event_type = map(lambda x: x.to(opt.device), batch)
            
            event_type_0 = torch.hstack([torch.zeros(event_type.shape[0],1).int().to('cuda'),event_type])

            """ forward """
            
            output, _, _ = model(event_type_0, opt.num_samples, opt.event_interest, opt.threshold)
            
            
            
            """ compute loss """
            # negative log-likelihood conditioned on influence sample
            event_ll_ave = 0
            for i in range(len(output)):
                event_ll = log_likelihood_event(model, output[i,:,:-1,:], event_type, opt.event_interest)
                event_ll_ave += event_ll
            event_ll_ave /= opt.num_samples
            
            total_event_ll +=  event_ll_ave
            

    return  total_event_ll

def get_posterior(model, validation_data, opt, prior):
    """ Epoch operation in evaluation phase. """
    opt.batch = 20000

    model.eval()

    pri = torch.flatten(prior[opt.event_interest-1]) 
    binpri = torch.stack([1-pri,pri])
    

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):

            """ prepare data """
            _,_, event_type = map(lambda x: x.to(opt.device), batch)

            event_type_0 = torch.hstack([torch.zeros(event_type.shape[0],1).int().to('cuda'),event_type])

            """ forward """
            
            _, _, relation = model(event_type_0, opt.num_samples, opt.event_interest, opt.threshold)
            
            
            rel = torch.flatten(relation[opt.event_interest-1]) 
            print("posterior is shape {}".format(rel.shape))
    return rel   


def train(model, training_data, validation_data,  optimizer, scheduler, opt, prior, event_interest):
    """ Start training. """
    opt.event_interest = event_interest
    print(" Event interest is {}".format( opt.event_interest))
    
    best_ll = -np.inf
    best_model = deepcopy(model.state_dict())

    train_loss_list = [] # train loss
    train_ll_list = [] # train log likelihood
    valid_ll_list = [] # valid log likelihood
    impatience = 0 
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event,train_ll = train_epoch(model, training_data, optimizer, opt, prior)
        
        train_loss_list += [train_event]
        train_ll_list +=[train_ll]
        
        print('  - (Training)     KL: {kldiv: 8.4f}, loglikelihood: {ll: 8.4f} ,'
              'elapse: {elapse:3.3f} min'
              .format(kldiv=train_event, ll=train_ll, elapse=(time.time() - start) / 60))
        
    
        start = time.time()
        
        valid_ll = eval_epoch(model, validation_data, opt, prior)
        valid_ll_list += [valid_ll]
        print('  - (validation)  loglikelihood: {ll: 8.4f}'
              'elapse: {elapse:3.3f} min'
              .format( ll= valid_ll, elapse=(time.time() - start) / 60))

        start = time.time()
        
#         test_ll = eval_epoch(model, test_data, opt, prior)
#         print('  - (test)  loglikelihood: {ll: 8.4f}'
#               'elapse: {elapse:3.3f} min'
#               .format( ll= test_ll, elapse=(time.time() - start) / 60))
        
        print('  - [Info] Maximum validation loglikelihood:{ll: 8.4f} '
              .format(ll = max(valid_ll_list) ))
        

        if (valid_ll- best_ll ) < 1e-4:
            impatient += 1
            if best_ll < valid_ll:
                best_ll = valid_ll
                best_model = deepcopy(model.state_dict())
        else:
            best_ll = valid_ll
            best_model = deepcopy(model.state_dict())
            impatient = 0
          
            
        if impatient >= 5:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break
    

        
        scheduler.step()
    

    return best_model
        
def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def log_likelihood(model, data, types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    all_hid = model.linear(data)
    
    all_scores = F.log_softmax(all_hid,dim=-1)
    types_3d =  F.one_hot(types, num_classes= model.num_types+1)
  
    ll = (all_scores*types_3d[:,:,1:]) #[:,1:,:]
    
    ll2 = torch.sum(ll,dim=-1)*non_pad_mask
    ll3 = torch.mean(torch.sum(ll2,dim=-1))

    return ll3



def log_likelihood_event(model, data, types, event_interest):
    """ Log-likelihood of observing event of interest in the sequence. """


    non_pad_mask = get_non_pad_mask(types).squeeze(2)
   
    all_hid = model.linear(data)

    all_scores = F.softmax(all_hid,dim=-1)
    all_scores_event = torch.log(all_scores[:,:,event_interest-1] +1e-12)
    all_scores_nonevent = torch.log(1 - all_scores[:,:,event_interest-1] +1e-12 )

    event_log_ll = (types == event_interest) * all_scores_event
    nonevent_log_ll = (types != event_interest) * all_scores_nonevent
    ll = (event_log_ll + nonevent_log_ll)*non_pad_mask#[:,1:]
    ll2 = torch.sum(ll)

    return ll2


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()
#     parser.add_argument('-device', required=True)
    parser.add_argument('-data', required=True)
    parser.add_argument('-prior', required=True)
    
                
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.01)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-num_samples', type=int, default=2)
    parser.add_argument('-event_interest', type=int, default=1)
    parser.add_argument('-threshold', type=float, default=0.5)

    
    opt = parser.parse_args()

    # default device is CUDA
#     opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # default device is CUDA temporary
    opt.device = torch.device("cuda")


    print('[Info] parameters: {}'.format(opt))

    np.random.seed(0)
    torch.manual_seed(0)

    """ prepare dataloader """
    trainloader, devloader, num_types = prepare_dataloader(opt)
    prior =  load_prior(opt)


    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_inner=opt.d_inner,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                            opt.lr, betas=(0.9, 0.999), eps=1e-08)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)


    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))


    """ train each model @ event of interest """
    for event_interest in [32]:

        best_model = train(model, trainloader, devloader, optimizer, scheduler, opt, prior , event_interest)

        model.load_state_dict(best_model)

        model.eval()

        valid_ll = eval_epoch(model, devloader, opt, prior)

        posterior_train = get_posterior(model, trainloader, opt, prior)

        # logging
        with open(opt.log, 'a') as f:
            f.write(' {event_interest}, {val_ll}, {train} \n'
                    .format(event_interest = opt.event_interest, val_ll= valid_ll, train =posterior_train.cpu().numpy()))

import time
start = time.time()
np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    main()
end= time.time()
print("total training time is {}".format(end-start))

