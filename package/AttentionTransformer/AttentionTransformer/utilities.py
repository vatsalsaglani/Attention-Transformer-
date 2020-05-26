import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import inspect
import tqdm
import contextlib
import os



@contextlib.contextmanager

def redirect_to_tqdm():

    '''
    Keep tqdm.write as the default print 
    '''

    old_print = print
    def new_print(*args, **kwargs):
        
        try:
            tqdm.tqdm.write(*args, **kwargs)

        except:
            old_print(*args, **kwargs)

    
    try:

        inspect.builtins.print = new_print

        yield

    finally:

        inspect.builtins.print = old_print


def tqdm_with_redirect(*args, **kwargs):
    '''
    Single bar for TQDM with print statements
    '''

    with redirect_to_tqdm():

        for x in tqdm.tqdm(*args, **kwargs):

            yield x



def device():
    '''
    keeping code device agnostic 
    '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_model_parameters(model):

    ''' Count of parameters which can be trained in the model defination '''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def patch_src(src, pad_idx):
    """Patching source with pad if needed"""

    return src

def patch_trg(trg, pad_idx):

    "patching target with mask of pad id"

    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)

    return trg, gold




def fit_attention(epoch, dataloader, model, optimizer, pbar, save_every = None, save_path = None, phase = 'training'):

    if phase == 'training':
        model.train()

    if phase == 'validation':
        model.eval()

    total_loss, n_label_total, n_label_correct = 0, 0, 0

    for ix, batch in enumerate(dataloader):

        src_seq = patch_src(batch['src'], source_pad_id).to(device())
        trg_seq, gold = map(lambda x: x.to(device()), patch_trg(batch['trg'], target_pad_id))

        if phase == 'training':

            optimizer.zero_grad()

        pred = model(src_seq, trg_seq)

        loss, n_correct, n_label = cal_performance(pred, gold, target_pad_id)

        if phase == 'training':

            loss.backward()
            optimizer.step()

        n_label_total += n_label
        n_label_correct += n_correct

        total_loss += loss.item()

    loss_per_label = total_loss / n_label_total
    accuracy = n_label_correct / n_label_total

    t = f'''
    {phase.upper()} EPOCH: {epoch} => 
        Loss: {loss_per_label}
        Accuracy: {accuracy}
    '''

    if save_every:
        if not save_path:
            return f"If you want to save after every {save_every} please provide a `save_path` to store the model"
        else:
            if epoch % save_every == 0:

                save_path_ = os.path.join(save_path, f'training_epoch_{epoch}.pt')
                save_path_dict = os.path.join(save_path, f'training_epoch_{epoch}_state_dict.pth')

                torch.save(model, save_path_)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_per_label,
                    'accuracy': accuracy
                }, save_path_dict)


    print(t)

    pbar.update(1)

    return loss_per_label, accuracy, t




    



