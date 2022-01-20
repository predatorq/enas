import torch
from childNet import ChildNet, ChildNet2
from utils import fill_tensor, indexes_to_actions
from torch.autograd import Variable

def training(policy, batch_size, total_actions, verbose = False, num_episodes = 500):
    ''' Optimization/training loop of the policy net. Returns the trained policy. '''
    device = torch.device('cuda')
    # training settings
    decay = 0.9
    training = True
    
    # childNet
    cn = ChildNet2(policy.layer_limit)
    nb_epochs = 100
    val_freq = 10
    
    # train policy network
    training_rewards = torch.zeros(1).to(device)
    losses = torch.zeros(1).to(device)
    baseline = torch.zeros(15, dtype=torch.float)
    
    print('start training')
    for i in range(num_episodes):
        if i%100 == 0: print('Epoch {}'.format(i))
        rollout, batch_r, batch_a_probs = [], [], []
        #forward pass
        with torch.no_grad():
            prob, actions = policy(training)
        batch_hid_units, batch_index_eos = indexes_to_actions(actions, batch_size, total_actions)
        
        #compute individually the rewards
        for j in range(batch_size):
            # policy gradient update 
            if verbose:
                print(batch_hid_units[j])
            r = cn.compute_reward(batch_hid_units[j], nb_epochs)**3
            if batch_hid_units[j]==['EOS']:
                r -= -1
            a_probs = prob[j, :batch_index_eos[j] + 1]

            batch_r += [r]
            batch_a_probs += [a_probs.view(1, -1)] 

        #rearrange the action probabilities
        a_probs = []
        for b in range(batch_size):
            a_probs.append(fill_tensor(batch_a_probs[b], policy.n_outputs, ones=True))
        a_probs = torch.stack(a_probs,0)

        #convert to pytorch tensors --> use get_variable from utils if training in GPU
        batch_a_probs = Variable(a_probs, requires_grad=True).to(device)
        batch_r = Variable(torch.tensor(batch_r), requires_grad=True).to(device)
        
        # classic traininng steps
        loss = policy.loss(batch_a_probs, batch_r, torch.mean(baseline))
        policy.optimizer.zero_grad()  
        loss.backward()
        policy.optimizer.step()

        # actualize baseline
        baseline = torch.cat((baseline[1:]*decay, torch.tensor([torch.mean(batch_r)*(1-decay)], dtype=torch.float)))
        
        # bookkeeping
        training_rewards += torch.mean(batch_r).detach()
        losses += loss.item()
        
        # print training
        if (i+1) % val_freq == 0:
            print(i+1)
            print(training_rewards/val_freq)
            print(losses/val_freq)
            # print('{:4d}. mean training reward: {:6.2f}, mean loss: {:7.4f}'.format(i+1, training_rewards/val_freq, losses/val_freq))

    print('done training')  
 
    return policy
