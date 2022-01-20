import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

activation_functions = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Linear': nn.Identity()
}

device = torch.device('cuda:0')

def create_dataset(p_val=0.1, p_test=0.2):
    import numpy as np
    import sklearn.datasets

    # Generate a dataset and plot it
    np.random.seed(0)
    num_samples = 1000

    X, y = sklearn.datasets.make_moons(num_samples, noise=0.2)
    
    train_end = int(len(X)*(1-p_val-p_test))
    val_end = int(len(X)*(1-p_test))
    
    # define train, validation, and test sets
    X_tr = X[:train_end]
    X_val = X[train_end:val_end]
    X_te = X[val_end:]

    # and labels
    y_tr = y[:train_end]
    y_val = y[train_end:val_end]
    y_te = y[val_end:]

    #plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.Spectral)
    return X_tr, y_tr, X_val, y_val

def create_dataset2():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5)),
    ])
    val_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])#把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
    trainset = datasets.CIFAR10(root='data/',train = True,download=True,transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=10)
    valset = datasets.CIFAR10(root='data/',train = True,download=True,transform=val_transform)
    valloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=10)
    testset = datasets.CIFAR10('data/',train=False,download=True,transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,num_workers=10)

    return trainloader,valloader,testloader


class Net(nn.Module):

    def __init__(self, layers, num_features, num_classes, layer_limit): 
        super(Net, self).__init__()
      
        #if hid_units is None or len(hid_units) == 0:
        #    raise Exception('You must specify at least one action!')

        layers_added = []
        
        max_layers = 7
        if max_layers < layer_limit:
            raise Exception('Maximum layers that ChildNet accepts is '.format(max_layers))

        hidd_unit_prev = num_features
        
        for i,layer in enumerate(layers):
            if isinstance(layer, int):
                layer_to_add = nn.Linear( in_features=hidd_unit_prev, out_features=layer)
                layers_added.append(layer_to_add)
                hidd_unit_prev = layer
            elif layer == 'EOS':
                break
            else:
                layers_added.append(activation_functions[layer])
                
        #last layer must contain 2 out_features (2 classes)
        layers_added.append(nn.Linear(in_features=hidd_unit_prev, out_features=num_classes))

        self.layers = nn.Sequential(*layers_added)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
    
    def forward(self, x):
        return self.layers(x)
    
def accuracy(ys, ts):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(ts.long(), torch.max(ys, 1)[1])
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())
    
def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class ChildNet():

    def __init__(self, layer_limit):
        self.criterion = nn.CrossEntropyLoss()

        X_tr, y_tr, X_val, y_val = create_dataset()
        self.X_tr = X_tr.astype('float32')
        self.y_tr = y_tr.astype('float32')
        self.X_val = X_val.astype('float32')
        self.y_val = y_val.astype('float32')
        
        self.num_features = X_tr.shape[-1]
        self.num_classes = 2
        self.layer_limit = layer_limit

    def compute_reward(self, layers, num_epochs):
        # store loss and accuracy for information
        train_losses = []
        val_accuracies = []
        patience = 10
        net = Net(layers, self.num_features, self.num_classes, self.layer_limit).to(device)
        #print(net)
        max_val_acc = 0
        
        # get training input and expected output as torch Variables and make sure type is correct
        tr_input = Variable(torch.from_numpy(self.X_tr)).to(device)
        tr_targets = Variable(torch.from_numpy(self.y_tr)).to(device)

        # get validation input and expected output as torch Variables and make sure type is correct
        val_input = Variable(torch.from_numpy(self.X_val)).to(device)
        val_targets = Variable(torch.from_numpy(self.y_val)).to(device)

        patient_count = 0
        # training loop
        for e in range(num_epochs):

            # predict by running forward pass
            tr_output = net(tr_input)
            # compute cross entropy loss
            #tr_loss = F.cross_entropy(tr_output, tr_targets.type(torch.LongTensor)) 
            tr_loss = self.criterion(tr_output.float(), tr_targets.long())
            # zeroize accumulated gradients in parameters
            net.optimizer.zero_grad()
            
            # compute gradients given loss
            tr_loss.backward()
            #print(net.l_1.weight.grad)
            # update the parameters given the computed gradients
            net.optimizer.step()
            
            train_losses.append(tr_loss.data)
        
            #AFTER TRAINING

            # predict with validation input
            val_output = net(val_input)
            val_output = torch.argmax(F.softmax(val_output, dim=-1), dim=-1).to(device)
            
            # compute loss and accuracy
            #val_loss = self.criterion(val_output.float(), val_targets.long())
            val_acc = torch.mean(torch.eq(val_output, val_targets.type(torch.LongTensor).to(device)).type(torch.FloatTensor))
            
            #accuracy(val_output, val_targets)
            val_acc = float(val_acc)
            val_accuracies.append(val_acc)
            
            
            #early-stopping
            if max_val_acc > val_acc:
                patient_count += 1             
                if patient_count == patience:
                    break
            else:
                max_val_acc = val_acc
                patient_count = 0
            


        #reset weights
        net.apply(weight_reset)
            
        return val_acc

class ChildNet2():

    def __init__(self, layer_limit):
        self.criterion = nn.CrossEntropyLoss()
        self.num_features = 32
        self.num_classes = 32
        self.layer_limit = layer_limit

    def compute_reward(self, layers, num_epochs):
        # store loss and accuracy for information
        train_losses = []
        val_accuracies = []
        patience = 10
        
        net = Net(layers, self.num_features, self.num_classes, self.layer_limit).to(device)
        #print(net)
        max_val_acc = 0

        trainloader, valloader,_ = create_dataset2()

        patient_count = 0
        # training loop
        for e in range(num_epochs):
            print('child epoch {}'.format(e+1))
            for inputs,labels in trainloader:
                tr_input = inputs.to(device)
                tr_label = labels.to(device)
                tr_output = net(tr_input)
                print(tr_output.size())
                print(tr_label.size())

                # compute cross entropy loss
                #tr_loss = F.cross_entropy(tr_output, tr_targets.type(torch.LongTensor)) 
                tr_loss = self.criterion(tr_output, tr_label)
                # zeroize accumulated gradients in parameters
                net.optimizer.zero_grad()
            
                # compute gradients given loss
                tr_loss.backward()
                # update the parameters given the computed gradients
                net.optimizer.step()
            
                train_losses.append(tr_loss.data)
            acc = 0
            #AFTER TRAINING
            for inputs,labels in enumerate(valloader):
                val_input = inputs.to(device)
                val_label = labels.to(device)

                # predict with validation input
                val_output = net(val_input)
                val_output = torch.argmax(F.softmax(val_output, dim=-1), dim=-1).to(device)
            
                # compute loss and accuracy
                acc = val_output.eq(val_label.view_as(val_output)).sum().item()
            
            #accuracy(val_output, val_targets)
            val_acc = acc/len(valloader.dataset)
            val_accuracies.append(val_acc)
            print('val_acc:{:6.6f}'.format(val_acc))
            
            #early-stopping
            if max_val_acc > val_acc:
                patient_count += 1             
                if patient_count == patience:
                    break
            else:
                max_val_acc = val_acc
                patient_count = 0
            


        #reset weights
        net.apply(weight_reset)
            
        return val_acc
