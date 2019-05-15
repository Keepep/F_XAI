import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda
def string_to_int(label):
    print label
    enc=OneHotEncoder()
    lb=enc.fit_transform(label).toarray()
    print lb
    return lb
class MLP():
    def __init__(self,model_name, file_path):
        data_name = "MLP"
        # get filename and remove .csv
        file_name = os.path.basename(file_path)[:-4]
        self.trained_model_path = 'Classifier/trained_model/' + data_name + '_' + file_name + '.pt'
        self.input_size=0;
        self.hidden_size=10
        self.out_size=2
        self.epochs=20
        self.batch_size=10
        self.lr=0.001

    def train(self, tr_data, tr_label):


        tr_data=tr_data.values
        tr_label=np.reshape(tr_label.values,(-1,1))


        self.input_size=tr_data.shape[1]

        if cuda_available():
            model=Net(self.input_size,self.hidden_size,self.out_size).cuda()
        else:
            model=Net(self.input_size,self.hidden_size,self.out_size)

        criterion = nn.BCELoss()
        #criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        losses=[]
        model.train()





        tr_label=string_to_int(tr_label)
        for epoch in range(self.epochs):
            for bt in range(0,tr_data.shape[0],self.batch_size):
                tr_data_batch=tr_data[bt:bt+self.batch_size,:]
                tr_label_batch=tr_label[bt:bt+self.batch_size,:]

                tr_data_batch=torch.from_numpy(tr_data_batch).float()
                tr_label_batch=torch.from_numpy(tr_label_batch).float()



                if cuda_available():
                    tr_data_batch = Variable(tr_data_batch.cuda(), requires_grad=True)
                    tr_label_batch = Variable(tr_label_batch.cuda())
                else:
                    tr_data_batch = Variable(tr_data_batch, requires_grad=True)
                    tr_label_batch = Variable(tr_label_batch)
                optimizer.zero_grad()

                y_=torch.nn.Softmax()(model(tr_data_batch))
                loss=criterion(y_,tr_label_batch)

                loss.backward()
                optimizer.step()
                if cuda_available():
                    loss_tmp =loss.cpu().data.numpy()
                else:
                    loss_tmp =loss.data.numpy()
                losses.append(loss_tmp)
        print 'loss: {0:02f}'.format(loss_tmp)

        torch.save(model,self.trained_model_path)

    def test(self,te_data,te_label):
        te_data=te_data.values
        te_label=np.reshape(te_label.values,(-1,1))

        te_label=string_to_int(te_label)

        model = torch.load(self.trained_model_path)
        model.eval()
        if cuda_available():
            model.cuda()
        count=0
        with torch.no_grad():
            for i in range(te_data.shape[0]):

                te_data_i = torch.from_numpy(te_data[i,:]).float()

                if cuda_available():
                    te_data_i = Variable(te_data_i.cuda())
                else:
                    te_data_i = Variable(te_data_i)

                outputs=torch.nn.Softmax()(model.forward(te_data_i))

                outputs=outputs.cpu().data

                pred=outputs.max(dim=0,keepdim=True)[1]
                pred=pred.cpu().data.numpy()
                if (te_label[i][pred] ==1.):
                    count+=1

        Acc=float(count)/te_data.shape[0]
        print 'Acc: {0:02f}'.format(Acc)



    def get_prob(self,te_data):

        te_data=te_data.values

        model = torch.load(self.trained_model_path)
        model.eval()
        if cuda_available():
            model.cuda()

        outputs=np.empty((te_data.shape[0],2))
        with torch.no_grad():
            for i in range(te_data.shape[0]):

                te_data_i = torch.from_numpy(te_data[i, :]).float()

                if cuda_available():
                    te_data_i = Variable(te_data_i.cuda())
                else:
                    te_data_i = Variable(te_data_i)

                output = torch.nn.Softmax()(model.forward(te_data_i))

                output = output.cpu().data
                outputs[i,:]=output

        print outputs

        return outputs

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output
