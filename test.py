import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix,  f1_score
from unet.unet_1 import UNet1


sn.set(font_scale=1.4) # for label size #ABHI - NEW CODE
batch_size = 278
num_epochs = 200

# load data
data_amp = sio.loadmat('data/train_data.mat')
train_data_amp = data_amp['train_data_amp']
train_data = train_data_amp

train_label_mask = data_amp['train_label_instance']
num_train_instances = len(train_data)

train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label_mask).type(torch.LongTensor)

train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


data_amp = sio.loadmat('data/test_data.mat')
test_data_amp = data_amp['test_data_amp']
test_data = test_data_amp

test_label_mask = data_amp['test_label_mask'] #Used for action prediction
#test_label_mask = data_amp['test_label_instance']
num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data_amp).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label_mask).type(torch.LongTensor)
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#loading trained weights

#unet = torch.load('weights/classification.pkl')

unet = torch.load('weights/Unet1/prediction.pkl')

unet.eval()
loss_x = 0
correct_test = 0
for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labelsV = Variable(labels.cuda())
        print(labelsV)
        print(samplesV)
        predict_label = unet(samplesV)

        prediction = predict_label.data.max(1)[1]
        #this part of code is used to plot the confustio matrix
        y_true = labelsV.cpu().numpy().flatten()
        y_pred = prediction.cpu().numpy().flatten()
        cm = confusion_matrix(y_true, y_pred)
        # Normalise
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sn.heatmap(cmn, annot=True, fmt='.2f', annot_kws={"size": 16})
        plt.savefig("image/final/confusion_matrix_test_final_pred_" + str(i) + ".png", format="PNG")
        plt.show()


        correct_test += prediction.eq(labelsV.data.long()).sum()

#calculate the metrics and print them here
print('F1 score:', f1_score(y_true, y_pred,average='weighted'))
print ('Recall:', recall_score(y_true, y_pred,average='weighted'))
print ('Precision:', precision_score(y_true, y_pred,average='weighted'))
print("Test accuracy:", (100 * float(correct_test) / (num_test_instances * 192)))


sio.savemat('out/Test_eval_results.mat', {'out': predict_label.cpu().numpy()} )
