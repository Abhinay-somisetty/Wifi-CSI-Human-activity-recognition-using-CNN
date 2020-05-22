import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from unet.unet_1 import UNet1


#sn.set(font_scale=1.4)
batch_size = 128
num_epochs = 200

# load data
data_amp = sio.loadmat('data/train_data.mat')
train_data_amp = data_amp['train_data_amp']
train_data = train_data_amp

train_label_mask = data_amp['train_label_instance'] #used for action classification
#train_label_mask = data_amp['train_label_mask'] #used for action prediction
print(len(data_amp))
num_train_instances = len(train_data)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data_amp, train_label_mask, test_size=0.4, random_state=42)

train_data = torch.from_numpy(train_data_amp).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label_mask).type(torch.LongTensor)

train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)




data_amp = sio.loadmat('data/test_data.mat')
test_data_amp = data_amp['test_data_amp']
test_data = test_data_amp

#test_label_mask = data_amp['test_label_mask'] #USed for action prediction
test_label_mask = data_amp['test_label_instance']
num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label_mask).type(torch.LongTensor)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


unet = UNet1(n_classes=7)
unet = unet.cuda()

criterion = nn.CrossEntropyLoss(size_average=False).cuda()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[20,40,60,80,100,120,140,160,180,200,220,240,260,280,300],
                                                   gamma=0.5)


    #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
     #                                      factor=0.1, patience=10, verbose=False, threshold=0.0001,
      #                                     threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)



train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])
loss_train_values = []
loss_test_values = []

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    unet.train()
    scheduler.step()
    # for i, (samples, labels) in enumerate(train_data_loader):
    loss_x = 0
    loss_y = 0
    for (samples, labels) in tqdm(train_data_loader):
        samplesV = Variable(samples.cuda())
        labelsV = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_label = unet(samplesV)

        loss = criterion(predict_label, labelsV)
       # print(loss.item())

        loss.backward()
        optimizer.step()

    unet.eval()
    loss_x = 0
    correct_train = 0
    for i, (samples, labels) in enumerate(train_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())

            predict_label = unet(samplesV)

            prediction = predict_label.data.max(1)[1]
            correct_train += prediction.eq(labelsV.data.long()).sum()
            loss = criterion(predict_label, labelsV)
            loss_x += loss.item()
            #loss_train_values.append(loss_x)


    print("Training accuracy:", (100 * float(correct_train) / (num_train_instances*192)))
    train1 = (100 * float(correct_train) / (num_train_instances * 192))
    loss_train_values.append(train1)

    train_loss[epoch] = loss_x / num_train_instances
    train_acc[epoch] = 100 * float(correct_train) / (num_train_instances*192)
    trainacc = str(100 * float(correct_train) / (num_train_instances*192))[0:6]



    loss_x = 0
    correct_test = 0

    for i, (samples, labels) in enumerate(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())

            predict_label = unet(samplesV)

            prediction = predict_label.data.max(1)[1]
            correct_test += prediction.eq(labelsV.data.long()).sum()

            loss = criterion(predict_label, labelsV)
            loss_x += loss.item()
            #loss_test_values.append(loss_x)


    print("Test accuracy:", (100 * float(correct_test) / (num_test_instances * 192)))
    test1 = (100 * float(correct_test) / (num_test_instances * 192))
    loss_test_values.append(test1)

    test_loss[epoch] = loss_x / num_test_instances
    test_acc[epoch] = 100 * float(correct_test) / (num_test_instances * 192)
    testacc = str(100 * float(correct_test)/(num_test_instances * 192))[0:6]

    #loss_test_values.append(testacc)
    #saving the best weights for evaluation
    if epoch == 0:
        temp_test = correct_test
        temp_train = correct_train
    elif correct_test > temp_test:
        torch.save(unet, 'weights/Unet1/SGD_0.5_200_new_class_' + trainacc + 'Test' + testacc + '.pkl')
        temp_test = correct_test
        temp_train = correct_train

#to plot learning curves
from matplotlib import pyplot as plt
plt.plot(loss_train_values, label='Training accuarcy')
plt.plot(loss_test_values, label='Test accuracy')
plt.title('Learning Curve')
plt.legend(loc = 'best')
plt.show()

#saving the results

sio.savemat(
    'results/Unet1/SGD_TrainLoss_200_0.5_class' + 'Train' + str(100 * float(temp_train) / (num_test_instances * 192))[
                                                                 0:6] + 'Test' + str(
        100 * float(temp_test) / (num_test_instances * 192))[0:6] + '.mat', {'train_loss': train_loss})
sio.savemat(
    'results/Unet1/SGD_TestLoss_200_0.5_class' + 'Train' + str(100 * float(temp_train) / (num_test_instances * 192))[
                                                                0:6] + 'Test' + str(
        100 * float(temp_test) / (num_test_instances * 192))[0:6] + '.mat', {'test_loss': test_loss})
sio.savemat('results/Unet1/TrainAccuracy_200_0.5_class' + 'Train' + str(
    100 * float(temp_train) / (num_test_instances * 192))[0:6] + 'Test' + str(100 * float(temp_test) / (num_test_instances * 192))[
                                                                   0:6] + '.mat', {'train_acc': train_acc})
sio.savemat('results/Unet1/TestAccuracy_200_0.5_class' + 'Train' + str(
    100 * float(temp_train) / (num_test_instances * 192))[0:6] + 'Test' + str(100 * float(temp_test) / (num_test_instances * 192))[
                                                                   0:6] + '.mat', {'test_acc': test_acc})
