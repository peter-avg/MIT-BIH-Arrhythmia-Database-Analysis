import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

data = pd.read_csv('../Final/cnn_results_final.csv')
loss_acc = pd.read_csv('../Final/loss_accuracy.csv')

results = data.iloc[:,1:]

crayons = ['skyblue', 'red', 'green', 'yellow', 'grey']
offset = [2,3,4,5,6]
width = 0.1
x_axis = np.arange(len(data['Parameters']))

plt.figure(figsize=(14, 12))

for index,column in enumerate(results.columns):

    plt.barh(data['Parameters'], data[f"{column}"], color=crayons[index], height=0.5)
    plt.xlabel('Test Results')
    plt.ylabel('Parameters')
    plt.title(f'CNN Model {column} Results, Average: {data[column].mean()}')
    plt.xlim(min(data[f"{column}"]) - 0.01, max(data[f"{column}"]) + 0.01)

    plt.savefig(f'../Final/{column}.png')

plt.figure(figsize=(13, 14))

for index, column in enumerate(results.columns):
    plt.barh(x_axis - width*offset[index], data[f"{column}"], color=crayons[index], height=width, label=f'{column}, Average: {data[column].mean()}')

plt.xlabel('Test Results')
plt.ylabel('Parameters')
plt.title('CNN Model Results')
plt.yticks(x_axis - width*2,list(data['Parameters']))
plt.legend(bbox_to_anchor=(0.7, 1.1), loc='upper left', borderaxespad=0)

plt.savefig('../Final/combined_final.png')

plt.figure(figsize=(13, 12))

plt.subplot(1, 2, 1)
plt.plot(range(len(loss_acc['Train_Loss'])), loss_acc['Train_Loss'], label="Train Loss")
plt.plot(range(len(loss_acc['Train_Loss'])), loss_acc['Validate_Loss'], label="Validate Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(loss_acc['Train_Loss'])), loss_acc['Train_Accuracy'], label="Train Accuracy")
plt.plot(range(len(loss_acc['Train_Loss'])), loss_acc['Validate_Accuracy'], label="Validate Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.legend()

plt.savefig('../Final/loss_acc.png')
