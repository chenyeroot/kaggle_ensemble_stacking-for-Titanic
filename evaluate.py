#自定义一个混淆矩阵

def cm_plot(y, yp):  
  from sklearn.metrics import confusion_matrix 
  cm = confusion_matrix(y, yp) 
  import matplotlib.pyplot as plt 
  plt.matshow(cm, cmap=plt.cm.Greens)
  plt.colorbar() 
  for x in range(len(cm)): 
    for y in range(len(cm)):
      plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')  
  plt.ylabel('True label') 
  plt.xlabel('Predicted label') 
  return plt
  
print("============输出预测结果（混淆矩阵）============")  
cm_plot(y_predict, preds)  
plt.show()

# 把结果保存到本地
result = pd.DataFrame({ 'PassengerId':test.iloc[:,0],'Survived':preds.astype(np.int32)})
result.to_csv(".../stacking_predictions.csv", index=False)
