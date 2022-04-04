import pandas as pd
import os
import random

file_name, class_label = [], []
labels = [1,2,3,4,5,6,7,8,9]

for val in os.listdir("test"):
    file_name.append(val.split(".")[0])
    # Use random value for now
    class_label.append(random.choice(labels))

list_dict = {'Id':file_name, 'Class':class_label}
df = pd.DataFrame(list_dict)
df.to_csv('trainLabels.csv',index=False)
    

