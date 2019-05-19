import pandas as pd
import glob
import numpy as np
import configparser
from datetime import datetime
nowtime = datetime.now().strftime("%y%m%d_%H%M")

# read setting
inifile = configparser.ConfigParser()
inifile.read('./config.ini', 'UTF-8')
models = inifile.get('voting', 'models').split()

# sum
for i,model in enumerate(models):
    predict = np.load(f'../out/{model}/predicts_vec.npy')
    if i==0:
        predicts = predict
    else:
        predicts += predict

voting_labels = np.argmax(predicts, axis=1)

# create submit file
submit = pd.DataFrame(data={"ImageId": [], "Label": []})
submit.ImageId = list(range(1, voting_labels.shape[0]+1))
submit.Label = voting_labels
submit.to_csv(f"./submit_{nowtime}_voting.csv", index=False)
np.savetxt(f"./submit_{nowtime}_models.txt", models, fmt='%s')