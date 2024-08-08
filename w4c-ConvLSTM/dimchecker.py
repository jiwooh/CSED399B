# w4c dataset datapoint dimension checker

from utils.w4c_dataloader import RainData

import random

data = RainData(
    'training',
    data_root='../weather4cast-2023-lxz/data/',
    sat_bands=['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073'],
    regions=['boxi_0015'],
    full_opera_context=1512,
    size_target_center=252,
    years=['2019'],
    splits_path='../weather4cast-2023-lxz/data/timestamps_and_splits_stage2.csv'
)

idx = random.randint(0, len(data))
d = data[idx]

def printDim(arr): print(len(arr))

printDim(d)
printDim(d[0])
printDim(d[0][0])
printDim(d[0][0][0])
printDim(d[0][0][0][0])

# (3, 4, 11, 252, 252)