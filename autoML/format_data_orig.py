import os
import shutil
import utils


def mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass


read_dir = '../data/micin/orig'
write_dir = '../data/orig'
train_dir = os.path.join(write_dir, 'train')
test_dir = os.path.join(write_dir, 'validation')

df = utils.load_brain_data_fromdir(read_dir)
train_df, test_df = utils.train_test_split_df(df, 0.2)

labels = set(test_df['label'])

mkdir(write_dir)
mkdir(train_dir)
mkdir(test_dir)
for label in labels:
    mkdir(os.path.join(train_dir, label))
    mkdir(os.path.join(test_dir, label))

for k, col in train_df.iterrows():
    read_fname = os.path.join(read_dir, col['fname'])
    write_fname = os.path.join(train_dir, col['label'], col['fname'])
    shutil.copy(read_fname, write_fname)

for k, col in test_df.iterrows():
    read_fname = os.path.join(read_dir, col['fname'])
    write_fname = os.path.join(test_dir, col['label'], col['fname'])
    shutil.copy(read_fname, write_fname)

print('done.')
