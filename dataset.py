import numpy as np
import torch
import h5py
import os
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import Sampler


def _batch_sampler(d_sets, num_samples, batch_size, n_max=100000):
    ids = np.vstack([d.db['slide_id'][:][d.id_keep] for d in d_sets])
    id_, _ = np.unique(ids, return_counts=True)
    id_batches = np.zeros((0, batch_size)).astype(int)

    for i in id_:
        id_p = np.nonzero(ids == i)[0]
        id_p = np.random.permutation(id_p)[:len(id_p)-(len(id_p) % batch_size)]
        id_p = id_p.reshape(-1, batch_size)
        id_batches = np.vstack([id_batches, id_p])

    class SlideBatchSampler(Sampler):

        def __init__(self, id_batches, num_samples):
            self.id_batches = id_batches
            self.num_samples = num_samples

        def __iter__(self):
            return iter(self.id_batches[np.random.permutation(len(self.id_batches))[:self.num_samples//batch_size]].flatten())

        def __len__(self):
            return self.num_samples

    sampler = SlideBatchSampler(id_batches, num_samples=min(num_samples, n_max))
    return torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)


def from_folder(src_filename, batch_size, mode='train', ratio=None, seed=0, transform=None):

    if not os.path.exists(src_filename):
        return None

    d_main = h5py.File(src_filename, 'r')
    dumps = d_main['dumps']

    d_sets = []
    for d_file in dumps:
        d = HDF5_Dataset(
            filename=os.path.join(os.path.dirname(src_filename), d_file.decode('ascii')),
            mode=mode,
            transform=transform,
            )
        d_sets.append(d)

    id_slides = _ids_train_val(d_sets=d_sets, ratio=ratio, seed=seed)

    for d in d_sets:
        d.set_id_slides(id_slides[mode])

    for i, _ in enumerate(d_sets[1:]):
        d_sets[i+1].set_offset(d_sets[i].offset + len(d_sets[i]))

    ds = ConcatDataset(d_sets)
    return ds, _batch_sampler(d_sets, num_samples=len(ds), batch_size=batch_size)


def extract_filenames(src_filename):
    d_main = h5py.File(src_filename, 'r')
    slide_filenames = d_main['slide_filenames']
    return np.array([s.decode('ascii') for s in slide_filenames])


def _ids_train_val(d_sets, ratio, seed=0):

    if np.sum(ratio) != 1:
        print('Error, ratio do not sum to 1 ({})'.format(ratio))
        return None

    rnd = np.random.RandomState(seed)

    # Look for slide id
    v = [np.array(d.db['slide_id']) for d in d_sets]
    samples = np.concatenate(v, axis=0).squeeze()
    id_u = np.unique(samples)

    ratio_ = (len(np.unique(id_u)) * np.cumsum(ratio)).astype(int)
    idx_src_slide = rnd.permutation(np.unique(id_u))

    return {'train': idx_src_slide[:ratio_[0]], 'valid': idx_src_slide[ratio_[0]:]}


class HDF5_Dataset(Dataset):

    def __init__(self, filename, mode='train', transform=None, seed=0):

        self.mode = mode
        self.rnd = np.random.RandomState(seed)
        self.filename = filename
        self.transform = transform
        self.db = h5py.File(filename, 'r')
        self.data_keys = list(self.db.keys())
        self.id_keep = np.arange(self.db['image'].shape[0])
        self.offset = 0

    def set_id_slides(self, id_slides):

        id_keep = np.zeros(len(self)).astype(bool)
        for i in id_slides:
            id_keep = id_keep | (np.array(self.db['slide_id']) == i).flatten()
        self.id_keep = np.nonzero(id_keep)[0]

    def set_offset(self, offset):
        self.offset = offset

    def __len__(self):
        return len(self.id_keep)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            sample = {
                'image': self.db['image'][self.id_keep[idx]],
                'image_he': self.db['image_he'][self.id_keep[idx]],
                'patch_id': self.db['patch_id'][self.id_keep[idx]][0].astype(int),
                'slide_id': self.db['slide_id'][self.id_keep[idx]][0].astype(int),
                'coord': self.db['coord'][self.id_keep[idx]].astype(int),
                'idx': idx,
                'idx_overall': idx + self.offset
            }

            if 'image_pairs' in self.db.keys() and 'image_pairs_he' in self.db.keys():
                sample['image_pairs'] = self.db['image_pairs'][self.id_keep[idx]]
                sample['image_pairs_he'] = self.db['image_pairs_he'][self.id_keep[idx]]

        except KeyError:

            print('Non valid id: {}/{}'.format(idx, len(self)))
            sample = {
                'image': np.uint8(np.random.uniform(150, 180, (224, 224, 3))),
                'image_he': np.uint8(np.random.uniform(150, 180, (224, 224, 2))),
                'patch_id': 0,
                'slide_id': 0,
                'idx': idx,
                'idx_overall': idx + self.offset
            }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ApplyOnKey(object):

    def __init__(self, on_key, func):
        self.on_key = on_key
        self.func = func

    def __call__(self, sample):
        data = sample[self.on_key]
        sample[self.on_key] = self.func(data)
        return sample

    def __repr__(self):
        return '{}(on_key={})'.format(self.__class__.__name__, self.on_key) + ' - ' + self.func.__repr__()