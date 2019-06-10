import numpy as np
import torch
from torchvision.datasets import MNIST, SVHN, CIFAR10, STL10
from torchvision import transforms
import os
import torchvision.utils as vutils
# for coil
from PIL import Image


# from scipy import io

class DataLoader(object):
    def __init__(self, config, raw_loader, indices, batch_size):
        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images, 0)
        self.labels = np.array(self.labels, dtype=np.int64)
        if config.num_label != 10 and config.dataset == 'cifar':  # reorder the label
            lbl_range = config.allowed_label.split(",")
            lbl_range = [int(i) for i in lbl_range]
            lbl_range.sort()
            for i, j in enumerate(lbl_range):
                self.labels[self.labels == j] = i
        self.labels = torch.from_numpy(self.labels).squeeze()

        if config.dataset == 'mnist':
            self.images = self.images.view(self.images.size(0), -1)

        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)

    def get_zca_cuda(self, reg=1e-6):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        mean = images.mean(0)
        images -= mean.expand_as(images)
        sigma = torch.mm(images.transpose(0, 1), images) / images.size(0)
        U, S, V = torch.svd(sigma)
        components = torch.mm(torch.mm(U, torch.diag(1.0 / torch.sqrt(S) + reg)), U.transpose(0, 1))
        return components, mean

    def apply_zca_cuda(self, components):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        self.images = torch.mm(images, components.transpose(0, 1)).cpu()

    def generator(self, inf=False, shuffle=True):
        while True:
            indices = np.arange(self.images.size(0))
            if shuffle:
                np.random.shuffle(indices)
            indices = torch.from_numpy(indices)
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))
                ret_images, ret_labels = self.images[indices[start: end]], self.labels[indices[start: end]]
                yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self, shuffle=True):
        return self.generator(shuffle=shuffle)

    def __len__(self):
        return self.len

class DataBatchLoader(object):
    def __init__(self, config, raw_loader, indices, batch_size,
                 transform=None, target_transform=None, img_side=224):
        # todo: add str load img
        self.folder_root = raw_loader.folder_root

        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.transform = transform
        self.target_transform = target_transform
        self.img_side = img_side
        if config.num_label != 1000:  # reorder the label
            lbl_range = np.unique(self.labels)
            for i, j in enumerate(lbl_range):
                self.labels[self.labels == j] = i
        self.labels = torch.from_numpy(self.labels).squeeze()


        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)

    def generator(self, inf=False, shuffle=True):
        while True:
            indices = np.arange(self.images.shape[0])
            if shuffle:
                np.random.shuffle(indices)
            # indices = torch.from_numpy(indices)
            for start in range(0, indices.shape[0], self.batch_size):
                end = min(start + self.batch_size, indices.shape[0])
                ret_images = self.__loadimg(indices[start: end])
                lab_ind = torch.from_numpy(indices[start: end])
                ret_labels = self.labels[lab_ind]
                yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self, shuffle=True):
        return self.generator(shuffle=shuffle)

    def __len__(self):
        return self.len

    def __loadimg(self, inds):
        fns = self.images[inds]
        images = []
        for fn in fns:
            path_to_data = os.path.join(self.folder_root, fn)
            data = Image.open(path_to_data).convert('RGB')  # HWC
            data = data.resize((self.img_side, self.img_side), Image.ANTIALIAS)  # ANTIALIAS;BILINEAR
            # transforms
            if self.transform is not None:
                data = self.transform(data)
            images.append(data)

        images = torch.stack(images, 0)
        return images


class coil20(CIFAR10):
    """`COIL20 <https://>`_ Dataset.


    """
    base_folder = 'coil-20-proc'
    url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
    filename = "coil-20-proc.zip"
    tgz_md5 = '464dec76a6abfcd00e8de6cf1e7d0acc'  # tar.gz: 891c9b54622b6b676d91b54eae340c1b
    class_names_file = ''

    def __init__(self, root,
                 transform=None, target_transform=None, download=False, img_side=128):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.img_side = img_side

        if download:
            self.download2()

        # if not self._check_integrity():
        #     raise RuntimeError(
        #         'Dataset not found or corrupted. '
        #         'You can use download=True to download it')

        # now load the picked numpy arrays
        self.data, self.labels = self.__loadfile()

        class_file = os.path.join(
            root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    def __loadfile(self):
        images = []
        labels = np.arange(20, dtype=int).repeat(72)
        data_pattern = lambda a, b: "obj{}__{}.png".format(a + 1, b)  # a: [0,19]; b: [0,71]
        for c in range(20):  # 0-based
            for i in range(72):
                data_file = data_pattern(c, i)
                path_to_data = os.path.join(self.root, self.base_folder, data_file)
                data = Image.open(path_to_data).convert('RGB')  # HWC
                data = data.resize((self.img_side, self.img_side), Image.ANTIALIAS)  # ANTIALIAS;BILINEAR
                data = np.array(data)
                data = np.expand_dims(data, axis=0)
                images.append(data)

        images = np.concatenate(images, axis=0)
        # images = np.transpose(images, (0, 3, 1, 2))
        # images = images / 255.  # value:[0, 1]

        return images, labels

    def download2(self):
        import zipfile

        def download_url(url, froot, filename, md5):
            import hashlib
            import errno
            from six import moves

            def check_integrity(fpath_, md5_):
                if not os.path.isfile(fpath_):
                    return False
                md5o = hashlib.md5()
                with open(fpath_, 'rb') as f:
                    # read in 1MB chunks
                    for chunk in iter(lambda: f.read(1024 * 1024 * 1024), b''):
                        md5o.update(chunk)
                md5c = md5o.hexdigest()
                if md5c != md5_:
                    return False
                return True

            froot = os.path.expanduser(froot)
            fpath = os.path.join(froot, filename)
            try:
                os.makedirs(froot)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise
            # downloads file
            if os.path.isfile(fpath) and check_integrity(fpath, md5):
                print('Using downloaded and verified file: ' + fpath)
            else:
                print('Downloading ' + url + ' to ' + fpath)
                moves.urllib.request.urlretrieve(url, fpath)

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        # cwd = os.getcwd()
        zf = zipfile.ZipFile(os.path.join(root, self.filename), "r")
        zf.extractall(root)
        zf.close()
        # os.chdir(cwd)


class imagenet10(CIFAR10):
    """`imagenet10 <https://>`_ Dataset.


    """
    base_folder = 'imagenet'
    foldernames = ["ILSVRC2012_img_train", "ILSVRC2012_img_val", "ILSVRC2012_img_test"]

    def __init__(self, root, splitpart, target_transform=None):
        self.root = os.path.expanduser(root)
        self.target_transform = target_transform
        self.data = []
        self.labels = []
        if "train" in splitpart:
            data, labels = self.__loadfile(self.foldernames[0], "train")
            self.data.append(data)
            self.labels.append(labels)
        if "val" in splitpart:
            data, labels = self.__loadfile(self.foldernames[1], "val")
            self.data.append(data)
            self.labels.append(labels)
        if "test" in splitpart:
            data, labels = self.__loadfile(self.foldernames[2], "test")
            self.data.append(data)
            self.labels.append(labels)
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.folder_root = os.path.join(self.root, self.base_folder)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_str, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    def __loadfile(self, foldername, labfname):
        images = []
        labels = []
        fn = os.path.join(self.root, self.base_folder, "label", labfname+".txt")
        fp = open(fn, "r")
        for li in fp:
            image, label = li.split(" ")
            images.append(os.path.join(foldername, image))
            labels.append(int(label[:-1]))   # cut '\n'
        fp.close()

        return images, labels


def get_mnist_loaders(config):
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = MNIST(config.data_root, train=True, download=True, transform=transform)
    dev_set = MNIST(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: config.size_labeled_data / 10]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print 'labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0]

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels == i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, dev_loader, special_set


def get_svhn_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN(config.data_root, split='train', download=True, transform=transform)
    dev_set = SVHN(config.data_root, split='test', download=True, transform=transform)

    def preprocess(data_set):
        for j in range(len(data_set.data)):
            if data_set.labels[j][0] == 10:
                data_set.labels[j][0] = 0

    preprocess(training_set)
    preprocess(dev_set)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: config.size_labeled_data / 10]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    labeled_indices, unlabeled_indices = indices[mask], indices
    print 'labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(
        dev_set)

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels == i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, dev_loader, special_set


def get_cifar_loaders_test(config, lab_ind=True):
    tr_list = []
    if hasattr(config, 'image_side') and config.image_side != 32:  # resize
        tr_list.append(transforms.Resize(config.image_side))
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform = transforms.Compose(tr_list)
    training_set = CIFAR10('cifar', train=True, download=True, transform=transform)
    dev_set = CIFAR10('cifar', train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    dev_indices = np.arange(len(dev_set))
    if config.num_label != 10:
        assert hasattr(config, 'allowed_label') and config.allowed_label != "", "No allowed_label"
        # dev
        dev_labels = np.array([dev_set[i][1] for i in dev_indices], dtype=np.int64)
        mask3 = np.zeros(dev_indices.shape[0], dtype=np.bool)
        mask2 = np.zeros(indices.shape[0], dtype=np.bool)
        lbl_range = config.allowed_label.split(",")
        lbl_range = [int(i) for i in lbl_range]
        for i in lbl_range:
            mask2[np.where(labels == i)[0][:]] = True
            mask3[np.where(dev_labels == i)[0][:]] = True
        indices = indices[mask2]
        dev_indices = dev_indices[mask3]

    unlabeled_loader = DataLoader(config, training_set, indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, dev_indices, config.dev_batch_size)

    ind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
    if lab_ind and os.path.exists(ind_path):
        labeled_indices = np.load(ind_path)
        print("Find lab_ind!")
        labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
        if config.num_label != 10:
            # lbl: convert data_ind to unl_ind
            mask = np.isin(indices, labeled_indices)
            labeled_indices = np.arange(len(indices))
            labeled_indices = labeled_indices[mask]
        return unlabeled_loader, dev_loader, labeled_loader, labeled_indices
    else:
        print("no lab_ind!")
    return unlabeled_loader, dev_loader


def get_cifar_loaders(config):
    save_ind = True
    tr_list = []
    if hasattr(config, 'flip') and config.flip:
        tr_list.append(transforms.RandomHorizontalFlip())
    if hasattr(config, 'image_side') and config.image_side != 32:  # resize
        tr_list.append(transforms.Resize(config.image_side))
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform = transforms.Compose(tr_list)
    tr_list = []
    if hasattr(config, 'image_side') and config.image_side != 32:  # resize
        tr_list.append(transforms.Resize(config.image_side))
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform2 = transforms.Compose(tr_list)
    training_set = CIFAR10('cifar', train=True, download=True, transform=transform)
    dev_set = CIFAR10('cifar', train=False, download=True, transform=transform2)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)

    dev_indices = np.arange(len(dev_set))
    if config.num_label != 10:
        assert hasattr(config, 'allowed_label') and config.allowed_label != "", "No allowed_label"
        # dev
        dev_labels = np.array([dev_set[i][1] for i in dev_indices], dtype=np.int64)
        mask3 = np.zeros(dev_indices.shape[0], dtype=np.bool)
        mask2 = np.zeros(indices.shape[0], dtype=np.bool)
        lbl_range = config.allowed_label.split(",")
        lbl_range = [int(i) for i in lbl_range]
        for i in lbl_range:
            mask[np.where(labels == i)[0][: config.size_labeled_data / config.num_label]] = True
            mask2[np.where(labels == i)[0][:]] = True
            mask3[np.where(dev_labels == i)[0][:]] = True
        labeled_indices, unlabeled_indices = indices[mask], indices[mask2]
        dev_indices = dev_indices[mask3]

        ind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
        if os.path.exists(ind_path) or \
                (hasattr(config, "train_step") and config.train_step != 1):  # try to load step 1 inds
            assert os.path.exists(ind_path), "step {} Unknown label inds".format(config.train_step)
            labeled_indices = np.load(ind_path)
            print("Find lab_ind!")
            save_ind = False

    else:
        lbl_range = range(10)
        for i in lbl_range:
            mask[np.where(labels == i)[0][: config.size_labeled_data / 10]] = True
        labeled_indices, unlabeled_indices = indices[mask], indices
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print 'labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(
        dev_set)

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, dev_indices, config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels == i)[0][0]]][0])
    special_set = torch.stack(special_set)
    # save label indices
    ind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
    if save_ind:  # save step-1 / non-step inds
        np.save(ind_path, labeled_indices)

    return labeled_loader, unlabeled_loader, dev_loader, special_set


def get_stl10_loaders_test(config, lab_ind=True):
    tr_list = []
    if hasattr(config, 'image_side') and config.image_side != 96:  # resize
        tr_list.append(transforms.Resize(config.image_side))
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform = transforms.Compose(tr_list)
    training_set = STL10(config.data_root, split='train+unlabeled', download=True, transform=transform)
    dev_set = STL10(config.data_root, split='test', download=True, transform=transform)

    indices = np.arange(len(training_set))
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    dev_indices = np.arange(len(dev_set))
    if config.num_label != 10:
        assert hasattr(config, 'allowed_label') and config.allowed_label != "", "No allowed_label"
        # dev
        dev_labels = np.array([dev_set[i][1] for i in dev_indices], dtype=np.int64)
        mask3 = np.zeros(dev_indices.shape[0], dtype=np.bool)
        mask2 = np.zeros(indices.shape[0], dtype=np.bool)
        lbl_range = config.allowed_label.split(",")
        lbl_range = [int(i) for i in lbl_range]
        for i in lbl_range:
            mask2[np.where(labels == i)[0][:]] = True
            mask3[np.where(dev_labels == i)[0][:]] = True
        indices = indices[mask2]
        dev_indices = dev_indices[mask3]

    unlabeled_loader = DataLoader(config, training_set, indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, dev_indices, config.dev_batch_size)

    ind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
    if lab_ind and os.path.exists(ind_path):
        labeled_indices = np.load(ind_path)
        print("Find lab_ind!")
        labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
        if config.num_label != 10:
            # lbl: convert data_ind to unl_ind
            mask = np.isin(indices, labeled_indices)
            labeled_indices = np.arange(len(indices))
            labeled_indices = labeled_indices[mask]
        return unlabeled_loader, dev_loader, labeled_loader, labeled_indices
    else:
        print("no lab_ind!")
    return unlabeled_loader, dev_loader


def get_stl10_loaders(config):  # n*3*96*96
    save_ind = True
    tr_list = []
    if hasattr(config, 'flip') and config.flip:
        tr_list.append(transforms.RandomHorizontalFlip())
    if hasattr(config, 'image_side') and config.image_side != 96:  # resize
        tr_list.append(transforms.Resize(config.image_side))
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform = transforms.Compose(tr_list)
    tr_list = []
    if hasattr(config, 'image_side') and config.image_side != 96:  # resize
        tr_list.append(transforms.Resize(config.image_side))
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform2 = transforms.Compose(tr_list)
    training_set = STL10(config.data_root, split='train+unlabeled', download=True, transform=transform)
    dev_set = STL10(config.data_root, split='test', download=True, transform=transform2)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)

    dev_indices = np.arange(len(dev_set))
    if config.num_label != 10:
        assert hasattr(config, 'allowed_label') and config.allowed_label != "", "No allowed_label"
        # dev
        dev_labels = np.array([dev_set[i][1] for i in dev_indices], dtype=np.int64)
        mask3 = np.zeros(dev_indices.shape[0], dtype=np.bool)
        mask2 = np.zeros(indices.shape[0], dtype=np.bool)
        lbl_range = config.allowed_label.split(",")
        lbl_range = [int(i) for i in lbl_range]
        for i in lbl_range:
            mask[np.where(labels == i)[0][: config.size_labeled_data / config.num_label]] = True
            mask2[np.where(labels == i)[0][:]] = True
            mask3[np.where(dev_labels == i)[0][:]] = True
        labeled_indices, unlabeled_indices = indices[mask], indices[mask2]
        dev_indices = dev_indices[mask3]

        ind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
        if os.path.exists(ind_path) or \
                (hasattr(config, "train_step") and config.train_step != 1):  # try to load step 1 inds
            assert os.path.exists(ind_path), "step {} Unknown label inds".format(config.train_step)
            labeled_indices = np.load(ind_path)
            print("Find lab_ind!")
            save_ind = False

    else:
        lbl_range = range(10)
        for i in lbl_range:
            mask[np.where(labels == i)[0][: config.size_labeled_data / 10]] = True
        labeled_indices, unlabeled_indices = indices[mask], indices
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print 'labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(
        dev_set)

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, dev_indices, config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels == i)[0][0]]][0])
    special_set = torch.stack(special_set)
    # save label indices
    ind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
    if save_ind:  # save step-1 / non-step inds
        np.save(ind_path, labeled_indices)

    return labeled_loader, unlabeled_loader, dev_loader, special_set


def get_coil20_loaders_test(config, lab_ind=True):
    tr_list = []
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform = transforms.Compose(tr_list)
    all_set = coil20(config.data_root, download=True, transform=transform, img_side=config.image_side)

    num_data = len(all_set)
    indices = np.arange(num_data)
    labels = np.array([all_set[i][1] for i in indices], dtype=np.int64)
    mask2 = np.zeros(indices.shape[0], dtype=np.bool)
    if hasattr(config, 'allowed_label') and config.allowed_label != "":
        lbl_range = config.allowed_label.split(",")
        lbl_range = [int(i) for i in lbl_range]
        for i in lbl_range:
            mask2[np.where(labels == i)[0][:]] = True
        indices = indices[mask2]

    unlabeled_loader = DataLoader(config, all_set, indices, config.train_batch_size_2)

    lind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
    uind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.uind.npy'.format(config.dataset, config.suffix))
    if lab_ind and os.path.exists(lind_path):
        assert os.path.exists(uind_path), "uind does not exist"
        labeled_indices = np.load(lind_path)
        unlabeled_indices = np.load(uind_path)
        print("Find lab_ind!")
        labeled_loader = DataLoader(config, all_set, labeled_indices, config.train_batch_size)
        unlabeled_loader = DataLoader(config, all_set, unlabeled_indices, config.train_batch_size)
        # lbl: convert data_ind to unl_ind
        mask = np.isin(indices, labeled_indices)
        labeled_indices = np.arange(len(indices))
        labeled_indices = labeled_indices[mask]
        mask3 = np.isin(indices, unlabeled_indices)
        mask3 = ~mask3
        dev_indices = indices[mask3]
        dev_loader = DataLoader(config, all_set, dev_indices, config.dev_batch_size)
        return unlabeled_loader, dev_loader, labeled_loader, labeled_indices
    else:
        print("no lab_ind!")

    return unlabeled_loader, unlabeled_loader


def get_coil20_loaders(config):  # n*1*128*128
    save_ind = True
    tr_list = []
    if hasattr(config, 'flip') and config.flip:
        tr_list.append(transforms.RandomHorizontalFlip())
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform = transforms.Compose(tr_list)
    all_set = coil20(config.data_root, download=True, transform=transform, img_side=config.image_side)
    # dev ratio:1040/1440; lab ratio: 20/1440
    # ref: https://github.com/csyanbin/Semi-supervised_Neural_Network/blob/master/utils/coil_data.py

    num_data = len(all_set)
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    labels = np.array([all_set[i][1] for i in indices], dtype=np.int64)

    assert hasattr(config,
                   'size_unlabeled_data') and config.size_unlabeled_data <= num_data, "size_unlabeled_data too large"

    lind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
    uind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.uind.npy'.format(config.dataset, config.suffix))
    if os.path.exists(lind_path) or \
            (hasattr(config, "train_step")
             and config.train_step != 1):  # try to load step 1 inds
        assert os.path.exists(lind_path) and os.path.exists(uind_path), "step {} Unknown label inds".format(
            config.train_step)
        labeled_indices = np.load(lind_path)
        unlabeled_indices = np.load(uind_path)
        # dev data, part labels or not
        mask3 = np.zeros(num_data, dtype=np.bool)
        if hasattr(config, 'allowed_label') and config.allowed_label != "":
            lbl_range = config.allowed_label.split(",")
            lbl_range = [int(i) for i in lbl_range]
        else:
            lbl_range = range(20)
        for i in lbl_range:
            mask3[np.where(labels == i)[0][:]] = True
        indices = indices[mask3]
        mask3 = np.isin(indices, unlabeled_indices)
        mask3 = ~mask3
        dev_indices = indices[mask3]
        print("Find lab_ind!")
        save_ind = False
    else:
        mask = np.zeros(num_data, dtype=np.bool)
        mask2 = np.zeros(num_data, dtype=np.bool)
        mask3 = np.zeros(num_data, dtype=np.bool)
        if hasattr(config, 'allowed_label') and config.allowed_label != "":
            lbl_range = config.allowed_label.split(",")
            lbl_range = [int(i) for i in lbl_range]
        else:
            lbl_range = range(20)
        for i in lbl_range:
            mask[np.where(labels == i)[0][: config.size_labeled_data // config.num_label]] = True
            mask2[np.where(labels == i)[0][:config.size_unlabeled_data // config.num_label]] = True
            mask3[np.where(labels == i)[0][config.size_unlabeled_data // config.num_label:]] = True
        labeled_indices, unlabeled_indices = indices[mask], indices[mask2]
        dev_indices = indices[mask3]

    print 'labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], \
        'dev size', dev_indices.shape[0]

    labeled_loader = DataLoader(config, all_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, all_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, all_set, dev_indices, config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(all_set[indices[np.where(labels == i)[0][0]]][0])
    special_set = torch.stack(special_set)
    # save label indices
    if save_ind:  # save step-1 / non-step inds
        np.save(lind_path, labeled_indices)
        np.save(uind_path, unlabeled_indices)

    return labeled_loader, unlabeled_loader, dev_loader, special_set


def get_imagenet10_loaders_test(config, lab_ind=True):
    tr_list = []
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform = transforms.Compose(tr_list)
    train_set = imagenet10(config.data_root, splitpart='train')  # untested
    test_set = imagenet10(config.data_root, splitpart='val')  # untested

    num_data = len(train_set)
    indices = np.arange(num_data)
    labels = np.array([train_set[i][1] for i in indices], dtype=np.int64)
    mask2 = np.zeros(indices.shape[0], dtype=np.bool)
    if hasattr(config, 'allowed_label') and config.allowed_label != "":
        lbl_range = config.allowed_label.split(",")
        lbl_range = [int(i) for i in lbl_range]
    else:
        lbl_range = np.arange(1000)
        np.random.shuffle(lbl_range)
        lbl_range = lbl_range[:config.num_label]
    for i in lbl_range:
        mask2[np.where(labels == i)[0][:]] = True
    indices = indices[mask2]

    unlabeled_loader = DataBatchLoader(config, train_set, indices, config.train_batch_size_2,
                                       transform=transform, img_side=config.image_side)

    lind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
    uind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.uind.npy'.format(config.dataset, config.suffix))
    tind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.tind.npy'.format(config.dataset, config.suffix))
    if lab_ind and os.path.exists(lind_path):
        assert os.path.exists(uind_path), "uind does not exist"
        labeled_indices = np.load(lind_path)
        unlabeled_indices = np.load(uind_path)
        dev_indices = np.load(tind_path)
        print("Find lab_ind!")
        labeled_loader = DataBatchLoader(config, train_set, labeled_indices, config.train_batch_size,
                                         transform=transform, img_side=config.image_side)
        unlabeled_loader = DataBatchLoader(config, train_set, unlabeled_indices, config.train_batch_size,
                                           transform=transform, img_side=config.image_side)
        dev_loader = DataBatchLoader(config, test_set, dev_indices, config.dev_batch_size,
                                     transform=transform, img_side=config.image_side)
        return unlabeled_loader, dev_loader, labeled_loader, labeled_indices
    else:
        print("no lab_ind!")

    return unlabeled_loader, unlabeled_loader


def get_imagenet10_loaders(config):  # n*1*128*128
    save_ind = True
    tr_list = []
    if hasattr(config, 'flip') and config.flip:
        tr_list.append(transforms.RandomHorizontalFlip())
    tr_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # tr_list += [transforms.ToTensor(), transforms.Normalize((0.53, 0.53, 0.52), (0.29, 0.29, 0.28))]
    transform = transforms.Compose(tr_list)
    train_set = imagenet10(config.data_root, splitpart='train')  # untested
    test_set = imagenet10(config.data_root, splitpart='val')  # untested
    # 1000 cls
    # train: 1,281,167, val: 50,000, test: 100,000, but no cls on test

    num_data = len(train_set)
    num_data2 = len(test_set)
    indices = np.arange(num_data)
    indices2 = np.arange(num_data2)
    np.random.shuffle(indices)
    labels = np.array([train_set[i][1] for i in indices], dtype=np.int64)
    labels2 = np.array([test_set[i][1] for i in indices2], dtype=np.int64)

    lind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.lind.npy'.format(config.dataset, config.suffix))
    uind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.uind.npy'.format(config.dataset, config.suffix))
    tind_path = os.path.join(config.save_dir, '{}.FM+VI.{}.tind.npy'.format(config.dataset, config.suffix))
    if os.path.exists(lind_path) or \
            (hasattr(config, "train_step")
             and config.train_step != 1):  # try to load step 1 inds
        assert os.path.exists(lind_path) and os.path.exists(uind_path), "step {} Unknown label inds".format(
            config.train_step)
        labeled_indices = np.load(lind_path)
        unlabeled_indices = np.load(uind_path)
        dev_indices = np.load(tind_path)
        print("Find lab_ind!")
        save_ind = False
    else:
        mask = np.zeros(num_data, dtype=np.bool)
        mask2 = np.zeros(num_data, dtype=np.bool)
        mask3 = np.zeros(num_data2, dtype=np.bool)
        if hasattr(config, 'allowed_label') and config.allowed_label != "":
            lbl_range = config.allowed_label.split(",")
            lbl_range = [int(i) for i in lbl_range]
        else:
            lbl_range = np.arange(1000)
            np.random.shuffle(lbl_range)
            lbl_range = lbl_range[:config.num_label]

        # the amount in a class: [732, 1300]; there are less than 1300 images in 104 classes.
        min_num = 1301
        for i in lbl_range:
            mask[np.where(labels == i)[0][: config.size_labeled_data // config.num_label]] = True
            mask3[np.where(labels2 == i)[0][:]] = True
            min_num = min(min_num, np.where(labels == i)[0].shape[0])
        for i in lbl_range:
            mask2[np.where(labels == i)[0][:min_num]] = True
        labeled_indices, unlabeled_indices = indices[mask], indices[mask2]
        dev_indices = indices2[mask3]

    print 'labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], \
        'dev size', dev_indices.shape[0]

    labeled_loader = DataBatchLoader(config, train_set, labeled_indices, config.train_batch_size,
                                transform=transform, img_side=config.image_side)
    unlabeled_loader = DataBatchLoader(config, train_set, unlabeled_indices, config.train_batch_size_2,
                                  transform=transform, img_side=config.image_side)
    dev_loader = DataBatchLoader(config, test_set, dev_indices, config.dev_batch_size,
                            transform=transform, img_side=config.image_side)

    special_set = []
    # save label indices
    if save_ind:  # save step-1 / non-step inds
        np.save(lind_path, labeled_indices)
        np.save(uind_path, unlabeled_indices)
        np.save(tind_path, dev_indices)

    return labeled_loader, unlabeled_loader, dev_loader, special_set


def get_data_loaders_test(config):
    dataset = config.dataset
    if dataset == 'cifar':
        return get_cifar_loaders_test(config)
    elif dataset == 'stl10':
        return get_stl10_loaders_test(config)
    elif dataset == 'coil20':
        return get_coil20_loaders_test(config)
    elif dataset == 'imagenet10':
        return get_imagenet10_loaders_test(config)
    else:
        print("dataset wrong: {}".format(dataset))


def get_data_loaders(config):
    dataset = config.dataset
    if dataset == 'cifar':
        return get_cifar_loaders(config)
    elif dataset == 'stl10':
        return get_stl10_loaders(config)
    elif dataset == 'coil20':
        return get_coil20_loaders(config)
    elif dataset == 'imagenet10':
        return get_imagenet10_loaders(config)
    else:
        print("dataset wrong: {}".format(dataset))
