import numpy as np
import subprocess
import random
import progressbar
from glob import glob
from skimage import io
from os import makedirs
from errno import EEXIST
from os.path import isdir

np.random.seed(5) # for reproducibility
progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])

def mkdir_p(path):
    """
    mkdir -p function, makes folder recursively if required
    :type path: basestring
    :param path:
    :return:
    """
    try:
        makedirs(path)
    except OSError as exc:  
        if exc.errno == EEXIST and isdir(path):
            pass
        else:
            raise

class BrainPipeline(object):
    '''
    A class for processing brain scans for one patient
    INPUT:  (1) filepath 'path': path to directory of one patient. Contains following mha files:
            flair, t1, t1c, t2, ground truth (gt)
            (2) bool 'n4itk': True to use n4itk normed t1 scans (defaults to True)
            (3) bool 'n4itk_apply': True to apply and save n4itk filter to t1 and t1c scans for given patient. This will only work if the
    '''
    print(object);
    def __init__(self, path, n4itk = False, n4itk_apply = False):           #agr false n4itk ko false kr den to normalize kr den ge
        self.path = path
        self.n4itk = n4itk
        self.n4itk_apply = n4itk_apply
        self.modes = ['flair', 't1', 't1c', 't2', 'gt']
        # slices=[[flair x 155], [t1], [t1c], [t2], [gt]], 155 per modality
        # print(self.slices_by_mode);
        # print('aaaaaaa');
        self.slices_by_mode, n = self.read_scans()
        # [ [slice1 x 5], [slice2 x 5], ..., [slice155 x 5]]
        self.slices_by_slice = n
        self.normed_slices = self.norm_slices()

    def read_scans(self):
        '''
        goes into each modality in patient directory and loads individual scans.
        transforms scans of same slice into strip of 5 images
        '''
        print ('Loading scans...')
        slices_by_mode =np.zeros((5, 176, 216, 160), dtype=np.float)     #changing to 20 256 256 np.zeros((155, 5, 240, 240), dtype=float)
        print(slices_by_mode);                        #[np.zeros((155, 240, 240)), np.zeros((5, 240, 240)), np.zeros((240, 240))]   dtype=np.int
        print('ssss')
        slices_by_slice = np.zeros((176, 5, 216, 160), dtype=np.float)       #orignally ((155, 5, 240, 240))
     
        t2 = glob(self.path + '/*_T2*/*.mha')
        gt = glob(self.path + '/*more*/*.mha')
        t1s = glob(self.path + '/**/*T1*.mha')
        t1_n4 = glob(self.path + '/*T1*/*_n.mha')
        print(flair,t2)
        t1 = [scan for scan in t1s if scan not in t1_n4]
        scans = [flair[0], t1[0], t1[1], t2[0], gt[0]]  # directories to each image (5 total)
        # print('sssss');
        print(scans);
        if self.n4itk_apply:
            print ('-> Applyling bias correction...')
            for t1_path in t1:
                self.n4itk_norm(t1_path) # normalize files
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
            print('idher aya');
        elif self.n4itk:
            scans = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
            print('pagal nhi idher aya');
        for scan_idx in range(5):			#changed xrange into range
            # read each image directory, save to self.slices
            # print(scans) #(io.imread(labels[label_idx], plugin = 'simpleitk'))
            # slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
            print(io.imread(scans[scan_idx], plugin='simpleitk').astype(float).shape)
            print(scans[scan_idx])
            print('*' * 100)
            try:
                slices_by_mode[scan_idx] = io.imread(scans[scan_idx], plugin='simpleitk').astype(float)
            except:
                continue
        for mode_ix in range(slices_by_mode.shape[0]): # modes 1 thru 5
            for slice_ix in range(slices_by_mode.shape[1]): # slices 1 thru 155
                slices_by_slice[slice_ix][mode_ix] = slices_by_mode[mode_ix][slice_ix] # reshape by slice
        return slices_by_mode, slices_by_slice

    def norm_slices(self):
        '''
        normalizes each slice in self.slices_by_slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
        if n4itk == True, will apply n4itk bias correction to T1 and T1c images
        '''
        print ('Normalizing slices...')

        normed_slices = np.zeros((176, 5, 216, 160))
        for slice_ix in range(155):
            normed_slices[slice_ix][-1] = self.slices_by_slice[slice_ix][-1]
            for mode_ix in range(4):
                normed_slices[slice_ix][mode_ix] =  self._normalize(self.slices_by_slice[slice_ix][mode_ix])
        print ('Done.')
        return normed_slices

    def _normalize(self, slice):
        '''
        INPUT:  (1) a single slice of any given modality (excluding gt)
                (2) index of modality assoc with slice (0=flair, 1=t1, 2=t1c, 3=t2)
        OUTPUT: normalized slice
        '''
        b, t = np.percentile(slice, (0.5,99.5))
        slice = np.clip(slice, b, t)
        if np.std(slice) == 0:
            return slice
        else:
            return (slice - np.mean(slice)) / np.std(slice)

    def save_patient(self, reg_norm_n4, patient_num):
        '''
        INPUT:  (1) int 'patient_num': unique identifier for each patient
                (2) string 'reg_norm_n4': 'reg' for original images, 'norm' normalized images, 'n4' for n4 normalized images
        OUTPUT: saves png in Norm_PNG directory for normed, Training_PNG for reg
        '''
        print ('Saving scans for patient {}...'.format(patient_num))
        progress.currval = 0
        a=0;
        b=176; #155
        print(a,b)
        if reg_norm_n4 == 'norm': #saved normed slices
            for slice_ix in range(a, b): # reshape to strip
                print(slice_ix)
                strip = self.normed_slices[slice_ix].reshape(1080, 160)
                if np.max(strip) != 0: # set values < 1
                    strip /= np.max(strip)
                if np.min(strip) <= -1: # set values > -1
                    strip /= abs(np.min(strip))
                # save as patient_slice.png
                io.imsave('/Norm_PNG/{}_{}.png'.format(patient_num, slice_ix), strip)
        elif reg_norm_n4 == 'reg':
            for slice_ix1 in range(a, b):
                print(slice_ix1);
                strip = self.slices_by_slice[slice_ix1].reshape(1080, 160)
                if np.max(strip) != 0:
                    strip /= np.max(strip)
                io.imsave('/Training_PNG/{}_{}.png'.format(patient_num, slice_ix1), strip)
        else:
            for slice_ix2 in range(a, b): # reshape to strip
                print(slice_ix2)
                strip = self.normed_slices[slice_ix2].reshape(1080, 160)
                if np.max(strip) != 0: # set values < 1
                    strip /= np.max(strip)
                if np.min(strip) <= -1: # set values > -1
                    strip /= abs(np.min(strip))
                # save as patient_slice.png
                io.imsave('/n4_PNG/{}_{}.png'.format(patient_num, slice_ix2), strip)

    def n4itk_norm(self, path, n_dims=3, n_iters='[20,20,10,5]'):
        '''
        INPUT:  (1) filepath 'path': path to mha T1 or T1c file
                (2) directory 'parent_dir': parent directory to mha file
        OUTPUT: writes n4itk normalized image to parent_dir under orig_filename_n.mha
        '''
        output_fn = path[:-4] + '_n.mha'
        # run n4_bias_correction.py path n_dim n_iters output_fn
        subprocess.call('python n4_bias_correction.py ' + path + ' ' + str(n_dims) + ' ' + n_iters + ' ' + output_fn, shell = True)


def save_patient_slices(patients, type):
    '''
    INPUT   (1) list 'patients': paths to any directories of patients to save. for example- glob("Training/HGG/**")
            (2) string 'type': options = reg (non-normalized), norm (normalized, but no bias correction), n4 (bias corrected and normalized)
    saves strips of patient slices to approriate directory (Training_PNG/, Norm_PNG/ or n4_PNG/) as patient-num_slice-num
    '''
    for patient_num, path in enumerate(patients):
        print(path);
        a = BrainPipeline(path)
        a.save_patient(type, patient_num)


def save_labels(fns):
    '''
    INPUT list 'fns': filepaths to all labels
    '''
    # print(fns);
    # print('aaa')
    progress.currval = 0
    for label_idx in progress(range(len(labels))):
        slices = (io.imread(labels[label_idx], plugin = 'simpleitk'))
        # print(slices);
        # print(labels);
        for slice_idx in range(len(slices)):
            # print(slices)
            io.imsave('/Labels/{}_{}L.png'.format(label_idx, slice_idx), slices[slice_idx])


if __name__ == '__main__':
    labels = glob('/Original_Data/Training/HGG/**/*more*/**.mha')
    # print(labels);
    # print('aaa');
    save_labels(labels)
    patients = glob('/Training/HGG/**')
    save_patient_slices(patients, 'reg')
    save_patient_slices(patients, 'norm')
    save_patient_slices(patients, 'n4')