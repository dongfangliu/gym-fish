import  numpy as np
import glob
import  os
def all_npz_to_one(file_format,file_output):
    np_files = glob.glob(file_format)
    print("Processing following files.....")
    for f in np_files:
        print(f)
    all_arrays = {}

    for npfile in np_files:
        array = np.load(npfile)
        for k in array.files:
            if k not in all_arrays.keys():
                all_arrays[k] = array[k]
            else:
                all_arrays[k] = np.concatenate((all_arrays[k],array[k]))

    np.savez_compressed(file_output,**all_arrays)


