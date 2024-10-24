import sys; sys.path.insert(0, "build/python")

import numpy as np
import pandas as pd

import drjit as dr
import mitsuba as mi

mi.set_variant("llvm_rgb")

def parse_csv_dataset(filename: str):
    df = pd.read_csv(filename)
    df.pop('RMSE')
    df.pop('MAE')
    df.pop('Volume')
    df.pop('Azimuth')

    arr = df.to_numpy()

    sort_args = np.lexsort([arr[::, 0], arr[::, 1]])

    etas = np.bincount(arr[sort_args, 1].astype(np.uint32))
    print(len(etas), etas)

    simplified_arr = arr[sort_args, 2:]
    print(simplified_arr)



parse_csv_dataset("sunsky-testing/res/datasets/model_hosek.csv")

