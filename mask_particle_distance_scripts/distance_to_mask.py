
'''
This takes a star file and a mask pattern and calculates the distance to the mask for each particle in the star file.
This is useful for analyzing the distance of particles to a membrane or other structure in cryo-EM data.
However: The way the name of the tomogram is defined based on the mask is not robust, and will require some tinkering.
'''

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import star_tools.star_tools as st
import pandas as pd
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
import mrcfile as mf

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class InputData:
    star_path: str
    mode: str = 'new'
    pixel_size_star: float = 1.7005
    mask_pattern: str | None = None
    pixel_size_tomogram: float = 13.604
    position: tuple | None = None
    output_folder: str = '.'
    threshold: float | None = None
    cpu: int = False
    verbose: bool = False


def calculate_normalized_coordinates(star: pd.DataFrame, pixel_size: float, pos: tuple, tomo_pixel_size: float = 1) -> pd.DataFrame:
    for c in 'XYZ':
        if f'_rlnOrigin{c}' in star.columns:
            star[c] = ((star[f'_rlnCoordinate{c}'] - star[f'_rlnOrigin{c}']) * pixel_size)/tomo_pixel_size
        elif f'_rlnOrigin{c}Angst' in star.columns:
            star[c] = (star[f'_rlnCoordinate{c}'] * pixel_size - star[f'_rlnOrigin{c}Angst']) / tomo_pixel_size
        else:
            logging.warning(f'No _rlnOrigin{c} or _rlnOrigin{c}Angst column found in star file. Using _rlnCoordinate{c} as is.')
            star[c] = star[f'_rlnCoordinate{c}'] * pixel_size / tomo_pixel_size
    if pos is not None:
        rotation_matrices = R.from_euler('ZYZ', star[['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']].values, degrees=True)
        offset = rotation_matrices.apply(pos, inverse=True) * pixel_size/tomo_pixel_size
        for idx, coord in enumerate('XYZ'):
            star[coord] = star[coord] + offset[:,idx]
    return star
def generate_mask_dict(file_pattern):
    if file_pattern[0] == '/':
        file_pattern = file_pattern[1:]
    return {m.parts[11] : m for m in Path('/').glob(file_pattern)}

def open_mask(mask_path):
    with mf.open(mask_path, mode='r', permissive=True) as mrc:
        return mrc.data.copy()

def main(inp: InputData):
    star = st._open_star(inp.star_path, mode=inp.mode)
    star = calculate_normalized_coordinates(star, inp.pixel_size_star, inp.position, inp.pixel_size_tomogram)
    masks = generate_mask_dict(inp.mask_pattern)
    combined_star = pd.DataFrame()
    with Pool(inp.cpu) as p:
        results = p.map(extract_distance_to_mask, ((star, masks, tomo) for tomo in star._rlnMicrographName.unique()))
    combined_star = pd.concat(results, ignore_index=True)
    st._writeStar(combined_star, Path(inp.output_folder) / 'distance_to_mask.star', inp.mode)
    c_star = combined_star.loc[combined_star.dist < inp.threshold]
    c_star = c_star.drop(['dist','X','Y','Z'], axis=1)
    st._writeStar(c_star, Path(inp.output_folder) / 'threshold_star.star', inp.mode)

def extract_distance_to_mask(args):
    star,masks,tomo = args
    tomo_star = star.loc[star['_rlnMicrographName'] == tomo].copy()
    mask = open_mask(masks[tomo[0:5]])
    distance_transform = distance_transform_edt(mask == 0)
    indices = np.round(tomo_star[['Z', 'Y', 'X']].values).astype(int)
    tomo_star['dist'] = distance_transform[indices[:,0], indices[:,1], indices[:,2]]
    return tomo_star
    

if __name__ == '__main__':
    inp = InputData(
        star_path = '/g/scb/mahamid/rasmus/cage/distance_analysis_2024/wt/WT_warp107_job013_run_ct24_it027_data.star',
        mode = 'old',
        mask_pattern='/g/scb/mahamid/rasmus/processing/membrane_prediction/all/all_out/predictions/membrane_all/*/membrane/post_processed_prediction.mrc',
        pixel_size_star = 1.7005,
        pixel_size_tomogram = 13.604,
        output_folder = '/scratch/kjeldsen/mask_test',
        threshold = 8,
        position = (50.8,-50.8,7.7),
        cpu = 40,
        verbose = True,)
    main(inp)
    



    #(410.67,388.09,331.22)