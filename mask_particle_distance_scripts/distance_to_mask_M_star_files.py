
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
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
    rotation_matrices = R.from_euler('ZYZ', star[['_wrpAngleRot1', '_wrpAngleTilt1', '_wrpAnglePsi1']].values, degrees=True)
    offset = rotation_matrices.apply(pos, inverse=True)
    for idx, coord in enumerate('XYZ'):
        star[coord] = (star[f'_wrpCoordinate{coord}1'] + offset[:,idx])*pixel_size / tomo_pixel_size
    return star
def generate_mask_dict(file_pattern):
    if file_pattern[0] == '/':
        file_pattern = file_pattern[1:]
    return {m.name.split('_8_corrected_MemBrain_seg_v10_beta.ckpt_segmented')[0] + '.tomostar' : m for m in Path('/').glob(file_pattern)}

def open_mask(mask_path):
    with mf.open(mask_path, mode='r', permissive=True) as mrc:
        return mrc.data.copy()

def main(inp: InputData):
    star = st._open_star(inp.star_path, mode=inp.mode)
    star = calculate_normalized_coordinates(star, inp.pixel_size_star, inp.position, inp.pixel_size_tomogram)
    masks = generate_mask_dict(inp.mask_pattern)
    combined_star = pd.DataFrame()
    if inp.cpu:
        with Pool(inp.cpu) as p:
            results = p.map(extract_distance_to_mask, ((star, masks, tomo) for tomo in star._wrpSourceName.unique()))
    else:
        results = list(map(extract_distance_to_mask, ((star, masks, tomo) for tomo in star._wrpSourceName.unique())))
    combined_star = pd.concat(results, ignore_index=True)
    st._writeStar(combined_star, Path(inp.output_folder) / 'distance_to_mask.star', inp.mode)
    c_star = combined_star.loc[combined_star.dist < inp.threshold]
    c_star = c_star.drop(['dist','X','Y','Z'], axis=1)
    st._writeStar(c_star, Path(inp.output_folder) / 'threshold_star.star', inp.mode)

def extract_distance_to_mask(args):
    star,masks,tomo = args
    tomo_star = star.loc[star['_wrpSourceName'] == tomo].copy()
    mask = open_mask(masks[tomo])
    distance_transform = distance_transform_edt(mask == 0)
    indices = np.round(tomo_star[['Z', 'Y', 'X']].values).astype(int)
    tomo_star['dist'] = distance_transform[indices[:,0], indices[:,1], indices[:,2]]
    return tomo_star
    

if __name__ == '__main__':
    inp = InputData(
        star_path = '/scratch/kjeldsen/syn/10442/warp/merged/relion/ribo2/M/species/ribosome_497e9c8e/ribosome_particles.star',
        mode = 'old',
        mask_pattern='/scratch/kjeldsen/syn/10442/warp/merged/afterMCtf/membrain/*.mrc',
        pixel_size_star = 1,
        pixel_size_tomogram = 8,
        output_folder = '/scratch/kjeldsen/syn/10442/warp/merged/relion/ribo2/M/near_membrane',
        threshold = 8,
        position = [(d/3-234/2)*3 for d in (410.67,388.09,331.22)],
        cpu = None,
        verbose = True,)
    main(inp)
    



    