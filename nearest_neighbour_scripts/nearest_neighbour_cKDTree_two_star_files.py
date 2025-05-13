'''
Author: Rasmus K Jensen
Date: 2025-05-12
Description:
This script calculates the nearest neighbour distance between two star files, using cKDTree for fast nearest neighbour search.

It takes two star files as input, and calculates the distance between each particle in the first star file to the nearest particle in the second star file.
It can also take a threshold distance, and will output a new star file with only the particles that are within the threshold distance.

Has been checked against brute force nearest neighbour search, and gives the exact same results, but runs 50x faster running on 1 core vs brute force running on 32 cores.
'''


from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Tuple
import star_tools.star_tools as st
from scipy.spatial import cKDTree 
import pandas as pd
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class InputData:
    star_file_1: str
    star_file_2: str
    mode_1: str = 'new'
    mode_2: str = 'new'
    pixel_size_1: float = 1
    pixel_size_2: float = 1
    pos_1: tuple = None
    pos_2: tuple = None
    output_folder: str = '.'
    threshold: float | None = None
    cpu: int = False
    verbose: bool = False


def map_coordinates3(star: pd.DataFrame, pixel_size: float, pos: tuple) -> pd.DataFrame:
    for c in 'XYZ':
        if f'_rlnOrigin{c}' in star.columns:
            star[c] = (star[f'_rlnCoordinate{c}'] - star[f'_rlnOrigin{c}']) * pixel_size
        elif f'_rlnOrigin{c}Angst' in star.columns:
            star[c] = (star[f'_rlnCoordinate{c}']) * pixel_size - star[f'_rlnOrigin{c}Angst']
        else:
            logging.warning(f'No _rlnOrigin{c} or _rlnOrigin{c}Angst column found in star file. Using _rlnCoordinate{c} as is.')
            star[c] = star[f'_rlnCoordinate{c}'] * pixel_size
    if pos is not None:
        rotation_matrices = R.from_euler('ZYZ', star[['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']].values, degrees=True)
        offset = rotation_matrices.apply(pos, inverse=True) * pixel_size
        for idx, coord in enumerate(['X', 'Y', 'Z']):
            star[coord] = star[coord] + offset[:,idx]
    return star

def open_star(star_file: str, mode: str, pixel_size: float, pos: tuple) -> pd.DataFrame:
    star: pd.DataFrame = st._open_star(star_file,mode=mode)
    if pixel_size == 1:
        logging.warning(f'Pixel size is 1 for {star_file}. This is unlikely to be true, and probably coming from the default value. If the input is old style star file this is fine, but output will be in pixels.\n If this is new-style star file, and the file contains the _rlnOriginAngst columns it will give wrong results')
    star = map_coordinates3(star, pixel_size, pos)
    return star

def main(inp: InputData):    
    if inp.verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info('Opening star files, and mapping coordinates')
    star1,star2 = open_star(inp.star_file_1,inp.mode_1,inp.pixel_size_1, inp.pos_1), open_star(inp.star_file_2,inp.mode_2,inp.pixel_size_2, inp.pos_2)
    logging.info('Calculating nearest neighbour')
    if not inp.cpu:
        inp.cpu = cpu_count()//2
    if inp.cpu == 1:
        logging.info('Using single core')
        results = list(map(measure_distances, [(star1.loc[star1['_rlnMicrographName'] == mic], star2.loc[star2['_rlnMicrographName'] == mic]) for mic in star1['_rlnMicrographName'].unique()]))
    else:
        logging.info(f'Using {inp.cpu} cores, with multiprocessing')
        with Pool(inp.cpu) as p:
            results = p.map(measure_distances, [(star1.loc[star1['_rlnMicrographName'] == mic], star2.loc[star2['_rlnMicrographName'] == mic]) for mic in star1['_rlnMicrographName'].unique()]) 
    output_star = pd.concat(results, ignore_index = True)
    logging.info(f'Saving output in {inp.output_folder}')
    save_output(output_star,inp)

def save_output(star: pd.DataFrame, inp: InputData) -> None:
    inp.output_folder = Path(inp.output_folder)
    if not inp.output_folder.exists():
        inp.output_folder.mkdir(parents=True)
    star['dist'].to_csv(inp.output_folder / 'nearest_neighbour.csv', index=False, header=False)
    if inp.threshold is not None:
        star = star.loc[star['dist'] < inp.threshold]
        star = star.drop(['X','Y','Z','dist'], axis=1)
        st._writeStar(star, inp.output_folder / 'threshold_star.star', inp.mode_1)

def measure_distances(star: Tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    star1,star2 = (s.copy() for s in star)
    if len(star2) == 0:
        star1['dist'] = np.nan
        return star1
    tree = cKDTree(star2[['X', 'Y', 'Z']].values)
    distances, _ = tree.query(star1[['X', 'Y', 'Z']].values, k=1)
    star1['dist'] = distances
    return star1


if __name__ == '__main__':
    inp = InputData(
        star_file_1= '/g/scb/mahamid/rasmus/processing/cage_20240417/bin2/Refine3D/job036/run_data.star',
        mode_1 = 'new',
        pos_1 = (2,14,29),
        pos_2 = (50.8,-50.8,7.7),
        pixel_size_1 = 3.401,
        star_file_2= '/g/scb/mahamid/rasmus/from_others/from_Laing/for_Rasmus/previous_files/WT_warp107_job013_run_ct24_it027_data.star',
        mode_2 = 'old',
        pixel_size_2 = 1.7005,
        threshold = 150,
        output_folder = '/g/scb/mahamid/rasmus/tt/tt4',
        cpu=1, # Multiprocessing is implemented, but really does not give much improvement unless you have many particles pr. tomogram (2.116 seconds with 1 cpu vs. 2.004 seconds for 60 cpus for 14,000 particles over 356 tomograms)
        verbose=True)
    main(inp)