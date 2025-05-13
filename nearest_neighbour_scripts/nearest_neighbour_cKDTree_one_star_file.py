'''
Author: Rasmus K Jensen
Date: 2025-05-12
Description:
This script calculates the nearest neighbour distance within one star files, using cKDTree for fast nearest neighbour search.
'''

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
import star_tools.star_tools as st
from scipy.spatial import cKDTree 
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class InputData:
    star_file: str
    mode: str = 'new'
    pixel_size: float = 1
    output_folder: str = '.'
    threshold: float | None = None
    cpu: int = False
    verbose: bool = False


def map_coordinates(star: pd.DataFrame, pixel_size: float) -> pd.DataFrame:
    for c in 'XYZ':
        if f'_rlnOrigin{c}' in star.columns:
            star[c] = (star[f'_rlnCoordinate{c}'] - star[f'_rlnOrigin{c}']) * pixel_size
        elif f'_rlnOrigin{c}Angst' in star.columns:
            star[c] = (star[f'_rlnCoordinate{c}']) * pixel_size - star[f'_rlnOrigin{c}Angst']
        else:
            logging.warning(f'No _rlnOrigin{c} or _rlnOrigin{c}Angst column found in star file. Using _rlnCoordinate{c} as is.')
            star[c] = star[f'_rlnCoordinate{c}'] * pixel_size
    return star


def open_star(inp: InputData):
    star: pd.DataFrame = st._open_star(inp.star_file,mode=inp.mode)
    if inp.pixel_size == 1:
        logging.warning('Pixel size is 1. This is unlikely to be true, and probably coming from the default value. If the input is old style star file this is fine, but output will be in pixels.\n If this is new-style star file, and the file contains the _rlnOriginAngst columns it will give wrong results')
    star = map_coordinates(star, inp.pixel_size)
    return star

def main(inp: InputData):
    if inp.verbose:
        logging.getLogger().setLevel(logging.INFO)
    logging.info('Opening star file, and mapping coordinates')
    star = open_star(inp)
    logging.info('Calculating nearest neighbour')
    if not inp.cpu:
        inp.cpu = cpu_count()//2
    if inp.cpu == 1:
        logging.info('Using single core')
        results = list(map(measure_distances, [star.loc[star['_rlnMicrographName'] == mic] for mic in star['_rlnMicrographName'].unique()]))
    else:
        logging.info(f'Using {inp.cpu} cores, with multiprocessing')
        with Pool(inp.cpu) as p:
            results = p.map(measure_distances, [star.loc[star['_rlnMicrographName'] == mic] for mic in star['_rlnMicrographName'].unique()]) 
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
        st._writeStar(star, inp.output_folder / 'threshold_star.star', inp.mode)

def measure_distances(local_star: pd.DataFrame) -> pd.DataFrame:
    local_star = local_star.copy()
    if len(local_star) == 1:
        local_star['dist'] = np.nan
        return local_star
    tree = cKDTree(local_star[['X', 'Y', 'Z']].values)
    distances, _ = tree.query(local_star[['X', 'Y', 'Z']].values, k=2)
    local_star['dist'] = distances[:, 1]
    return local_star


if __name__ == '__main__':
    inp = InputData(
        star_file= '/g/scb/mahamid/rasmus/processing/cage_20240417/bin2/Refine3D/job036/run_data.star',
        mode = 'new',
        pixel_size = 3.401,
        threshold = 250,
        output_folder = '/g/scb/mahamid/rasmus/tt',
        cpu=1, # Multiprocessing is implemented, but really does not give much improvement unless you have many particles pr. tomogram (2.116 seconds with 1 cpu vs. 2.004 seconds for 60 cpus for 14,000 particles over 356 tomograms)
        verbose=True)
    main(inp)