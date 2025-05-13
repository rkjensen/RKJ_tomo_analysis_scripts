'''
Author: Rasmus K Jensen
Date: 2025-05-12
Description:
This script calculates the nearest neighbour distance within one star files, using cKDTree for fast nearest neighbour search. Thereafter uses networkx to identify polysome chains.

On EMBL cluster module load scikit-image is all you need :-)

Old took: (with 32 cores): 1m15.535s for 356 tomograms
New took: (1 core): 0m7.062s for 356 tomograms
'''

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
import star_tools.star_tools as st
from scipy.spatial import cKDTree 
import pandas as pd
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
import networkx as nx

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class InputData:
    star_file: str
    mode: str = 'new'
    pixel_size: float = 1
    output_folder: str = '.'
    pos1: tuple | None = None
    pos2: tuple | None= None
    threshold: float | None = None
    cpu: int | bool = False
    verbose: bool = False
    save_polysome_star: bool = False
    many_ribosomes: bool = False

def map_coordinates3(star: pd.DataFrame, pixel_size: float, pos1: tuple, pos2: tuple) -> pd.DataFrame:
    for c in 'XYZ':
        if f'_rlnOrigin{c}' in star.columns:
            star[c] = (star[f'_rlnCoordinate{c}'] - star[f'_rlnOrigin{c}']) * pixel_size
        elif f'_rlnOrigin{c}Angst' in star.columns:
            star[c] = (star[f'_rlnCoordinate{c}']) * pixel_size - star[f'_rlnOrigin{c}Angst']
        else:
            logging.warning(f'No _rlnOrigin{c} or _rlnOrigin{c}Angst column found in star file. Using _rlnCoordinate{c} as is.')
            star[c] = star[f'_rlnCoordinate{c}'] * pixel_size
    if pos1 is None:
        logging.warning('No position vectors supplied for pos1. This is likely not intended, but will continue anyway.')
        pos1 = (0,0,0)
    if pos2 is None:
        logging.warning('No position vectors supplied for pos2. This is likely not intended, but will continue anyway.')
        pos2 = (0,0,0)
    rotation_matrices = R.from_euler('ZYZ', star[['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']].values, degrees=True)
    offset1 = rotation_matrices.apply(pos1, inverse=True) * pixel_size
    offset2 = rotation_matrices.apply(pos2, inverse=True) * pixel_size
    for idx, c in enumerate(['X', 'Y', 'Z']):
        star[f'{c}1'] = star[c] + offset1[:,idx]
        star[f'{c}2'] = star[c] + offset2[:,idx]
    return star

def open_star(inp: InputData):
    star: pd.DataFrame = st._open_star(inp.star_file,mode=inp.mode)
    if inp.pixel_size == 1:
        logging.warning('Pixel size is 1. This is unlikely to be true, and probably coming from the default value. If the input is old style star file this is fine, but output will be in pixels.\n If this is new-style star file, and the file contains the _rlnOriginAngst columns it will give wrong results')
    star = map_coordinates3(star, inp.pixel_size, inp.pos1, inp.pos2)
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
    distance_star = pd.concat(results, ignore_index = True)
    logging.info(f'Saving nearest neighbour distances in {inp.output_folder}/nearest_neighbour.csv')
    save_distance_measurement_for_histogram(distance_star,inp)
    if inp.threshold is not None:
        logging.info('Identifying polysome chains')
        if not inp.many_ribosomes:
            polysome_star = identify_polysome_chains(distance_star, inp.threshold)
        else:
            polysome_star = identify_polysome_chains_pr_tomogram(distance_star, inp.threshold)
        logging.info(f'Save polysomes to {Path(inp.output_folder) / "polysome_chains.csv"}')
        save_polysome_chains(polysome_star, inp.output_folder, inp.save_polysome_star, inp.mode)
        return
    logging.info('No threshold given, not identifying polysome chains')

def save_distance_measurement_for_histogram(star: pd.DataFrame, inp: InputData, save_histogram: bool = False) -> None:
    inp.output_folder = Path(inp.output_folder)
    if not inp.output_folder.exists():
        inp.output_folder.mkdir(parents=True)
    star['dist'].to_csv(inp.output_folder / 'nearest_neighbour.csv', index=False, header=False)
        
def measure_distances(local_star: pd.DataFrame) -> pd.DataFrame:
    local_star = local_star.copy()
    mic = local_star._rlnMicrographName.unique()[0]
    local_star['idx'] =  [f"{mic}_{i}" for i in range(len(local_star))]
    if len(local_star) == 1:
        local_star['dist'] = np.nan
        local_star['neighbour_idx'] = np.nan
        return local_star
    tree = cKDTree(local_star[['X1', 'Y1', 'Z1']].values)
    distances, indices = tree.query(local_star[['X2', 'Y2', 'Z2']].values, k=2)
    neighbour_idx = np.where(indices[:,0] != np.arange(len(indices)), 0, 1)
    local_star['neighbour_idx'] =  [f"{mic}_{i}" for i in indices[np.arange(len(indices)), neighbour_idx]]
    local_star['dist'] = distances[np.arange(len(distances)), neighbour_idx]
    return local_star

def save_polysome_chains(star: pd.DataFrame, output_folder: str, save_polysome_star = False, mode='new'):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    polysome_lengths = star['chain_id'].value_counts().drop(-1, errors='ignore').tolist()
    with open(output_folder / "polysome_chains.csv", 'w') as f:
        f.write('Polysome length:,' + ','.join([str(x) for x in range(2,max(polysome_lengths)+1)]) + '\n')
        f.write('Polysome count:,' +','.join([str(polysome_lengths.count(x)) for x in range(2,max(polysome_lengths)+1)]))
    if save_polysome_star:
        star.rename({'chain_id' : '_rlnUnknownLabel'}, axis=1, inplace=True)
        st._writeStar(star.drop(['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'dist', 'neighbour_idx', 'idx', 'X', 'Y', 'Z'], axis=1), output_folder / 'polysome_chains.star', mode=mode)

def identify_polysome_chains(star: pd.DataFrame, threshold: float) -> pd.DataFrame:
    G = generate_graph(star, threshold)
    id_to_df_index = {row['idx']: i for i, row in star.iterrows()} # Maps the pr. tomo index to global index
    chain_ids = np.full(len(star), -1)
    for chain_id, component in enumerate(nx.weakly_connected_components(G)):
        for node in component:
            chain_ids[id_to_df_index[node]] = chain_id
    star['chain_id'] = chain_ids
    return star

def identify_polysome_chains_pr_tomogram(star: pd.DataFrame, threshold: float) -> pd.DataFrame:
    chain_ids = np.full(len(star), -1)
    current_chain = 0

    for mic in star['_rlnMicrographName'].unique():
        local_star = star[star['_rlnMicrographName'] == mic]
        G = generate_graph(local_star, threshold)
        
        id_to_df_index = {row['idx']: i for i, row in local_star.iterrows()}
        for component in nx.weakly_connected_components(G):
            for node in component:
                chain_ids[id_to_df_index[node]] = current_chain
            current_chain += 1

    star['chain_id'] = chain_ids
    return star


def generate_graph(star: pd.DataFrame, threshold: float) -> nx.DiGraph:
    G = nx.DiGraph()
    valid = star['dist'] < threshold
    edges = zip(star.loc[valid, 'idx'], star.loc[valid, 'neighbour_idx'])
    G.add_edges_from((n, nbr) for n, nbr in edges)
    return G

if __name__ == '__main__':
    inp = InputData(
        star_file= '/g/scb/mahamid/rasmus/from_others/from_Laing/for_Rasmus/previous_files/WT_warp107_job013_run_ct24_it027_data.star',
        mode = 'old', # pre- (old) or post- (new) RELION 3.1
        pixel_size = 1.700500, # Given in Angstrom/pixel
        threshold = 80, # Given in Angstrom,
        pos1 = [x/1.700500 - 256/2 for x in (123.14,239.22,243.64)], #tuple or list given in pixels. Should be peptide entry tunnel
        pos2 = [x/1.700500 - 256/2 for x in (189.72,253.07,320.75)], #tuple or list given in pixels. Should be peptide exit tunnel
        output_folder = '/g/scb/mahamid/rasmus/tt/test_new3',
        cpu=1, # Multiprocessing is implemented, but really does not give much improvement unless you have huge number of particles pr. tomogram.
        verbose=True, # Will give info during run. False gives only warnings and errors
        save_polysome_star= False, # Set to true if you want a star file to continue processing with each polysome marked in the _rlnUnknownLabel column
        many_ribosomes = False, #If this is set to False, one graph will be made for the dataset, whereas if it is True, one graph is made per tomogram. This will be faster if you have !MANY! polysomes in your data but adds some extra overhead.
    )
    main(inp)