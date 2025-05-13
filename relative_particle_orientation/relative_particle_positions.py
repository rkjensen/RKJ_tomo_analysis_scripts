'''
Author: Rasmus K Jensen
This script analyzes particle positions and orientations from a RELION STAR file to identify
and filter particles based on the transformed (rotated) spatial relationships between neighboring
particles within the same micrograph.

For each pair of nearby particles (within a distance threshold), it computes the relative
position vector and applies the first particle's Euler angle rotation and its inverse. The
script then filters these pairs based on user-defined thresholds applied to the inverse-rotated
coordinates (typically to define a region of interest in rotated space).

Main steps:
1. Load and preprocess the STAR file (origin correction, coordinate scaling).
2. Identify particle pairs closer than the given threshold.
3. Compute relative vectors and apply ZYZ Euler rotations.
4. Filter particle pairs based on rotated X and Y bounds.
5. Write selected particle data to:
   - A new STAR file with the filtered subset of particles.
   - A CSV file containing rotated and inverse-rotated relative vectors.

Typical use case: selecting particles based on spatial configuration or orientation-transformed
neighborhood structure — e.g., to isolate specific structural arrangements like heptamers or
stacked assemblies in subtomogram averaging.

Inputs:
- RELION STAR file with particle coordinates and angles.
- Pixel size and distance threshold (Ångströms).
- Rotated-space X and Y thresholds for filtering.

Outputs:
- Filtered STAR file (`selected.star`).
- CSV file with transformed relative vectors.

Dependencies:
- scipy
- pandas
- star_tools (custom module)
'''
from scipy.spatial.transform import Rotation as R
from pandas import DataFrame
import star_tools.star_tools as st
from dataclasses import dataclass
from typing import List

@dataclass
class InputData:
    star_file: str
    mode: str
    pixel_size: float
    threshold: float
    output_file: str
    Y_thresholds: tuple
    X_thresholds: tuple

class ParticlePosition:
    def __init__(self,idx,position: tuple,euler_angles: tuple):
        self.position = position
        self.rot,self.tilt,self.psi = euler_angles
        self.rotation_matrix = R.from_euler('ZYZ',(self.rot,self.tilt,self.psi),degrees=True)
        self.idx = idx

    @property
    def rotate_ref(self):
        return self.rotation_matrix.apply(self.position,inverse=False)

    @property
    def rotate_ref_inv(self):
        return self.rotation_matrix.apply(self.position,inverse=True)

    def save_rotated_reference(self):
        self.x,self.y,self.z = self.rotate_ref_inv

def open_star_file(inp = InputData) -> DataFrame:
    star = st._open_star(inp.star_file,mode=inp.mode)
    if inp.mode == 'new':
        star = st._fix_origins(star,inp.pixel_size)
    star = calculate_center(star,inp.pixel_size)
    return star

def calculate_center(star: DataFrame ,pixel_size: float) -> DataFrame:
    for coord in ['X','Y','Z']:
        if f'_rlnOrigin{coord}' in star.columns:
            star[coord] = (star[f'_rlnCoordinate{coord}'] - star[f'_rlnOrigin{coord}']) * pixel_size
        else:
            star[coord] = star[f'_rlnCoordinate{coord}'] * pixel_size
    return star

def calculate_particle_positions(star,threshold) -> None:
    particle_positions = []
    micrograph_list = star['_rlnMicrographName'].unique()
    for micrograph in micrograph_list:
        local_star = star.loc[star["_rlnMicrographName"] == micrograph]
        particle_positions.extend(calculate_local_particle_positions(local_star,threshold))
    return particle_positions

def calculate_distance(particle1, particle2) -> float:
    return  (
        (particle1['X']-particle2['X'])**2 +
        (particle1['Y']-particle2['Y'])**2 +
        (particle1['Z']-particle2['Z'])**2)**0.5

def find_particles_positions(particle1,particle2,threshold,idx):
    distance = calculate_distance(particle1,particle2)
    if distance < threshold:
        position = (particle1['X'] - particle2['X'],
                    particle1['Y'] - particle2['Y'],
                    particle1['Z'] - particle2['Z'])
        euler_angles = (particle1['_rlnAngleRot'],particle1['_rlnAngleTilt'],particle1['_rlnAnglePsi'])
        return ParticlePosition(idx,position, euler_angles)
    return None

def write_particles(particle_position_list: list,output_file: str) -> None:
    with open(output_file, 'w') as f:
        for particle_position in particle_position_list:
            f.write(f'{particle_position.rotate_ref[0]},{particle_position.rotate_ref[1]},{particle_position.rotate_ref[2]},{particle_position.rotate_ref_inv[0]},{particle_position.rotate_ref_inv[1]},{particle_position.rotate_ref_inv[2]}\n')

def calculate_local_particle_positions(local_star: DataFrame,threshold:float) -> list:
    local_particle_positions = []
    for idx, particle1 in local_star.iterrows():
        for idy,particle2 in local_star.iterrows():
            if idy == idx:
                continue
            prt_position = find_particles_positions(particle1,particle2,threshold,idx=idx)
            if prt_position is not None:
                local_particle_positions.append(prt_position)
    return local_particle_positions
    
def locate_particles(particle_positions: List[ParticlePosition],X_threshold,Y_threshold):
    particle_ids = []
    x_min, x_max = X_threshold
    y_min, y_max = Y_threshold
    for particle in particle_positions:
        particle.save_rotated_reference()
        if x_min<particle.x<x_max and y_min<particle.y<y_max:
            particle_ids.append(particle.idx)
    return particle_ids

def main(inp: InputData) -> None:
    star = open_star_file(inp)
    particle_positions = calculate_particle_positions(star,inp.threshold)
    particle_ids = locate_particles(particle_positions,inp.X_thresholds,inp.Y_thresholds)
    particle_ids = list(set(particle_ids))
    neighbour_star = star.iloc[particle_ids]
    neighbour_star = neighbour_star.drop(['X','Y','Z'],axis=1)
    st._writeStar(neighbour_star,'/g/scb/mahamid/rasmus/edmp/selected.star',mode='old')
    new_particle_positions = []
    for prt in particle_positions:
        if prt.idx in particle_ids:
            new_particle_positions.append(prt)
    write_particles(new_particle_positions,inp.output_file)

if __name__ == '__main__':
    inp = InputData(
        star_file = '/g/scb/mahamid/rasmus/heptamer/relion/20240603_tm/bin2/small_box/Refine3D/job007/run_it006_data.star',
        mode = 'new',
        pixel_size = 3.401,
        threshold = 300,
        output_file = '/g/scb/mahamid/rasmus/edmp/pos_all_new3.csv',
        X_thresholds=(-175, -75),
        Y_thresholds=(-200,25),
    )
    main(inp)