from dataclasses import dataclass

import numpy as np
import star_tools.star_tools as st
from pandas import DataFrame
from scipy.spatial.transform import Rotation as R

@dataclass
class InputData:
    star_path: str
    mode: str = 'new'
    z_dim_old: int = 1800
    z_dim_new: int = 2000
    output: str = './rotated.star'
    
    def __post_init__(self):
        self.star: DataFrame = st._open_star(self.star_path, self.mode)
        self.star['_rlnCoordinateZ'] = self.z_dim_old - self.star['_rlnCoordinateZ']
        self.rescale_z()
    
    def rescale_z(self):
        z_shift = (self.z_dim_new - self.z_dim_old)/2
        self.star['_rlnCoordinateZ'] = self.star['_rlnCoordinateZ'] + z_shift

def calculate_transformed_eulerangles(star: DataFrame) -> DataFrame:
    rotation_matrices = R.from_euler("ZYZ", star[['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']].values, degrees=True).as_matrix()
    flipZ = np.diag([1,1,-1])
    R_flipped = np.einsum('ij,njk,kl->nil', flipZ, rotation_matrices, flipZ)
    flipped_eulers = R.from_matrix(R_flipped).as_euler("ZYZ", degrees=True)
    star[['_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']] = flipped_eulers
    return star

def main(inp: InputData):
    star = calculate_transformed_eulerangles(inp.star)
    st._writeStar(star,inp.output,inp.mode)


if __name__ == '__main__':
    inp = InputData(
        star_path = '/struct/mahamid/rasmus/data/ribo_memb/Refine3D/job008/run_data.star',
        mode = 'new',
        z_dim_old = 1800/4,
        z_dim_new = 2000/4,
        output = '/struct/mahamid/rasmus/data/ribo_memb/flipped.star',
    )
    main(inp)