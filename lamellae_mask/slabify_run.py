'''
This is just a small wrapper to run slabify on a set of tomograms with configurable parameters, set up to run easily on the EMBL cluster.
'''
from pathlib import Path
from subprocess import run
from dataclasses import dataclass
import logging
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class InputData:
    mrc_folder: str
    output_folder: str
    tomo_pattern: str = '*.mrc'
    conda: str = 'slabify'
    conda_env: str = 'Miniforge3'
    iterations: int = 5
    offset: int | None = None
    percentile: float = 95
    submit: bool = False
    verbose: bool = False

def main(inp: InputData) -> None:
    tomos = list(Path(inp.mrc_folder).resolve().glob(inp.tomo_pattern))
    if inp.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if not tomos:
        logging.error(f'No tomograms found in {inp.mrc_folder} matching pattern {inp.tomo_pattern}')
    logging.info(f'Found {len(tomos)} tomograms in {inp.mrc_folder} matching pattern {inp.tomo_pattern}')
    output_folder = Path(inp.output_folder).resolve()
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    with open(output_folder / 'submission.sh', 'w') as f:
        f.write(f'#!/bin/bash\nmodule load {inp.conda_env}\nsource activate {inp.conda}\n')
        for tomo in tomos:
            cmd = f'slabify --input {tomo} --output {output_folder / tomo.name} --iterations {inp.iterations} --percentile {inp.percentile}'
            if inp.offset:
                cmd += f' --offset {inp.offset}'
            f.write(cmd + '\n')
    if inp.submit:
        logging.info(f'Submitting job to cluster with command: srun --time=01-00:00 -p htc-el8 --mem 60G --tasks=1 --cpus-per-task=20 bash {output_folder / "submission.sh"}')
        run(f'srun -p htc-el8 --mem 60G --tasks=1 --time=01-00:00 --cpus-per-task=20 bash {output_folder / "submission.sh"}', shell=True)
    else:
        logging.info(f'Not submitting job, run the following command to execute: bash {output_folder / "submission.sh"}')

if __name__ == '__main__':
    inp = InputData(
        mrc_folder = '/struct/mahamid/rasmus/data/20250423/warp/deconv/reconstruction/deconv',
        output_folder= '/struct/mahamid/rasmus/data/20250423/warp/dewedge/masks/mask',
        tomo_pattern='*_13.00Apx.mrc',
        conda = 'slabify',
        submit=True,
        verbose= True,
    )
    main(inp)