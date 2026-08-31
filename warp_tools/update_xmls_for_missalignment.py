import torch
from warpylib import TiltSeries
from pathlib import Path
import argparse

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x","--xml_tilt_series", type=str, required=True, help="Path to warp_tilteseries processing xml files")
    parser.add_argument("-s","--original_stack_shape", type=int, default= (4096,4096),nargs=2, help="Original stack shape (x,y) used in warptools")
    parser.add_argument("-p","--original_pixel_size", type=float, default=2, help="Original pixel size used in warptools")
    parser.add_argument("-v","--volume_shape", type=int, default=(4096,4096,1500), nargs=3, help="Volume shape (x,y,z) used in warptools")
    return parser.parse_args()

def main():
    args = parse_input()
    original_stack_shape = args.original_stack_shape
    volume_shape = args.volume_shape
    original_pixel_size = args.original_pixel_size

    # find all xml_files in current directory and update them
    xml_files = Path(args.xml_tilt_series).resolve().glob('*.xml')
    for xml in xml_files:
        tilt_series = TiltSeries(xml)

        # Set physical dimensions
        tilt_series.image_dimensions_physical = torch.tensor(
            [
                original_stack_shape[0] * original_pixel_size,
                original_stack_shape[1] * original_pixel_size,
            ],
            dtype=torch.float32,
        )
        tilt_series.volume_dimensions_physical = torch.tensor(
            [
                volume_shape[0] * original_pixel_size,
                volume_shape[1] * original_pixel_size,
                volume_shape[2] * original_pixel_size,
            ],
            dtype=torch.float32,
        )

        tilt_series.save_meta(xml)

if __name__ == "__main__":
    main()
