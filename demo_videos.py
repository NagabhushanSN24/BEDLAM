# Runs demo.py or demox.py on individual frames of a set of videos

import argparse
import datetime
import os
import shutil
import time
import traceback

from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm


def execute_shell_command(cmd: str):
    print(cmd)
    os.system(cmd)
    return


def delete_directory(dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath)
    return


def clean_directory(dirpath: Path):
    delete_directory(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return


def extract_frames(video_path: Path, frames_dirpath: Path):
    frames_dirpath.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -i {video_path.as_posix()} -start_number 0 {frames_dirpath.as_posix()}/%04d.png'
    execute_shell_command(cmd)
    return


def merge_frames(video_name, output_dirpath: Path, frame_rate: int = 60):
    frames_dirpath = output_dirpath / f'{video_name}/frames'
    output_filepath = output_dirpath / f'{video_name}/{video_name}.mp4'
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -r {frame_rate} -pattern_type glob -i "{frames_dirpath.as_posix()}/*pred_bedlam.jpg" -c:v libx264 -pix_fmt yuv420p {output_filepath.as_posix()}'
    execute_shell_command(cmd)
    return


def main():
    args = parse_args()

    videos_dirpath = Path(args.videos_dirpath)
    output_dirpath = Path(args.output_folder)
    for video_frames_dirpath in sorted(videos_dirpath.iterdir()):
        video_name = video_frames_dirpath.stem
        frames_output_dirpath = output_dirpath / f'{video_name}/frames'
        demo_args = SimpleNamespace(cfg=args.cfg,
                                    ckpt=args.ckpt,
                                    image_folder=video_frames_dirpath.as_posix(),
                                    output_folder=frames_output_dirpath.as_posix(),
                                    tracker_batch_size=args.tracker_batch_size,
                                    detector=args.detector,
                                    yolo_img_size=args.yolo_img_size,
                                    display=args.display,
                                    save_result=args.save_result,
                                    eval_dataset=None,
                                    dataframe_path=None,
                                    data_split=None
                                    )
        if args.demo_file_name == 'demo':
            from demo import main as demo_main
            demo_main(demo_args)
        elif args.demo_file_name == 'demox':
            from demox import main as demox_main
            demox_main(demo_args)
        else:
            raise RuntimeError(f'Unknown demo file: {args.demo_file_name}')
        merge_frames(video_name, output_dirpath)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/demo_bedlam_cliff_x.yaml',
                        help='config file that defines model hyperparams')

    parser.add_argument('--ckpt', type=str, default='data/ckpt/bedlam_cliff_x.ckpt',
                        help='checkpoint path')

    parser.add_argument('--videos_dirpath', type=str,
                        default='../../../../../databases/Spree01/data/rgb_png',
                        help='input video frames folder')

    parser.add_argument('--output_dirpath', type=str,
                        default='../runs/testing/test0000',
                        help='output folder to write results')

    parser.add_argument('--tracker_batch_size', type=int, default=1,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--display', action='store_true',
                        help='visualize the 3d body projection on image')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--demo_file_name', type=str, default='demox',
                        help='demo file name')

    parser.add_argument('--save_result', action='store_true',
                        help='Save verts, joints, joints2d in pkl file to evaluate')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))