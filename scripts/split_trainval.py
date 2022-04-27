"""
@Fire
https://github.com/fire717
"""
import argparse
import json
import pathlib
import random


def main(args):

    input_path = pathlib.Path(args.input_path)
    input_path = pathlib.Path(input_path.resolve())

    with open(str(input_path), 'r') as f:
        data = json.loads(f.readlines()[0])
    print("total: ", len(data))
    print(data[0])

    random.shuffle(data)
    print(data[0])

    val_count = 20  # (percentage for validation)
    ratio = int((val_count / 100) * len(data))

    print("val_nums", val_count)
    print("ratio", ratio)

    data_val = data[:ratio]
    data_train = data[ratio:]
    # for d in data:
    #     if random.random()>ratio:
    #         data_train.append(d)
    #     else:
    #         data_val.append(d)

    print(len(data_train), len(data_val))

    # set training file path
    if args.output_path_train is None:
        output_path_train = input_path.parent / 'train.json'
    else:
        output_path_train = pathlib.Path(args.output_path_train)
        output_path_train = pathlib.Path(output_path_train.resolve())

    # create training json file
    with open(str(output_path_train), 'w') as f:
        json.dump(data_train, f, ensure_ascii=False)

    # set validation file path
    if args.output_path_val is None:
        output_path_val = input_path.parent / 'val.json'
    else:
        output_path_val = pathlib.Path(args.output_path_val)
        output_path_val = pathlib.Path(output_path_val.resolve())

    # create validation json file
    with open(str(output_path_val), 'w') as f:
        json.dump(data_val, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help='Path to the annotations json file', required=True, default='/home/ggoyal/data/mpii/anno/poses.json')
    parser.add_argument('-ot', '--output_path_train', help='Path to the output training annotations json file')
    parser.add_argument('-ov', '--output_path_val', help='Path to the output training annotations json file')
    args = parser.parse_args()

    main(args)
