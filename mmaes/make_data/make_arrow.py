import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ava', type=str, help='Datasets.')
parser.add_argument('--root', default='./datasets', type=str, help='Root of datasets')
args = parser.parse_args()

if args.dataset.lower() == 'ava':
    from vilt.utils.write_ava import make_arrow
    make_arrow(f'{args.root}/AVA', './datasets/ava')

