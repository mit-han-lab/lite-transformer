import argparse
from collections import defaultdict

def main(args):
    significant = []
    with open(args.file, 'r') as infile:
        ops = defaultdict(list)
        for line in infile.readlines():
            line = line.strip().split()
            try:
                op, time = line[0], float(line[-1][:-len('us')])
            except:
                continue
            ops[op].append(time)
            if time > 10000:
                significant.append((op, len(ops[op]), time))
    print(significant)
    # print(len(ops[args.operator_name]), ops[args.operator_name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', '--operator-name', type=str)
    parser.add_argument('file', type=str)
    args = parser.parse_args()
    main(args)