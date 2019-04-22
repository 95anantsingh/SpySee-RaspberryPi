import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--input', default='my_input.txt')
    parser.add_argument('--inpt', default='my_input.txt')
    return parser

def main(args):
    print(args.input)
    print(args.inpt)

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    main(args)
