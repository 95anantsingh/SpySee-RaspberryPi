from TestFile1 import main, parse_args
parser = parse_args()
args = parser.parse_args(['--input','abc']) # note the square bracket
#parser.parse_args(['--input','abc'])
#print(args) # Namespace(input='my_input.txt')
main(args)