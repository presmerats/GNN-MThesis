import argparse
import yaml
from pprint import pprint


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder',type=str)

args= parser.parse_args()

print("training",args.folder)


jobdict = yaml.load(open(args.folder,'r'), Loader=yaml.FullLoader)
pprint(jobdict)

yaml.dump(jobdict,open(args.folder.replace('train','test'),'w'))
