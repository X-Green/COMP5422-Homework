import argparse
import glob
import os
import zipfile
from os import path

BLACKLIST = ['__pycache__', '.pyc']
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('homework')
    parser.add_argument('utid')
    args = parser.parse_args()

    this_path = path.dirname(path.realpath(__file__))
    files = []
    for f in glob.glob(path.join(args.homework, '**')):
        if all(b not in f for b in BLACKLIST):
            files.append(f)

    if "homework/results.pdf" not in files:
        print("Warning: no results file found!")
        raise SystemExit("Please make sure you put your results in ./homework/results.pdf as instructed!!!")

    if "homework/q2.2_1.npz"  or "homework/q2.2_2.npz"  \
    or  "homework/q2.3_3.npz" or "homework/q2.4_1.npz" or \
        "homework/q2.4_2.npz" or "homework/q3_2.npz" not in files:
        print("Warning: no results file found!")
        raise SystemExit("Please make sure you put your .npz output files under ./homework as instructed!!!")

    zf = zipfile.ZipFile(args.utid + '.zip', 'w', compression=zipfile.ZIP_DEFLATED)
    for f in files:
        zf.write(f, f.replace(os.path.join(this_path, 'homework'), args.utid))
    zf.close()
    size = os.path.getsize(args.utid + '.zip')

    if size > 10 * 1024 * 1024:
        print("Warning: The created zip file is larger than expected!",
              "Please make sure you did not accidentally include the training data and the model checkpoints in the submission directory.")
    print("submission created for %s.zip. Please remember to submit" % args.utid)
