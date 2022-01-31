import argparse
import os
import boto3
import botocore
from dotenv import load_dotenv

load_dotenv()


def downloadDirectoryFroms3(bucket_name, remote_directory_name, local_dir):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=remote_directory_name):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--impath", default="None",
                        help="provide image path for transcirption, if none provided\nfolder in gin file will be used"
                             "for evaluation")
    args = parser.parse_args()
    impath = args.impath
    if not os.path.exists('bb_model/'):
        print('downloading model...')
        BUCKET_NAME = os.getenv('BUCKET_NAME')
        PATH = os.getenv('PATH_LOC_2')
        DIR_LOC = os.getenv('DIR_LOC_2')

        downloadDirectoryFroms3(BUCKET_NAME, PATH, DIR_LOC)
    else:
        print('model already in place...')

    