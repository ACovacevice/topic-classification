import nlpnet
import os
import pathlib
import requests
import shutil

local_dir = os.path.dirname(__file__)

pos_dir = os.path.join(local_dir, "pos")
pos_pt = os.path.join(pos_dir, "pos-pt")

# Download pos-pt ----------------------

if not os.path.exists(pos_pt):

    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)

    print("Downloading pos-pt...")
    url = "http://nilc.icmc.usp.br/nlpnet/data/pos-pt.tgz"
    response = requests.get(url)

    with open(f"{pos_pt}.tgz",'wb') as output_file:
        output_file.write(response.content)

    shutil.unpack_archive(f"{pos_pt}.tgz", os.path.realpath(pos_dir))

    if os.path.isfile(f"{pos_pt}.tgz"):
        pathlib.Path(f"{pos_pt}.tgz").unlink()

# --------------------------------------    

nlpnet.set_data_dir(pos_pt)
pos_tagger = nlpnet.POSTagger()

from ._processing import clean_text, get_keywords_from_text