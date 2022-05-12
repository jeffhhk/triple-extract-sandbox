"""
    Download pmcaws formatted XML.

    pmcaws: https://www.ncbi.nlm.nih.gov/pmc/tools/pmcaws/
"""
import os
import subprocess
import csample
from tqdm import tqdm

from joblib import Memory
adirProject=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
mem = Memory(os.path.join(adirProject, "derived_data", "joblib"))

urlPmcaws="https://s3.us-east-1.amazonaws.com/pmc-oa-opendata"

def download_stream_pubmed_oa():
    proc = subprocess.Popen(
        ["curl", "{}/oa_noncomm/xml/metadata/txt/oa_noncomm.filelist.txt".format(urlPmcaws)],
        stdout=subprocess.PIPE,
        encoding='utf-8')

    first=True
    for line in proc.stdout:
        if not first:
            F=line.split("\t")
            yield F
        first=False

def download_stream_pubmed_oa_sample(seed, rate):
    sampler=csample.HashSampler(seed=seed)
    return [
        x
        for x in download_stream_pubmed_oa()
        if sampler.should_sample(x[0],rate)]

pubmed_oa_subset = mem.cache(download_stream_pubmed_oa_sample)

def download_items():
    for x in tqdm(pubmed_oa_subset(seed="123", rate=0.001)):
        path=x[0]
        url="{}/{}".format(urlPmcaws, path)
        afileOut=os.path.join(adirProject, "derived_data", "pubmed",os.path.basename(path))
        adirOut=os.path.dirname(afileOut)
        if not os.path.exists(adirOut):
            os.makedirs(adirOut)
        if not os.path.exists(afileOut):
            subprocess.check_call(
                ["curl", "--silent", url, "-o", afileOut]
            )

if __name__ == '__main__':
    download_items()
