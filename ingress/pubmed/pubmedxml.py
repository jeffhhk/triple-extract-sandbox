"""
    Parse pmcaws formatted XML, extracting abstract and body.

    pmcaws: https://www.ncbi.nlm.nih.gov/pmc/tools/pmcaws/
"""
import os
adirProject=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
import xml.etree.cElementTree as etree

class Flags():
    def __init__(self) -> None:
        self.doPrintSummary = False

def pathForPmcid(pmcid):
    return os.path.join(adirProject, "derived_data", "pubmed",os.path.basename("PMC{}.xml".format(pmcid)))

def pmcidAll():
    for p in os.listdir(os.path.join(adirProject, "derived_data", "pubmed")):
        F=p.split(".")
        pmcid=F[0][3:]
        yield pmcid

def abstractFromPmcid(pmcid):
    with open(pathForPmcid(pmcid), "r") as fd:
        context = iter(etree.iterparse(fd, events=('start', 'end')))
        event0,root = next(context)
        for event, elem in context:
            if elem.tag == 'abstract':
                # print("hello event={}".format(event))
                return list(elem.itertext())
            # else:
            #     print("event={} elem.tag={}".format(event, elem.tag))
    return None
    
def bodyFromPmcid(pmcid):
    with open(pathForPmcid(pmcid), "r") as fd:
        context = iter(etree.iterparse(fd, events=('start', 'end')))
        event0,root = next(context)
        for event, elem in context:
            if elem.tag == 'body':
                # print("hello event={}".format(event))
                return list(elem.itertext())
            # else:
            #     print("event={} elem.tag={}".format(event, elem.tag))
    return None

def doPrintSummary():
    print("found {} abstracts {} bodies {} articles".format(
        sum(1 for pmcid in pmcidAll() if abstractFromPmcid(pmcid) is not None),
        sum(1 for pmcid in pmcidAll() if bodyFromPmcid(pmcid) is not None),
        sum(1 for _ in pmcidAll())))

def main(flags):
    if flags.doPrintSummary:
        doPrintSummary()
