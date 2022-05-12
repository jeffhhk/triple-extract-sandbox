
import sys
import json
import os
from xml.sax.saxutils import escape

import tqdm
from blue.ext.preprocessing import tokenize_text, print_ner_debug, write_bert_ner_file
import bioc

def mkdirp_for(rfile):
    os.makedirs(os.path.dirname(rfile), exist_ok=True)

class Chemprot(object):
    def __init__(self) -> None:
        self._abstracts={}
        self._entities_by_abstract={}
        self._rels=[]

    def add_abstract(self,abstract):
        docid=abstract["id"]
        self._abstracts[docid]=abstract

    def number_abstracts(self):
        ich = 0
        for k in sorted(self._abstracts.keys()):
            abstract=self._abstracts[k]
            abstract["ich"]=ich
            txt=abstract["txt"]
            ich += len(txt)+1   # 1 for size of EOL

    def add_entity(self,entity):
        docid=entity["docid"]
        v=[]
        if docid in self._entities_by_abstract:
            v=self._entities_by_abstract[docid]
        v.append(entity)
        self._entities_by_abstract[docid]=v
    
    def add_rel(self,rel):
        self._rels.append(rel)

    def load(self, rfile_abstracts, rfile_entities, rfile_rel):
        with open(rfile_abstracts) as f:
            for X0 in f:
                X1=X0.rstrip()
                X = X1.split("\t")
                docid = int(X[0])
                title = X[1]
                txt = X[2]
                if docid in self._abstracts:
                    raise "Found duplicate abstract id={}".format(docid)
                self.add_abstract({
                    "id":docid,
                    "title":title,
                    "txt":txt
                })
        self.number_abstracts()
        with open(rfile_entities) as f:
            for X in f:
                X=X.rstrip()
                X = X.split("\t")
                docid = int(X[0])
                entid = X[1]
                ent_name = X[2]
                ich_start = int(X[3])
                ich_stop = int(X[4])
                self.add_entity({
                    "docid":docid,
                    "entid":entid,
                    "ent_name":ent_name,
                    "ich_start":ich_start,
                    "ich_stop":ich_stop
                })
        with open(rfile_rel) as f:
            for X in f:
                X=X.rstrip()
                X = X.split("\t")
                docid = int(X[0])
                relid = X[1]
                relfoo = X[2]
                rel_name = X[3]
                entid_1 = X[4][5:]
                entid_2 = X[5][5:]
                self.add_rel({
                    "docid":int(docid),
                    "relid":relid,
                    "relfoo":relfoo,
                    "rel_name":rel_name,
                    "entid_1":entid_1,
                    "entid_2":entid_2
                })

    # Export data for brat labeling tool
    def export1(self):
        fnOutTxt="ChemProt_Corpus/chemprot_brat.txt"
        fnOutAnn="ChemProt_Corpus/chemprot_brat.ann"
        mkdirp_for(fnOutTxt)
        with open(fnOutTxt,"w") as f:
            for docid in sorted(self._abstracts.keys()):
                abstract=self._abstracts[docid]
                f.write(abstract["title"])
                f.write("\n")
                f.write(abstract["txt"])
                f.write("\n")
        mkdirp_for(fnOutAnn)
        with open(fnOutAnn,"w") as f:
            for docid in sorted(self._abstracts.keys()):
                for entity in sorted(self._entities_by_abstract[docid], key=lambda e:e["ich_start"]):
                    abstract=self._abstracts[docid]
                    ich=abstract["ich"]
                    ich_start=ich+entity["ich_start"]
                    ich_stop= ich+entity["ich_stop"]
                    personid_bogus="NTL"
                    e=[
                        entity["entid"],
                        entity["ent_name"],
                        str(ich_start),
                        str(ich_stop),
                        personid_bogus
                    ]
                    f.write("\t".join(e))
                    f.write("\n")
            # TODO: emit relations

    # Export xml-shaped data for 3 stage triple extraction
    def export2(self,fnOut1):
        mkdirp_for(fnOut1)
        with open(fnOut1,"w") as f:
            f.write("<xml>\n")
            for docid in sorted(self._abstracts.keys()):
                abstract=self._abstracts[docid]
                txt=abstract["title"]+"\t"+abstract["txt"]
                entities=sorted(
                    (self._entities_by_abstract[docid] if docid in self._entities_by_abstract else []),
                    key=lambda entity: entity["ich_start"])
                ichL=0
                for entity in entities:
                    ichR=entity["ich_start"]
                    ichRR=entity["ich_stop"]
                    n=entity["ent_name"]
                    f.write(escape(txt[ichL:ichR]))
                    f.write("<{}>".format(escape(n)))
                    f.write(escape(txt[ichR:ichRR]))
                    f.write("</{}>".format(escape(n)))
                    ichL=ichRR
                f.write(escape(txt[ichL:]))
                f.write("\n")
            f.write("</xml>\n")

    # From BLUE_Benchmark/blue/bert/create_cdr_bert.py
    def _find_toks(self, sentences, start, end):
        toks = []
        for sentence in sentences:
            for ann in sentence.annotations:
                span = ann.total_span
                if start <= span.offset and span.offset + span.length <= end:
                    toks.append(ann)
                elif span.offset <= start and end <= span.offset + span.length:
                    toks.append(ann)
        return toks

    # Export data formatted for run_bluebert_ner.py task_name=bc5cdr (BC5CDRProcessor)
    # Ported from BLUE_Benchmark/blue/bert/create_cdr_bert.py
    def export3(self,fnOut1):
        validate_mentions=None
        mkdirp_for(fnOut1)
        total_sentences = []

        for docid in tqdm.tqdm(sorted(self._abstracts.keys())):
            abstract=self._abstracts[docid]
            txt=abstract["title"]+"\t"+abstract["txt"]
            entities=sorted(
                (self._entities_by_abstract[docid] if docid in self._entities_by_abstract else []),
                key=lambda entity: entity["ich_start"])

            sents = tokenize_text(txt, docid)

            ichL=0
            for entity in entities:
                #if ann.type == entity_type:    # entity["ent_name"]
                anns = self._find_toks(sents, entity["ich_start"], entity["ich_stop"])
                if len(anns) == 0:
                    print(f'Cannot find {doc.pmid}: {ann}')
                    print_ner_debug(sents, ann.start, ann.end)
                    exit(1)
                has_first = False
                for ann in anns:
                    if not has_first:
                        ann.infons['NE_label'] = 'B'
                        has_first = True
                    else:
                        ann.infons['NE_label'] = 'I'

            total_sentences.extend(sents)

        cnt = write_bert_ner_file(fnOut1, total_sentences)
        if validate_mentions is not None and validate_mentions != cnt:
            print(f'Should have {validate_mentions}, but have {cnt} {entity_type} mentions')
        else:
            print(f'Have {cnt} mentions')


        

def main():
    rfile_abstracts=sys.argv[1]
    rfile_entities=sys.argv[2]
    rfile_rel=sys.argv[3]
    rfile_out=sys.argv[4]
    cp=Chemprot()
    cp.load(rfile_abstracts, rfile_entities, rfile_rel)
    #cp.export1()
    #cp.export2(rfile_out)
    cp.export3(rfile_out)


if __name__ == '__main__':
    main()
