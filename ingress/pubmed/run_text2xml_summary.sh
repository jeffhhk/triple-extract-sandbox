#!/bin/bash

python -c 'from ingress.pubmed.pubmedxml import *; f=Flags(); f.doPrintSummary=True; main(f)'
