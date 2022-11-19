import numpy as np
from document import Document as Doc
import pandas as pd
from query import Query
import math
import torch

def readData(file_path):
    def get_Value(a):
        b = a.split(':')
        return b[1]
    documents = []
    with open(file=file_path) as file_object:
        lines = file_object.readlines();
        for line in lines:
            line_split = line.split(' ')
            label = int(line[0])
            qid = int(get_Value(line_split[1]))
            feature = []
            for i in range(2, 138):
                feature.append(float(get_Value(line_split[i])))
            feature = torch.tensor(feature)
            doc = Doc(qid,label,feature)
            documents.append(doc)
    return documents
def getQuery(file_path,N):
    """

    :param file_path:
    :param N: number of documents in 1 query
    :return:
    """
    def isValid(N_doc):
        """
        :param N_doc: list of N documents
        :return: True if N documents have the same qid
        """
        qid = N_doc[0].qid
        for x in N_doc:
            if x.qid != qid : return False
        return True
    docs = readData(file_path)
    res_temp = []
    N_doc_temp = []
    for i,x in enumerate(docs):
        N_doc_temp.append(x)
        if len(N_doc_temp) == N:
            res_temp.append(N_doc_temp.copy())
            N_doc_temp.clear()
    res = []
    for x in res_temp:
        if isValid(x) :
            query = Query(x[0].qid,x)
            res.append(query)
    return res
q = getQuery("U:\\data.txt",10)
