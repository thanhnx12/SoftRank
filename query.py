import numpy as np
import document
import math
from scipy.stats import norm
import torch
N = 10
SIGMA = 0.01

class Query:
    def __init__(self,qid, documents,score = np.zeros(N)):
        """

        :param documents: list of N document
        """
        self.score = score
        self.qid = qid
        self.documents = documents


    def G_max(self):
        res = 0
        sorted_list = sorted(self.documents,key=lambda Document: Document.label,reverse=True)
        for i,x in enumerate(sorted_list):
            res += (2**x.label) * 1/math.log2(i+2)
        return res

    def claculate_pi(self):
        """
        calculate probability that rank(i) > rank(j)
        :param s: s[i] is score of doc i
        :param sigma: standard devation
        :return: PI
        O(N^2)
        """
        with torch.no_grad():
            PI = []
            for i in range(0, N):
                p = []
                for j in range(0, N):
                    temp = norm.cdf((self.score[i] - self.score[j]) / SIGMA * math.sqrt(2), 0, 1)
                    p.append(temp)
                PI.append(p.copy())
            return np.array(PI)

    def prob_all(self, PI):
        """

        :param N: total document in an input
        :param PI: PI[i][j] is the probability that rank(i) > rank(j)
        :return: calculate all probability of document j have rank r in i document (P[j][i][r])
        O(N^3)
        i + 1 is the number of documents
        """
        P = np.zeros((N, N, N))
        for j in range(0, N):
            P[j][0][0] = 1;
        for j in range(0, N):
            for i in range(1, N):
                for r in range(0, N):
                    if r != 0:
                        P[j][i][r] = P[j][i - 1][r - 1] * PI[i][j] + P[j][i - 1][r] * (1 - PI[i][j])
                    else :
                        P[j][i][r] = P[j][i - 1][r] * (1 - PI[i][j])
        return P

    def P_derivative_by_score(self, m, j,PI,P):
        """
        # PI = self.prob_all()
        # P = self.prob_all()
        calculate dP[j] / ds[m]
        :return:
        O(N^2)
        """
        with torch.no_grad():
            res = np.zeros((N,N))
            s = self.score
            for i in range (1,N):
                dpi_ds = 0
                if m == i and m != j: dpi_ds = norm.pdf(0, s[m] - s[j], SIGMA * math.sqrt(2))
                if m != i and m == j: dpi_ds = -norm.pdf(0, s[i] - s[m], SIGMA * math.sqrt(2))
                for r in range(N):
                    if r == 0:
                        res[i][r] = res[i-1][r]*(1-PI[i][j]) + ( -P[j][i-1][r] )* dpi_ds
                    else:
                        res[i][r] = res[i-1][r-1]*PI[i][j] + res[i-1][r]*(1-PI[i][j]) + (P[j][i-1][r-1] - P[j][i-1][r]) * dpi_ds
            return res[N-1]


    def G_derivative_by_score(self,PI,P):
        """
        calculate dG/ ds
        :return:[dG/ds[0] , dG/ds[1],...,dG/ds[N-1] ]
        """
        s = self.score
        def G_derivative_by_score_m(m):
            with torch.no_grad():
                res = []
                for j in range(N):
                    temp = self.P_derivative_by_score(m,j,PI,P)
                    res.append(temp)
                gain = []
                for i in range(N):
                    temp = 2**(self.documents[i].label)
                    gain.append(temp)
                discount = []
                for i in range(N):
                    temp = math.log2(i+2)
                    discount.append(temp)
                gain = np.array(gain)
                res = np.array(res)
                discount = np.array(discount)
                return (1/self.G_max()) * np.matmul(np.matmul(gain,res),discount)
        res = []
        for i in range (N):
            temp = G_derivative_by_score_m(i)
            res.append(temp)
        return res

    def SoftNDCG(self,P):
        """

        :param P:
        :return:
        """
        result = 0
        for j in range(N):
            gj = 2**self.documents[j].label
            temp = 0
            for r in range(N):
                temp += P[j][N-1][r]/math.log2(r+2)
            #print(f"g[{j}] = {gj},  D[{j}] = {temp}")
            gj *= temp
            result += gj
        result /= self.G_max()
        return result
    def NDCG(self):
        sorted_document = [x for _, x in sorted(zip(self.score,self.documents),key= lambda x : x[0])]
        result = 0
        r = 0
        for x in sorted_document:
            result += 2**x.label/math.log2(r+2)
            r += 1
        result /= self.G_max()
        return result






