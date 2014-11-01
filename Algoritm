# coding=utf-8
#K-Means pour Movies

from __future__ import division

import codecs
import math
import time
import re
import sys,os,random
import pprint
import operator

from collections import defaultdict
from operator import itemgetter
from itertools import izip
from math import *


##Calculo de la media
def mean(l):
    if (len(l) == 0):
	return 0.0
    else: 
	return sum(l)/len(l)
	
	
#Calculo de la varianza
def variance(l):
    if (len(l) == 0):
	return 0.0
    else:
	return sum([ (e - mean(l)) ** 2 for e in l ])/len(l)

def load_data(filepath):
    users = defaultdict(dict)
    movies = defaultdict(dict)    
    with codecs.open(filepath, encoding="utf-8") as file_:
        for line in file_:
            aut, mov, rating = line.strip().split("|")
            aut, rating = int(aut), int(rating)           
            users[aut][mov] = rating
            movies[mov][aut] = rating
    return users, movies


def cov(v1, v2):
	assert len(v1) == len(v2) and len(v1) > 0
    if len(v1) == 1:
        return 0.0
    else:
        mean1 = mean(v1)
        mean2 = mean(v2)
        return sum((e1 - mean1) * (e2 - mean2)
                   for e1, e2 in izip(v1, v2)) / (len(v1) - 1)
	  

def corr(v1, v2):
    cov1 = cov(v1, v1)
    cov2 = cov(v2, v2)
	o
    if cov1 == 0.0 or cov2 == 0.0:
        return 0.0
    else:
	    res = 1-((cov(v1, v2) / sqrt(cov1 * cov2) + 1)/2)
        return res

		
		
def similarity(mov1, mov2):
    mov1_users = movies[mov1]
    mov2_users = movies[mov2]
    common_users = set(mov1_users.keys())
    
    common_users.intersection(mov2_users.keys())
   
    ratings_vect1 = [mov1_users.get(u, 0) for u in common_users]
    ratings_vect2 = [mov2_users.get(u, 0) for u in common_users]
   
    return corr(ratings_vect1, ratings_vect2)
	
def compute_similarities(mov, movies):
    mov1 = mov
    res = {}
    for mov2 in movies:
        res[mov2] = similarity(mov1, mov2)
    return res


def Initialisation(movies):
    k=5
    centroides = defaultdict(dict)
    for index in range(0,k):
        number = random.randrange(len(movies))
        centroides[index] = (movies.keys())[number]
        print(centroides[index])
    return centroides

def Affectation(movies, centroides):
    clusters=defaultdict(dict)
    distance=-6
    for movie in movies:
        for c in centroides:
            similar=similarity(centroides[c],movie)
            if(similar>distance):
                distance=similar
            clusters[movie]=c
    return clusters

def AffectationSuivant(movies, centroides, clusters):
    modif = 0
    for movie in clusters:
        distance = similarity(centroides[clusters[movie]], movie)
        for c in centroides:
            similar=similarity(centroides[c],movie)
            if(similar>distance):
                distance=similar
                cluster[movie]=c
                modif = 1
    if modif == 1:
        recalc(movies, centroides, clusters)
    return cluster
    
def recalc(movies, centroides, clusters):
    for index in centroides:
        peliculas_cluster=defaultdict(dict)
        for index2 in clusters:
            if clusters[index2] == index:
                peliculas_cluster[index2] = index
        maxmoyen = -5000
        for index2 in peliculas_cluster:
            vector = compute_similarities(index2, peliculas_cluster.keys())
            moyen = sum(vector.values())
            if moyen> maxmoyen:
                maxmoyen=moyen
                centroides[index] = index2
    AffectationSuivant(movies, centroides, clusters)

	
if __name__ == "__main__":

    import k-means
	
    users, movies = projet.load_data("movie_lens.csv")
    TousLesCentroides = Initialisation(movies)
    Cluster=Affectation(movies,TousLesCentroides)
    recalc(movies, TousLesCentroides, Cluster)

