#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler

sampler = SimulatedAnnealingSampler()

x0, x1, y = ('x0', 'x1', 'y')
a, b, c = 1, 1, 3
Q = {(y, y): c, (x0, y): -2*b, (x1, y): -2*b, (x0, x1): a, (x0, x0): 0}

bqm = BinaryQuadraticModel.from_qubo(Q)
response = sampler.sample(bqm, num_reads=5000)

results = {}
for datum in response.data(['sample', 'energy', 'num_occurrences']):
    key = (tuple(dict(datum.sample).items()), float(datum.energy))
    if key in results:
        results[key] = (datum.sample, results[key][1] + datum.num_occurrences)
    else:
        results[key] = (datum.sample, datum.num_occurrences)

num_runs = sum([results[key][1] for key in results])
for key in results:
    try:
        print(results[key][0], "Energy: ", key[1],
              f"Occurrences: {results[key][1]/num_runs*100:.2f}%")
    except KeyError:
        pass
