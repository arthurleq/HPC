# HPC Repository

This repository aims to give an overview of the hostile world of high-performance computing (HPC): why we need parallel computers, how they are built, and how to program them.

It consists of:

- a synthesis of the **principles of HPC** ([`HPC_fr.pdf`](HPC_fr.pdf));
- a [`Codes/`](Codes/) folder containing commented examples of how to use the main parallelisation paradigms: **OpenMP**, **MPI**, **OpenACC** and **CUDA**;
- related considerations, such as making the field more accessible to the general public, the economy surrounding this sector, its future, and its link to AI ([`Economy.pdf`](Economy.pdf)).

Everything is written to be accessible, with a light tone: the documents start from the basics, and the README of each code folder explains every concept step by step.


## The documents

**[`HPC_fr.pdf`](HPC_fr.pdf) — the principles of HPC**

- why computing became parallel: Moore's law, the end of Dennard scaling, Amdahl's and Gustafson's laws;
- Flynn's taxonomy (SISD, SIMD, MISD, MIMD) and the two ways to decompose a problem (data or tasks);
- memory architectures (shared, NUMA, distributed, hybrid) and the programming models that go with them (OpenMP, MPI, MPI + OpenMP);
- GPUs: what they are, how to compare them (frequency, cores, TFLOPS, memory, price, energy), and CUDA.

**[`Economy.pdf`](Economy.pdf) — computing as an economic and strategic resource**

- where the demand comes from: AI, whose training data, model sizes and training compute grow much faster than the supercomputers of the TOP500;
- orders of magnitude: price and power consumption of a data-center GPU, total cost of ownership, renting;
- the value chain of a GPU, from raw materials (rare earths, ultra-pure silicon) to lithography (ASML, TSMC) and chip binning;
- software as an economic lock-in: CUDA and its alternatives.

## The codes
TODO later