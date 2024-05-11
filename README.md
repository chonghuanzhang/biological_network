# Exploration of bioinformatic domain based on data mining, reaction and enzyme promiscuity predictions

Preprint((https://chemrxiv.org/engage/chemrxiv/article-details/64aafc74ba3e99daefc8e433),

Repository(https://github.com/chonghuanzhang/biological_network)

## Introduction
Biochemical transformations may allow significant improvements in synthetic efficiency of complex functional molecules through reduction in the number of synthetic steps or avoidance of harsh conditions and/or toxic solvents/reactants. Yet, there is a limited access to biochemical reaction data, which reduces the opportunities of finding alternatives and discovering synergies with organic synthesis. We propose a workflow to explore the sparse synthetic biological domain. Using a molecular graph method we predict feasible biosynthetic reactions. The products of biosyntheses are learned from the functional transformations of the literature-excerpted reactions recorded in KEGG database. Through this approach we expanded the KEGG reaction dataset of biochemical transformations by approximately four times. To catalyse the novel reactions, we proposed a transformer model that learns from reaction SMILES and amino acid sequences of native enzymes and predicts promiscuous enzymes for potential substrates. The proposed transformer model calibrates the feasibility of the predicted reactions and reduces the search scope for promiscuous enzymes in the pool. A populated biological reaction space is eventually visualised in a two-dimensional t-SNE diagram.


## Usage
Load the target molecule and the KEGG reaction database into the file path, and run mol_main.py. 

## License

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/chonghuanzhang/balancing_rxn/blob/main/LICENSE) for additional details.


