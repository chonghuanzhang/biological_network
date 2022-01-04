import os

# Directory
DB_PATH = '/home/cz332/database/resource/'
MOL_FILE_PATH = '/home/projects/chemical_structures/data/mol_files'

# Molecule
REAXYS_MOL_PATH = os.path.join(DB_PATH, 'reaxys_database/reaxys_molecule_pd.pkl')
KEGG_MOL_PATH = os.path.join(DB_PATH, 'kegg_database/kegg_molecule_pd.pkl')
DRUG_MOL_PATH = os.path.join(DB_PATH, 'kegg_database/drug_molecule_pd.pkl')
MOL_FPS_PATH = os.path.join(DB_PATH, 'results/mol_fps.pkl')

# Curated metabolite library
METABOLITE_LIB_PATH = os.path.join(DB_PATH, 'kegg_database/metabolite_library.xlsx')

# Reaction
REAXYS_RXN_DIR = os.path.join(DB_PATH, 'reaxys_database/reaxys_smiles_reaction')
KEGG_RXN_PATH = os.path.join(DB_PATH, 'kegg_database/kegg_reaction_pd.pkl')

# StellarGraph Network
REAXYS_NET_PATH = os.path.join(DB_PATH, 'network/reaxys_stellargraph.pkl')
REAXYS_X_NET_PATH = os.path.join(DB_PATH, 'network/reaxys_stellargraph_exclusive.pkl')
KEGG_NET_PATH = os.path.join(DB_PATH, 'network/kegg_stellargraph.pkl')
HYBRID_NET_PATH = os.path.join(DB_PATH, 'network/hybrid_stellargraph.pkl')