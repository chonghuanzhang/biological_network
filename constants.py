import os

# Directory
DB_PATH = '/home/cz332/database/resource/'
MOL_FILE_PATH = '/home/projects/chemical_structures/data/mol_files'

# Molecule
REAXYS_MOL_PATH = os.path.join(DB_PATH, 'reaxys_database/reaxys_molecule_pd.pkl')
KEGG_MOL_PATH = os.path.join(DB_PATH, 'kegg_database/kegg_molecule_pd.pkl')
DRUG_MOL_PATH = os.path.join(DB_PATH, 'kegg_database/drug_molecule_pd.pkl')
MOL_FPS_PATH = os.path.join(DB_PATH, 'results/mol_fps.pkl')

# Enzyme
ENZYME_PATH = os.path.join(DB_PATH, 'kegg_database/enzyme_pd.pkl')

# Pre-calculated MCS results for molecule pairs
MCS_PAIRS_PATH = os.path.join(DB_PATH, 'kegg_database/mcs_pairs.pkl')

# Curated metabolite library
METABOLITE_LIB_PATH = os.path.join(DB_PATH, 'kegg_database/metabolite_library.xlsx')

# Reaction
REAXYS_RXN_DIR = os.path.join(DB_PATH, 'reaxys_database/reaxys_smiles_reaction')
KEGG_RXN_PATH = os.path.join(DB_PATH, 'kegg_database/kegg_reaction_pd.pkl')
KEGG_SINGLE_RXN_PATH = os.path.join(DB_PATH, 'kegg_database/kegg_one2one_reaction_pd.pkl')

# StellarGraph Network
REAXYS_NET_PATH = os.path.join(DB_PATH, 'network/reaxys_stellargraph.pkl')
REAXYS_X_NET_PATH = os.path.join(DB_PATH, 'network/reaxys_stellargraph_exclusive.pkl')
KEGG_NET_PATH = os.path.join(DB_PATH, 'network/kegg_stellargraph.pkl')
HYBRID_NET_PATH = os.path.join(DB_PATH, 'network/hybrid_stellargraph.pkl')

# KEGG molecule similarity (from SIMCOMP API)
SIMCOMP_PATH = os.path.join(DB_PATH, 'kegg_database/SIMCOMP.pkl')

# Save results
MCS_PAIR_PATH = os.path.join(DB_PATH, 'kegg_database/mcs_pairs.pkl')
PRED_MOL_DB_PATH = os.path.join(DB_PATH, 'kegg_database/pred_mol_db.pkl')
PRED_MOL_HISTORY_PATH = os.path.join(DB_PATH, 'kegg_database/pred_mol_history.pkl')
PRED_RXN_DB_PATH = os.path.join(DB_PATH, 'kegg_database/pred_rxn_db.pkl')
PRED_RXN_HISTORY_PATH = os.path.join(DB_PATH, 'kegg_database/pred_rxn_history.pkl')

# Amino acid sequence
AASEQ_DB_PATH = os.path.join(DB_PATH, 'aaseq/aaseqCrawler')
AASEQ_TEXT_PATH = os.path.join(DB_PATH, 'aaseq/aaseq_dict.pkl')
TOKEN_SEQ_PATH = os.path.join(DB_PATH, 'aaseq/token_aaseq_dict.pkl')
EC_GENE_PAIR_PATH = os.path.join(DB_PATH, 'aaseq/ec_gene_pair.pkl')
EC_GENE_TOKEN_PATH = os.path.join(DB_PATH, 'aaseq/ec_gene_token.pkl')

# local: delete later
# KEGG_MOL_PATH = 'kegg_database/kegg_molecule_pd.pkl'
# SIMCOMP_PATH = 'kegg_database/SIMCOMP.pkl'
# KEGG_SINGLE_RXN_PATH = 'kegg_database/kegg_one2one_reaction_pd.pkl'

# PRED_MOL_DB_PATH = 'kegg_database/pred_mol_db.pkl'
# PRED_MOL_HISTORY_PATH = 'kegg_database/pred_mol_history.pkl'
# PRED_RXN_DB_PATH = 'kegg_database/pred_rxn_db.pkl'
# PRED_RXN_HISTORY_PATH = 'kegg_database/pred_rxn_history.pkl'