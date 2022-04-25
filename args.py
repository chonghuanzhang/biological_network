import argparse

parser = argparse.ArgumentParser(description='Arguments for class "EnzymePrediction"')
parser.add_argument('-m', '--method', type=str, required=False,
                    help='Training method: "projection" or "imbalance", default: projection'
                    )
parser.add_argument('-fp', '--fingerprint', type=str, required=False,
                    help='Fingerprint method: "ECFP" or "BERT", default: ECFP'
                    )
parser.add_argument('-r', '--radius', type=int, required=False,
                    help='ECFP fingerprint radius, default: 3, only required in ECFP'
                    )
parser.add_argument('-bt', '--byte', type=int, required=False,
                    help='number of byte features in mol fingerprint, default: 128'
                    )
parser.add_argument('-e', '--epoch', type=int, required=False,
                    help='Number of epoch, default: 100'
                    )
parser.add_argument('-b', '--batch', type=int, required=False,
                    help='batch size, default: 128'
                    )
parser.add_argument('-lr', '--learning_rate', type=float, required=False,
                    help='learning rate, default: 2e-3'
                    )
parser.add_argument('-af', '--activation_function', type=str, required=False,
                    help='learning rate, default: elu'
                    )

args = parser.parse_args()