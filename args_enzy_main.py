import argparse
from enzy_pred import EnzymePrediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for class "EnzymePrediction"')
    parser.add_argument('-m', '--model_method', type=str, required=False,
                        help='Training method: "projection" or "imbalance", default: projection'
                        )
    parser.add_argument('-fp', '--fp_type', type=str, required=False,
                        help='Fingerprint method: "ECFP" or "BERT", default: ECFP'
                        )
    parser.add_argument('-r', '--radius', type=int, required=False,
                        help='ECFP fingerprint radius, default: 3, only required in ECFP'
                        )
    parser.add_argument('-bt', '--nBits', type=int, required=False,
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
    parser.add_argument('-op', '--optimizer', type=str, required=False,
                        help='optimzer, default: adam'
                        )
    parser.add_argument('-loss', '--loss_function', type=str, required=False,
                        help='loss function: default: binary_crossentropy')

    args = parser.parse_args()
    self = EnzymePrediction(fp_type=args.fp_type,
                            model_method=args.model_method,
                            radius=args.radius,
                            nBits=args.byte,
                            epoch=args.epoch,
                            batch=args.batch,
                            lr=args.learning_rate,
                            af=args.activation_function,
                            op=args.optimzer,
                            loss=args.loss_function,
                            )