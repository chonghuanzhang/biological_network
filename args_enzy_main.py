"""
command example:
python args_enzy_main.py -m projection -fp bert -e 500 -b 128 -lr 2e-3 -af 'elu' -op 'adam'
python args_enzy_main.py -m imbalance -fp bert -e 100 -lr 1e-3 -af 'sigmoid' -op 'adam' -loss 'mean_squared_error'
"""
import argparse
from enzy_pred import EnzymePrediction

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for class "EnzymePrediction"')
    parser.add_argument('-m', '--model_method', type=str, required=False, default='projection',
                        help='Training method: "projection" or "imbalance"'
                        )
    parser.add_argument('-fp', '--fp_type', type=str, required=False, default='ecfp',
                        help='Fingerprint method: "ecfp" or "bert"'
                        )
    parser.add_argument('-r', '--radius', type=int, required=False, default=3,
                        help='ecfp fingerprint radius, only required in ecfp'
                        )
    parser.add_argument('-bt', '--nBits', type=int, required=False, default=128,
                        help='number of byte features in mol fingerprint'
                        )
    parser.add_argument('-e', '--epoch', type=int, required=False, default=200,
                        help='Number of epoch'
                        )
    parser.add_argument('-b', '--batch', type=int, required=False, default=128,
                        help='batch size'
                        )
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=2e-3,
                        help='learning rate'
                        )
    parser.add_argument('-af', '--activation_function', type=str, required=False, default='elu',
                        help='learning rate'
                        )
    parser.add_argument('-op', '--optimizer', type=str, required=False, default='adam',
                        help='optimzer'
                        )
    parser.add_argument('-loss', '--loss_function', type=str, required=False, default='binary_crossentropy',
                        help='loss function')

    args = parser.parse_args()
    enz = EnzymePrediction(fp_type=args.fp_type,
                            model_method=args.model_method,
                            radius=args.radius,
                            nBits=args.nBits,
                            epoch=args.epoch,
                            batch=args.batch,
                            lr=args.learning_rate,
                            af=args.activation_function,
                            op=args.optimizer,
                            loss=args.loss_function,
                            )
    enz.main()
    enz.execuate_train()