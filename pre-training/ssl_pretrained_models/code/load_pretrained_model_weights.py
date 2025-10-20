import argparse
from helper import SimCLR, BYOL, HRCSCO

def load_ssl_pretrained_model_weights(model_name, model_path, model_trainable):
    if model_name.lower() == 'simclr':
        model = SimCLR(simclr_model_path=model_path, simclr_model_trainable=model_trainable, inp_shape=(512, 512, 3))
    elif model_name.lower() == 'byol':
        model = BYOL(byol_model_path=model_path, byol_model_trainable=model_trainable, inp_shape=(512, 512, 3))
    elif model_name.lower() == 'hrcsco':
        model = HRCSCO(hrcsco_model_path=model_path, hrcsco_model_trainable=model_trainable, inp_shape=(512, 512, 3))
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return model


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train.')
    parser.add_argument('--ssl_model_name', type=str, default='simclr', help='[simclr, byol, hrcsco]')
    parser.add_argument('--ssl_model_path', type=str, default='simclr/simclr_unet_encoder.h5')
    parser.add_argument('--ssl_model_trainable', type=bool, default=True)

    args = parser.parse_args()

    model = load_ssl_pretrained_model_weights(args.ssl_model_name, args.ssl_model_path, args.ssl_model_trainable)
    print(model.summary())
