import click
import torch

from data_generator import DataGenerator
from decoder import Decoder
from general import Config, minibatches
from img2seq import Img2SeqModel
from model.utils.image import greyscale, pad_batch_images
from model.utils.text import Vocab, pad_batch_formulas
from encoder import Encoder


@click.command()
@click.option('--data', default="configs/data_small.json",
        help='Path to data json config')
@click.option('--vocab', default="configs/vocab_small.json",
        help='Path to vocab json config')
@click.option('--training', default="configs/training_small.json",
        help='Path to training json config')
@click.option('--model', default="configs/model.json",
        help='Path to model json config')
@click.option('--output', default="results/small/",
        help='Dir for results and model weights')
def main(data, vocab, training, model, output):

    config = Config([data, vocab, training, model])
    vocab = Vocab(config)
    train_set = DataGenerator(path_formulas=config.path_formulas_train,
            dir_images=config.dir_images_train, img_prepro=greyscale,
            max_iter=config.max_iter, bucket=config.bucket_train,
            path_matching=config.path_matching_train,
            max_len=config.max_length_formula,
            form_prepro=vocab.form_prepro)
    batch_size = config.batch_size
    res  = []
    for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
        my_dict = {
            'img': pad_batch_images(img),
            'formula':pad_batch_formulas(formula,
                    vocab.id_pad, vocab.id_end)
        }
        res.append(my_dict)

    encoder = Encoder(config)
    encoder_img = encoder(torch.tensor(res[0]['img']))
    decoder = Decoder(config, vocab.n_tok, vocab.id_end)
    decoder(encoder_img,torch.tensor(res[0]['formula'][0]))

if __name__ == "__main__":
    main()