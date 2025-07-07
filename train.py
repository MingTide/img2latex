import click

from data_generator import DataGenerator
from general import Config, minibatches
from model.utils.image import greyscale
from text import Vocab


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
    for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
        print(formula)
        print(img)
if __name__ == "__main__":
    main()