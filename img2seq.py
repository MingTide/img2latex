from decoder import Decoder
from encoder import Encoder


class Img2SeqModel():
    def __init__(self, config, vocab):
        self._vocab = vocab
        self._config = config
    def build_train(self, config):
        """Builds model"""
        self.logger.info("Building model...")

        self.encoder = Encoder(self._config)
        self.decoder = Decoder(self._config, self._vocab.n_tok,
                self._vocab.id_end)

    def _add_pred_op(self,img,formula):
        """Defines self.pred"""
        encoded_img = self.encoder(img)
        train, test = self.decoder(encoded_img, formula,
                                   self.dropout)

        self.pred_train = train
        self.pred_test = test
