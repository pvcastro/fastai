from fastai import *  # Quick access to most common functionality
from fastai.text import *  # Quick access to NLP functionality

import fire


def pretrain(path, language_model_id, epochs, batch_size, use_ids=True):
    path = Path(path)
    # path = Path('/media/discoD/fastai/data/wiki/pt')

    train, valid, test = 'train_200kk', 'valid_200kk', 'test_200kk'

    if use_ids:
        data_lm = TextLMDataBunch.from_id_files(path=path, train=train, valid=valid, test=test,
                                                max_vocab=60000,
                                                min_freq=2, n_labels=0, chunksize=24000, bs=batch_size)
    else:
        data_lm = TextLMDataBunch.from_csv(path=path, tokenizer=Tokenizer(lang='pt'), train=train, valid=valid,
                                           test=test, max_vocab=60000, min_freq=2, n_labels=0, chunksize=24000)

    learn = RNNLearner.language_model(data_lm, drop_mult=0.5)

    lr = float(1e-3)

    # lrs = np.array([lr/6,lr/3,lr,lr])
    # lrs, 1, wds=wd, use_clr=(32,10), cycle_len=cl

    learn.fit(epochs=epochs, lr=slice(lr / 2.6, lr), wd=1e-7)
    # learn.fit(2, slice(1e-4,1e-2))
    learn.save(language_model_id)
    # learn.load()
    # learn.load_encoder()
    # learn.load_pretrained()


'''
python legal_docs_preprocessing.py --path '/media/discoD/Corpora/Anexos Trabalhistas/200kk' --language-model-id legal_model_lm --epochs 10 --batch-size 64 --use-ids 0
'''
if __name__ == '__main__': fire.Fire(pretrain)
