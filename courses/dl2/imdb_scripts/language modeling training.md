- Train and test CSV files created by imdb notebook

- Preparação de train.csv e test.csv dentro de cada pasta, de acordo com o notebook. Os sets de cada um são diferentes considerando que o clas só considera positivo e negativo para treinamento, e o lm considera tudo, pois vai criar um modelo de linguagem.

source activate fastai

python create_toks.py --dir-path /media/discoD/fastai/data/imdb_clas --lang en

python tok2id.py --dir-path /media/discoD/fastai/data/imdb_clas --max-vocab 60000 --min-freq 1

python create_bwd_data.py --dir-path /media/discoD/fastai/data/imdb_clas

python create_toks.py --dir-path /media/discoD/fastai/data/imdb_lm --lang en

python tok2id.py --dir-path /media/discoD/fastai/data/imdb_lm --max-vocab 60000 --min-freq 1

python create_bwd_data.py --dir-path /media/discoD/fastai/data/imdb_lm

python create_toks.py --dir-path /media/discoD/fastai/data/wiki/en --lang en

python tok2id.py --dir-path /media/discoD/fastai/data/wiki/en --max-vocab 60000 --min-freq 1

python create_bwd_data.py --dir-path /media/discoD/fastai/data/wiki/en

python pretrain_lm.py --dir-path /media/discoD/fastai/data/wiki/en --cuda-id 0 --lr 1e-3 --cl 1 --pretrain-id test-pretrain --backwards

python finetune_lm.py --dir-path /media/discoD/fastai/data/imdb_lm --pretrain-path /media/discoD/fastai/data/wiki/en --cuda-id 0 --pretrain-id test-pretrain --lm-id test-lm --lr 4e-3 --cl 1

python train_clas.py --dir-path /media/discoD/fastai/data/imdb_clas --lm-dir /media/discoD/fastai/data/imdb_lm --cuda-id 0 --lm-id test-lm --clas-id test-class --bs 30 --cl 1


- A idéia é criar um modelo pré-treinado com uma quantidade maior de texto, a partir dos tokens obtidos pelo pré-processamento do wikipedia.
- Posteriormente, o fine-tuning dá uma otimizada no modelo usando um corpus mais específico daquilo que se deseja treinar no final. Como é um modelo de classificação do imdb, então é feito um fine-tuning usando o corpus do imdb, a partir dos pesos do modelo pré-treinado com o wikipedia.
- Uma vez feito o fine-tuning, então treina-se o modelo final.

- Para cada corpus é necessário fazer a tokenização e a criação do mapeamento de tokens para ids.


1. Definição do caminho para o corpus do IMDB onde estão as pastas de treino e teste 
2. Definição de um caminho para o modelo de classificação final
3. Definição de um caminho para o modelo de modelagem de linguagem. Neste modelo, os labels são irrelevantes.
4. Criação de variáveis de texto e labels para treino e teste
	4.1. Split dos dados (já faz a randomização)
	4.2. Criação de um DataFrame com os textos e labels
	4.3. Salvar o DataFrame em CSV
5. 
	
	
source activate fastai

python create_toks.py --dir-path /media/discoD/fastai/data/cstnews_clas --lang pt

python tok2id.py --dir-path /media/discoD/fastai/data/cstnews_clas --max-vocab 60000 --min-freq 1

- python create_bwd_data.py --dir-path /media/discoD/fastai/data/cstnews_clas

python create_toks.py --dir-path /media/discoD/fastai/data/cstnews_lm --lang pt

python tok2id.py --dir-path /media/discoD/fastai/data/cstnews_lm --max-vocab 60000 --min-freq 1

- python create_bwd_data.py --dir-path /media/discoD/fastai/data/cstnews_lm

python create_toks.py --dir-path /media/discoD/fastai/data/wiki/pt --lang pt

python tok2id.py --dir-path /media/discoD/fastai/data/wiki/pt --max-vocab 60000 --min-freq 1

- python create_bwd_data.py --dir-path /media/discoD/fastai/data/wiki/pt

python pretrain_lm.py --dir-path /media/discoD/fastai/data/wiki/pt --cuda-id 0 --lr 1e-3 --cl 50 --pretrain-id pretrain-pt

python finetune_lm.py --dir-path /media/discoD/fastai/data/cstnews_lm --pretrain-path /media/discoD/fastai/data/wiki/pt --cuda-id 0 --pretrain-id pretrain-pt --lm-id lm-cstnews --lr 4e-3 --cl 50 --bs 1

python train_clas.py --dir-path /media/discoD/fastai/data/cstnews_clas --lm-dir /media/discoD/fastai/data/cstnews_lm --cuda-id 0 --lm-id lm-cstnews --clas-id class-cstnews --bs 4 --cl 50
(bs = 4 for 90-10 split and bs = 6 for 70-30 split)