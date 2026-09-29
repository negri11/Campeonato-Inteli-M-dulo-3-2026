# Campeonato Inteli Módulo 3 2026

Solução para a competição do Kaggle que pede as probabilidades de vitória do visitante (A), empate (D) e vitória do mandante (H) para partidas do Brasileirão Série A, avaliadas por LogLoss multiclasse. O treino cobre 2013 a 2021 e o teste cobre 2022 a 2026.

## Abordagem

O modelo é uma regressão logística multinomial treinada com rótulos suaves. Em vez de aprender direto do resultado da partida, que é uma amostra muito ruidosa, o modelo aprende a reproduzir a probabilidade implícita das odds de fechamento do treino, que é uma estimativa bem mais estável da mesma quantidade. Como o teste não tem odds, a previsão depende só das colunas da própria partida.

As decisões que sustentam o score são as seguintes:

- **Alvo de mercado sem margem.** A margem das odds é removida pelo método da potência, que corrige o viés de a casa carregar mais margem nos azarões.
- **Features calculadas por linha.** Toda feature sai de operações entre colunas da mesma partida, sem `groupby`, `shift` ou qualquer operação que olhe outras linhas do teste.
- **Efeito fixo de time.** Os nomes do mandante e do visitante entram com regularização, e os coeficientes são aprendidos só no treino.
- **Validação temporal.** A seleção usa três cortes por temporada, sem shuffle, incluindo dois cortes de horizonte longo que imitam a distância entre treino e teste.
- **Ensemble das melhores configurações.** A previsão final é a média das melhores configurações da validação, para não depender de uma escolha única feita sobre ruído.

## Como rodar

Os dados não ficam no repositório. Baixe `train.csv`, `test.csv` e `sample_submission.csv` na aba Data da competição e coloque na pasta `data/`.

```bash
pip install -r requirements.txt
```

```bash
jupyter nbconvert --to notebook --execute brasileirao_probabilidades.ipynb --inplace
```

A execução leva poucos minutos e gera o `submission.csv` na raiz, com as colunas na ordem `Id, A, D, H`. O resultado é determinístico, então duas execuções produzem o mesmo arquivo.

## Regras da competição que o código respeita

- A modelagem usa apenas NumPy, Pandas e Scikit-Learn.
- Nenhum dado externo entra no pipeline.
- Cada previsão do teste usa somente a própria linha do `test.csv` e informação do `train.csv`. O notebook verifica isso prevendo partidas uma a uma e comparando com a previsão em lote.
- O placar público não é usado para ajustar o modelo.

## Aviso

Projeto acadêmico, sem finalidade comercial. As probabilidades geradas não são recomendação de aposta.
