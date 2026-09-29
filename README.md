# Campeonato Inteli Módulo 3 2026

Solução para a competição do Kaggle que pede as probabilidades de vitória do visitante (A), empate (D) e vitória do mandante (H) para partidas do Brasileirão Série A, avaliadas por LogLoss multiclasse. O treino cobre 2013 a 2021 e o teste cobre 2022 a 2026.

## Resultado

| Referência | LogLoss na validação local |
|---|---|
| Frequência histórica | 1.0569 |
| Modelo final | 1.0144 |
| Mercado (meta) | 0.9959 |

A validação usa três cortes temporais por temporada, sem shuffle. A distância que sobra entre o modelo e o mercado é informação que as colunas da base não trazem, como escalação e lesões.

| Submissão | Placar público |
|---|---|
| `submission.csv`, mistura de `ridge` e `logit` com regime | 1.01884 |
| `variantes/submission_logit.csv`, só `logit` com regime | 1.01838 |
| `variantes/submission_ridge.csv`, só `ridge` com regime | 1.01879 |
| `submission_sem_regime.csv`, mistura sem regime | 1.02175 |

O placar público usa 30% do teste e tem ruído maior que a diferença entre essas linhas, exceto a última. As duas submissões finais são as duas primeiras da tabela, e a razão está na decisão D10.

## Abordagem

O projeto se apoia em três decisões, e cada uma tem a evidência registrada em `docs/decisoes.md`.

- **Alvo de mercado.** O modelo aprende a reproduzir a probabilidade implícita das odds de fechamento do treino, sem a margem da casa, em vez de aprender direto do resultado. O resultado de uma partida é uma amostra muito ruidosa, e o mercado é uma estimativa bem mais estável da mesma quantidade. Como o teste não tem odds, a previsão depende só das colunas da própria partida.
- **Regime de mando reduzido.** Entre agosto de 2020 e setembro de 2021 a vantagem do mandante cai de forma abrupta no treino e depois volta. Uma feature calculada pela data da própria linha isola esse período, para que ele não contamine a previsão de 2022 em diante.
- **Busca evolutiva entre famílias de modelos.** Oito famílias do Scikit-Learn competem em uma busca com seleção por torneio, cruzamento e mutação, e a meta é chegar o mais perto possível do LogLoss do mercado. O modelo final é a média dos 5 melhores indivíduos das famílias lineares.

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `brasileirao_probabilidades.ipynb` | Caderno completo, da leitura dos dados aos arquivos de submissão. |
| `submission.csv` | Submissão principal, com a feature de regime. |
| `submission_sem_regime.csv` | Mesmo modelo sem a feature de regime, para comparação. |
| `variantes/` | Submissões por família de modelo (`logit` e `ridge`), com regime. |
| `artefatos/genomas_finais.json` | Configurações escolhidas pela busca, gravadas para garantir a reprodução. |
| `docs/decisoes.md` | Registro de cada decisão, com evidência e alternativas descartadas. |
| `docs/experimentos.md` | Registro de todos os testes, com os números de cada um. |
| `docs/evolucao_log.csv` | Log da busca evolutiva, com todos os indivíduos de todas as gerações. |
| `experimentos/` | Scripts exploratórios que geraram os números de `docs/experimentos.md`. |

## Como rodar

Os dados não ficam no repositório. Baixe `train.csv`, `test.csv` e `sample_submission.csv` na aba Data da competição e coloque na pasta `data/`.

```bash
pip install -r requirements.txt
```

```bash
jupyter nbconvert --to notebook --execute brasileirao_probabilidades.ipynb --inplace
```

A execução leva cerca de 5 minutos e gera o `submission.csv`, o `submission_sem_regime.csv` e os arquivos da pasta `variantes/`, todos com as colunas na ordem `Id, A, D, H`. A busca usa semente fixa, então duas execuções no mesmo ambiente produzem os mesmos arquivos. Se o ambiente for diferente e a busca chegar a outras configurações, o caderno avisa e usa as configurações gravadas em `artefatos/genomas_finais.json`.

Os scripts da pasta `experimentos/` rodam a partir da raiz, por exemplo:

```bash
python experimentos/exp05_quebra_mando.py
```

## Regras da competição que o código respeita

- A modelagem usa apenas NumPy, Pandas e Scikit-Learn.
- Nenhum dado externo entra no pipeline.
- Cada previsão do teste usa somente a própria linha do `test.csv` e informação do `train.csv`. O caderno verifica isso prevendo partidas uma a uma e comparando com a previsão em lote.
- O placar público não é usado para ajustar o modelo.

## Aviso

Projeto acadêmico, sem finalidade comercial. As probabilidades geradas não são recomendação de aposta.
