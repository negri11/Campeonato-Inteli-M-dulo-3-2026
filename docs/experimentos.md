# Registro de experimentos

Este documento registra os testes na ordem em que foram feitos, com os números de cada um. Os scripts estão na pasta `experimentos/` e rodam a partir da raiz do repositório, com os dados em `data/`. As conclusões que viraram decisão estão em `decisoes.md`.

As colunas `wf`, `long` e `long2` são os três cortes temporais de validação. A nota `hard` é o LogLoss contra o resultado real, e a nota `soft` é a entropia cruzada contra o mercado. Quando aparecem sem indicação de corte, são a média dos três.

## Exp 01. Método de de-vig, deriva do mando, efeito de time e recência

Script: `experimentos/exp01_devig_deriva_times.py`

### LogLoss do mercado no treino inteiro, por método de de-vig

| Método | LogLoss | 2013 a 2017 | 2018 a 2021 | Melhor T | LogLoss com melhor T |
|---|---|---|---|---|---|
| Proporcional | 0.99696 | 1.00589 | 0.98580 | 0.88 | 0.99570 |
| Potência | 0.99599 | 1.00513 | 0.98458 | 0.96 | 0.99580 |
| Shin | 0.99618 | 1.00541 | 0.98466 | 0.94 | 0.99579 |
| Aditivo | 0.99603 | 1.00539 | 0.98434 | 0.96 | 0.99583 |

A normalização proporcional pede temperatura de 0.88 para ficar calibrada, o que mostra que ela achata as probabilidades. Os outros três métodos já saem quase calibrados.

### Probabilidade média do mandante por temporada

| Temporada | Mercado H | Mercado D | Mercado A | Real H | Real D | Real A |
|---|---|---|---|---|---|---|
| 2013 | 0.476 | 0.263 | 0.261 | 0.484 | 0.284 | 0.232 |
| 2014 | 0.478 | 0.268 | 0.254 | 0.518 | 0.242 | 0.239 |
| 2015 | 0.487 | 0.270 | 0.244 | 0.526 | 0.239 | 0.234 |
| 2016 | 0.490 | 0.265 | 0.245 | 0.533 | 0.248 | 0.219 |
| 2017 | 0.495 | 0.265 | 0.240 | 0.439 | 0.271 | 0.289 |
| 2018 | 0.480 | 0.273 | 0.247 | 0.532 | 0.289 | 0.179 |
| 2019 | 0.481 | 0.265 | 0.255 | 0.484 | 0.258 | 0.258 |
| 2020 | 0.438 | 0.271 | 0.292 | 0.450 | 0.284 | 0.266 |
| 2021 | 0.443 | 0.280 | 0.277 | 0.458 | 0.297 | 0.245 |

O mercado fica entre 0.476 e 0.495 de 2013 a 2019 e cai para 0.44 em 2020 e 2021. A frequência real oscila muito mais que o mercado, o que ilustra o ruído do resultado.

### Efeito de time (logística, alvo de mercado, C 0.1, meia-vida 4)

| Efeito de time | Escala | wf | long | long2 |
|---|---|---|---|---|
| Nenhum | | 1.0232 | 1.0100 | 1.0246 |
| Só mandante | 1.0 | 1.0216 | 1.0088 | 1.0238 |
| Só visitante | 1.0 | 1.0215 | 1.0078 | 1.0234 |
| Os dois | 0.5 | 1.0213 | 1.0079 | 1.0234 |
| Os dois | 1.0 | 1.0199 | 1.0068 | 1.0227 |
| Os dois | 1.5 | 1.0194 | 1.0069 | 1.0229 |
| Os dois | 2.5 | 1.0194 | 1.0082 | 1.0238 |

A escala multiplica as colunas de time e equivale a regularizar menos os efeitos. A escala 1.0 é a mais estável entre os cortes.

### Peso de recência (meia-vida em temporadas)

| Meia-vida | wf | long | long2 |
|---|---|---|---|
| Sem peso | 1.0203 | 1.0071 | 1.0228 |
| 8 | 1.0200 | 1.0068 | 1.0227 |
| 4 | 1.0199 | 1.0068 | 1.0227 |
| 2 | 1.0196 | 1.0067 | 1.0227 |
| 1 | 1.0196 | 1.0069 | 1.0228 |

### Quanto o modelo reproduz o mercado fora da amostra

A correlação entre a previsão do modelo e a probabilidade do mercado no walk-forward é de 0.824 para A, 0.729 para D e 0.824 para H. O empate é a classe que as features explicam pior.

## Exp 02. Conjuntos de features, gradient boosting e blends

Script: `experimentos/exp02_features_blends.py`

| Conjunto de features | hard | soft |
|---|---|---|
| Só diferença de Elo | 1.0168 | 1.0225 |
| Elo e pontos por jogo | 1.0176 | 1.0226 |
| Força (Elo, forma, saldo, pontos por jogo) | 1.0168 | 1.0221 |
| Tudo em diferença | 1.0164 | 1.0219 |
| Tudo sem o módulo do Elo | 1.0162 | 1.0220 |
| Tudo com polinômios do Elo | 1.0167 | 1.0221 |
| Tudo com interações com a rodada | 1.0167 | 1.0219 |
| Tudo com indicadores extras | 1.0165 | 1.0219 |
| Colunas brutas | 1.0158 | 1.0218 |
| Todas as anteriores juntas | 1.0169 | 1.0219 |

A diferença entre o pior e o melhor conjunto é de 0.0008 na `soft`. Quase todo o sinal está no Elo e no efeito de time.

| Gradient boosting com alvo de mercado | hard | soft |
|---|---|---|
| Profundidade 2, taxa 0.03, 150 iterações | 1.0203 | 1.0259 |
| Profundidade 2, taxa 0.03, 300 iterações | 1.0199 | 1.0257 |
| Profundidade 3, taxa 0.03, 200 iterações | 1.0203 | 1.0262 |

| Blend logística e gradient boosting | hard | soft |
|---|---|---|
| Só logística | 1.0164 | 1.0219 |
| 10% de gradient boosting | 1.0166 | 1.0221 |
| 20% de gradient boosting | 1.0167 | 1.0223 |
| 30% de gradient boosting | 1.0170 | 1.0226 |
| 50% de gradient boosting | 1.0175 | 1.0232 |

## Exp 03. Ganho do efeito de time por horizonte

Script: `experimentos/exp03_horizonte_efeito_time.py`

O modelo treina até um ano de corte (2015, 2016, 2017 e 2018) e prevê cada temporada seguinte. O ganho é a nota do modelo sem efeito de time menos a nota do modelo com.

| Horizonte em anos | Ganho contra o mercado | Ganho contra o resultado |
|---|---|---|
| 1 | 0.0034 | 0.0031 |
| 2 | 0.0027 | 0.0027 |
| 3 | 0.0027 | 0.0021 |
| 4 | 0.0026 | 0.0016 |
| 5 | 0.0022 | 0.0030 |
| 6 | 0.0028 | 0.0023 |

## Exp 04. Modelo ordinal, regressão em log-odds e features de calendário

Script: `experimentos/exp04_ordinal_calendario.py`

### Coeficiente da diferença de Elo por temporada

Regressão logística binária de vitória do mandante contra o mercado, uma por temporada.

| Temporada | Coeficiente por 100 pontos | Intercepto | Desvio padrão da diferença de Elo |
|---|---|---|---|
| 2013 | 0.658 | -0.095 | 59.1 |
| 2014 | 0.582 | -0.081 | 67.5 |
| 2015 | 0.523 | -0.043 | 69.5 |
| 2016 | 0.606 | -0.035 | 67.9 |
| 2017 | 0.569 | -0.028 | 66.3 |
| 2018 | 0.771 | -0.080 | 68.8 |
| 2019 | 0.729 | -0.080 | 87.3 |
| 2020 | 0.660 | -0.279 | 82.4 |
| 2021 | 0.672 | -0.241 | 75.5 |

O intercepto de 2020 e 2021 é a primeira pista do regime de mando reduzido. O coeficiente das temporadas recentes é maior que o das primeiras, o que justifica o peso de recência.

### Features adicionais sobre as colunas brutas

| Conjunto | hard | soft |
|---|---|---|
| Colunas brutas | 1.0156 | 1.0217 |
| Com calendário (mês, meio de semana) | 1.0159 | 1.0219 |
| Com indicador de time novo e primeira rodada | 1.0156 | 1.0217 |
| Com descanso detalhado | 1.0158 | 1.0216 |
| Com pontos por jogo na reta final | 1.0155 | 1.0216 |
| Com todas | 1.0158 | 1.0219 |

### Famílias lineares

| Modelo | hard | soft |
|---|---|---|
| Logística multinomial, C 0.2 | 1.0156 | 1.0217 |
| Ordinal, C 0.2 | 1.0168 | 1.0223 |
| Ordinal, C 0.5 | 1.0168 | 1.0225 |
| Ridge em log-odds, C 0.03 | 1.0153 | 1.0219 |
| Ridge em log-odds, C 0.1 | 1.0152 | 1.0218 |

Os blends entre logística, ordinal e ridge ficaram todos entre 1.0154 e 1.0164 de `hard`, sem ganho sobre o melhor componente.

## Exp 05. Quebra estrutural do mando

Script: `experimentos/exp05_quebra_mando.py`

O modelo treina com as temporadas até 2019. O resíduo é a probabilidade de vitória do mandante do mercado menos a do modelo.

| Trimestre | Partidas | Mercado H | Modelo H | Resíduo |
|---|---|---|---|---|
| 2018 T2 | 119 | 0.481 | 0.477 | +0.005 |
| 2018 T3 | 150 | 0.477 | 0.483 | -0.007 |
| 2018 T4 | 111 | 0.483 | 0.487 | -0.004 |
| 2019 T2 | 89 | 0.466 | 0.474 | -0.007 |
| 2019 T3 | 126 | 0.491 | 0.484 | +0.007 |
| 2019 T4 | 165 | 0.481 | 0.484 | -0.004 |
| 2020 T3 | 113 | 0.430 | 0.479 | -0.049 |
| 2020 T4 | 155 | 0.437 | 0.490 | -0.053 |
| 2021 T1 | 112 | 0.446 | 0.492 | -0.046 |
| 2021 T2 | 70 | 0.447 | 0.490 | -0.043 |
| 2021 T3 | 144 | 0.415 | 0.479 | -0.064 |
| 2021 T4 | 166 | 0.466 | 0.486 | -0.020 |

| Mês de 2021 | Partidas | Mercado H | Resíduo |
|---|---|---|---|
| Maio | 10 | 0.463 | -0.056 |
| Junho | 60 | 0.444 | -0.041 |
| Julho | 58 | 0.423 | -0.059 |
| Agosto | 49 | 0.407 | -0.071 |
| Setembro | 37 | 0.415 | -0.062 |
| Outubro | 71 | 0.462 | -0.025 |
| Novembro | 67 | 0.459 | -0.035 |
| Dezembro | 28 | 0.493 | +0.029 |

## Exp 06. Feature de regime

Script: `experimentos/exp06_regime.py`

### Teste no período pós-regime

Treino até 2021-09-30, teste de outubro a dezembro de 2021, com 166 partidas. No período o mercado dá 0.466 para o mandante e a frequência real foi 0.560.

| Modelo | Meia-vida | hard | soft | Média prevista de H |
|---|---|---|---|---|
| Sem regime | Sem peso | 0.9766 | 1.0226 | 0.467 |
| Sem regime | 4 | 0.9748 | 1.0213 | 0.463 |
| Sem regime | 2 | 0.9743 | 1.0207 | 0.459 |
| Com regime | Sem peso | 0.9676 | 1.0229 | 0.485 |
| Com regime | 4 | 0.9643 | 1.0216 | 0.484 |
| Com regime | 2 | 0.9626 | 1.0208 | 0.483 |

Só em dezembro de 2021, com 28 partidas, a `soft` vai de 1.0427 (sem regime) para 1.0394 (com regime), e o mercado dá 0.493 para o mandante.

### Cortes usuais e previsão média no teste

| Modelo | Meia-vida | wf | long | long2 | Média de H no teste |
|---|---|---|---|---|---|
| Sem regime | 4 | 1.0188 | 1.0065 | 1.0216 | 0.4628 |
| Com regime | 4 | 1.0173 | 1.0065 | 1.0216 | 0.4803 |
| Com regime e interação com Elo | 4 | 1.0172 | 1.0065 | 1.0216 | 0.4804 |

Os cortes `long` e `long2` não mudam porque o treino deles termina antes do regime. A interação do regime com o Elo não acrescenta nada.

## Exp 07. Times raros e inéditos

Script: `experimentos/exp07_times_raros.py`

Times com menos jogos em casa do que o limite são agrupados em uma categoria única, e times inéditos caem nessa mesma categoria.

| Limite de jogos | wf soft | long soft | long2 soft |
|---|---|---|---|
| Sem agrupamento | 1.0200 | 1.0223 | 1.0224 |
| 20 | 1.0192 | 1.0230 | 1.0222 |
| 39 | 1.0195 | 1.0235 | 1.0224 |
| 58 | 1.0201 | 1.0227 | 1.0241 |

O agrupamento melhora um corte e piora outro. Sem direção clara, a opção virou um gene da busca evolutiva, que não a selecionou.

## Exp 08. Protótipo da busca evolutiva

Script: `experimentos/exp08_busca_evolutiva.py`

O protótipo usou população de 28, 12 gerações e uma única semente de referência. Foram 311 configurações únicas em 339 segundos. A família `ridge` liderou com fitness 1.0178, seguida de `logit` com 1.0183 e `mlp` com 1.0184.

Ao portar a busca para o caderno com a mesma população, o caminho mudou e a família `ridge` terminou com 1.0209, quase sem ser explorada. Essa diferença entre duas execuções do mesmo algoritmo foi o que motivou semear um indivíduo de referência por família na versão final.

## Busca evolutiva final

Roda dentro do caderno `brasileirao_probabilidades.ipynb`, seção 8. O log completo, com todos os indivíduos de todas as gerações, fica em `docs/evolucao_log.csv`.

| Parâmetro | Valor |
|---|---|
| População | 32 (8 sementes, uma por família, e 24 aleatórios) |
| Gerações | 14 |
| Elitismo | 4 |
| Seleção | Torneio de 3 |
| Cruzamento | Uniforme |
| Mutação | 15% por gene, perturbação gaussiana na regularização |
| Configurações únicas avaliadas | 404 |
| Tempo | 270 segundos |

| Geração | Melhor fitness | hard | soft | Mediana da população | Família do melhor |
|---|---|---|---|---|---|
| 0 | 1.0184 | 1.0152 | 1.0216 | 1.0280 | logit |
| 3 | 1.0181 | 1.0145 | 1.0216 | 1.0197 | ridge |
| 6 | 1.0179 | 1.0144 | 1.0215 | 1.0191 | ridge |
| 8 | 1.0179 | 1.0142 | 1.0215 | 1.0191 | ridge |
| 14 | 1.0179 | 1.0142 | 1.0215 | 1.0183 | ridge |

A mediana da população cai de 1.0280 para 1.0183, o que mostra que a seleção funciona. O melhor indivíduo melhora 0.0005 de fitness, quase tudo na `hard`.

### Testes das decisões de base (caderno, seção 6)

| Teste | wf | long | long2 | hard | soft |
|---|---|---|---|---|---|
| Referência | 1.0174 | 1.0065 | 1.0217 | 1.0152 | 1.0216 |
| Alvo igual ao resultado real | 1.0483 | 1.0374 | 1.0575 | 1.0477 | 1.0570 |
| Alvo metade mercado | 1.0258 | 1.0142 | 1.0297 | 1.0232 | 1.0309 |
| Sem efeito de time | 1.0212 | 1.0099 | 1.0237 | 1.0183 | 1.0245 |
| Sem feature de regime | 1.0189 | 1.0065 | 1.0217 | 1.0157 | 1.0217 |
| Sem peso de recência | 1.0181 | 1.0070 | 1.0220 | 1.0157 | 1.0217 |
| Só diferença de Elo | 1.0181 | 1.0079 | 1.0225 | 1.0162 | 1.0222 |
| Só diferença de Elo, sem time, alvo real | 1.0265 | 1.0145 | 1.0291 | 1.0233 | 1.0289 |

## Submissões

| Versão | Data | Arquivo | Validação hard | Placar público | Descrição |
|---|---|---|---|---|---|
| v1 | 2026-09-29 | `submission.csv` | 1.0158 | 1.01989 | Logística com alvo de mercado, efeito de time, ensemble das 5 melhores de uma grade de 244 configurações. Sem regime. |
