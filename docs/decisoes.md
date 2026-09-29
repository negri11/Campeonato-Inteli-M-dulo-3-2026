# Registro de decisões

Este documento lista cada decisão de modelagem do projeto, a evidência que a sustenta e o que foi descartado no caminho. Os números vêm da validação local, e o detalhe de cada teste está em `experimentos.md`. O placar público não entrou em nenhuma decisão de modelagem, e o único uso dele está descrito na D12.

## Como ler os números

Toda configuração é avaliada em três cortes temporais, sem shuffle. O corte `wf` é um walk-forward nas cinco últimas temporadas do treino. Os cortes `long` e `long2` treinam com as cinco e as quatro primeiras temporadas e preveem todas as seguintes de uma vez, o que imita a distância de até cinco anos entre treino e teste.

Cada configuração recebe duas notas. A nota `hard` é o LogLoss contra o resultado real, que é a métrica da competição. A nota `soft` é a entropia cruzada contra a probabilidade do mercado. As duas estimam a mesma coisa, e a `soft` tem muito menos ruído. Com cerca de 1.900 partidas de validação, uma diferença de `hard` menor que 0.001 não se distingue de sorte, e por isso a `soft` serve de desempate.

| Referência | wf | long | long2 | média |
|---|---|---|---|---|
| Frequência histórica | 1.0590 | 1.0519 | 1.0599 | 1.0569 |
| Mercado (meta) | 1.0016 | 0.9846 | 1.0016 | 0.9959 |
| Modelo final (ensemble) | 1.0167 | 1.0056 | 1.0209 | 1.0144 |

## D1. O alvo do treino é a probabilidade do mercado

**Decisão.** O modelo aprende a reproduzir a probabilidade implícita das odds de fechamento, com peso 1 para o mercado e 0 para o resultado real.

**Por quê.** O resultado de uma partida é uma amostra com um único sorteio da probabilidade verdadeira. O mercado é uma estimativa da mesma probabilidade com muito menos ruído. Com 3.419 partidas, a variância dos coeficientes é o que mais limita o modelo, e o alvo suave reduz essa variância.

**Evidência.** Na configuração de referência, trocar o alvo de mercado pelo resultado real leva a nota `hard` de 1.0152 para 1.0477. Misturar metade de cada leva para 1.0232. A busca evolutiva confirmou: todos os melhores indivíduos de todas as famílias usam peso 1 para o mercado.

**Custo.** O modelo herda qualquer viés sistemático do mercado. A decisão D2 trata o viés conhecido.

## D2. A margem das odds sai pelo método da potência

**Decisão.** Para cada partida, o expoente `k` que faz a soma de `(1/odd)^k` dar 1 é encontrado por bisseção, e as probabilidades são `(1/odd)^k`.

**Por quê.** A normalização proporcional assume margem igual nos três resultados, mas a casa carrega mais margem nos azarões. Isso achata as probabilidades e subestima os favoritos.

**Evidência.** O LogLoss do mercado no treino inteiro cai de 0.99696 (proporcional) para 0.99599 (potência). O sinal mais claro é a temperatura ótima: com a normalização proporcional a melhor temperatura do modelo era 0.90, o que indica probabilidades achatadas, e com a potência ela sobe para 0.96 a 1.00. Os métodos de Shin e aditivo ficaram empatados com a potência.

## D3. A validação usa três cortes temporais e duas notas

**Decisão.** A seleção usa o `fitness`, que é a média entre `hard` e `soft`, cada uma calculada como média dos três cortes.

**Por quê.** Só o walk-forward de uma temporada à frente não representa a tarefa, porque o teste vai de 2022 a 2026. Os dois cortes de horizonte longo cobrem previsões de 1 a 5 anos à frente. A nota `soft` entra no `fitness` para que a seleção não fique à mercê do ruído do resultado.

**Custo.** A `soft` favorece de leve os modelos treinados no próprio mercado. Como a D1 já mostrou que esses modelos também ganham na `hard` com folga, o viés não muda nenhuma conclusão.

## D4. Toda feature usa só a própria linha

**Decisão.** A função `build_features` só faz operações entre colunas da mesma linha. O imputador, o scaler e o encoder de times são ajustados só no treino.

**Por quê.** A regra da competição proíbe usar outras linhas do teste, de forma direta ou indireta. Padronizar com estatísticas do teste já seria uso indireto.

**Evidência.** O caderno prevê 300 partidas do teste uma a uma, em ordem embaralhada, e o resultado bate com a previsão em lote com tolerância de 1e-10.

## D5. Os times entram como efeito fixo regularizado

**Decisão.** `Home` e `Away` entram como variáveis indicadoras, com a mesma regularização L2 das outras features. Times inéditos no treino recebem efeito zero.

**Por quê.** O Elo da base é calculado só com resultados e demora para refletir a força de elenco. O efeito de time captura a diferença persistente entre o que o mercado pensa de um clube e o que o Elo diz, além do mando de campo mais forte de alguns estádios.

**Evidência.** Tirar o efeito de time piora os três cortes (`hard` de 1.0152 para 1.0183). O risco era o efeito envelhecer, já que os elencos mudam. O teste por horizonte mostra que o ganho contra o mercado é de 0.0035 com 1 ano de distância e ainda é de 0.0021 a 0.0029 com 5 e 6 anos.

**Descartado.** Usar só o efeito do mandante ou só o do visitante rende metade do ganho. Agrupar times raros e inéditos em uma categoria única deu resultado misto entre os cortes, e a busca evolutiva não selecionou essa opção em nenhum dos melhores.

## D6. Uma feature isola o regime de mando reduzido

**Decisão.** O indicador `regime` vale 1 para partidas entre 2020-08-01 e 2021-09-30 e 0 para todas as outras, incluindo todo o teste. Ele é calculado pela data da própria linha.

**Por quê.** Um modelo treinado até 2019 acerta a probabilidade média do mandante até o fim de 2019, com resíduo contra o mercado entre -0.008 e +0.006. A partir do terceiro trimestre de 2020 o resíduo cai para a faixa de -0.043 a -0.064 e fica lá até setembro de 2021. Em outubro e novembro de 2021 ele encolhe para -0.025 e -0.034, e em dezembro fica em +0.029. A quebra é abrupta e reverte, então não é tendência. Como as temporadas recentes pesam mais no treino, sem o indicador o modelo levaria esse nível reduzido para 2022 em diante.

**Evidência.** No walk-forward, o indicador melhora a `hard` de 1.0189 para 1.0174. Com o indicador, a previsão média de vitória do mandante no teste sobe de 0.463 para 0.480, que é o nível das temporadas de 2013 a 2019.

**Limite.** Só existem 166 partidas no treino depois do fim do regime. Nesse período os dois modelos empatam contra o mercado (1.0214 sem e 1.0216 com), e o modelo com regime vai melhor contra o resultado (0.9641 contra 0.9747), mas a amostra é pequena demais para fechar a questão. A decisão se apoia na forma da quebra. Por isso existe a D10.

## D7. A escolha do modelo sai de uma busca evolutiva com meta definida

**Decisão.** Oito famílias de modelos competem em uma busca com população de 32, 14 gerações, elitismo de 4, seleção por torneio de 3, cruzamento uniforme e mutação de 15% por gene. A meta é o LogLoss do mercado na validação (0.9959), e a coluna `gap_meta` mede a distância até ela.

**Por quê.** A meta do mercado é o piso de quem só tem informação pública, então ela dá uma medida absoluta de quanto falta. A busca evolutiva explora combinações de família, features e hiperparâmetros que uma grade fixa não cobre.

**Ajuste feito no caminho.** Na primeira versão a população inicial era toda aleatória, com uma única semente. O resultado dependia do sorteio: em uma execução a família `ridge` liderou, e em outra ela quase não foi explorada. A versão final semeia um indivíduo de referência por família, para que todas partam em condição justa.

**Resultado.** Foram 404 configurações únicas avaliadas. A tabela mostra o melhor indivíduo de cada família.

| Família | hard | soft | fitness | gap_meta |
|---|---|---|---|---|
| ridge (log-odds) | 1.0142 | 1.0215 | 1.0179 | 0.0183 |
| logit | 1.0149 | 1.0216 | 1.0183 | 0.0190 |
| mlp | 1.0149 | 1.0230 | 1.0190 | 0.0190 |
| ordinal | 1.0162 | 1.0221 | 1.0192 | 0.0203 |
| hgb | 1.0164 | 1.0223 | 1.0193 | 0.0204 |
| rf | 1.0212 | 1.0268 | 1.0240 | 0.0253 |
| knn | 1.0237 | 1.0316 | 1.0276 | 0.0278 |
| et | 1.0286 | 1.0342 | 1.0314 | 0.0327 |

**Leitura honesta.** A nota `soft` do melhor indivíduo foi de 1.0216 na geração 0 para 1.0215 na última. A `hard` foi de 1.0152 para 1.0142. Como a `soft` é a medida de menor ruído, a busca não achou um modelo melhor do que as sementes, ela confirmou um platô. A distância até a meta (0.018) é informação que o mercado tem e as colunas da base não têm, como escalação, lesões e contexto de tabela.

## D8. O modelo final é um ensemble de famílias convexas, com genomas gravados

**Decisão.** A previsão final é a média dos 5 melhores indivíduos das famílias `ridge`, `logit` e `ordinal`, com no máximo 3 por família. Os genomas ficam gravados em `artefatos/genomas_finais.json`.

**Por quê.** A diferença entre os melhores é menor que o ruído, então a média de vários é mais confiável que a escolha de um. As famílias convexas têm ótimo único, e o resultado delas não muda com a versão da biblioteca. A rede neural empata na validação, mas depende de inicialização e de otimizador, e ficou fora por reprodutibilidade.

**Por que gravar os genomas.** A busca é determinística na mesma máquina, mas uma diferença numérica na décima casa pode mudar a ordem de um torneio em outro ambiente. Se isso acontecer, o caderno avisa e treina os genomas gravados, e o arquivo de submissão sai igual.

**Resultado.** O ensemble tem 3 indivíduos `ridge` e 2 `logit`, com `hard` 1.0144 e `soft` 1.0214.

**Por que misturar as duas famílias em vez de ficar só com a melhor.** A busca prefere `ridge` pela nota `hard`, por 0.0010 no corte mais longo. A comparação partida a partida mostra que essa diferença muda de sinal entre temporadas (de +0.0014 em 2017 a -0.0046 em 2018) e que na nota `soft` as famílias empatam. As duas são equivalentes dentro do ruído, então a mistura é a escolha que não aposta em nenhuma.

## D9. A temperatura só é aplicada se passar de um limiar

**Decisão.** A temperatura é testada de 0.80 a 1.30, e só é aplicada se melhorar a `hard` em pelo menos 0.0005. O valor aplicado foi 1.0.

**Por quê.** Um ganho menor que o limiar é ruído, e aplicar um ajuste por ruído só acrescenta um parâmetro sem base. Depois da D2 a temperatura ótima ficou em 1.0, o que mostra que o modelo já sai calibrado.

## D10. Escolha das duas submissões finais

**Decisão.** A primeira vaga é o `submission.csv`, que é a mistura de `ridge` e `logit` com regime. A segunda vaga é o `variantes/submission_logit.csv`, que é o ensemble só de `logit` com regime.

**Plano original.** A segunda vaga seria o `submission_sem_regime.csv`, como proteção contra a aposta da D6. As duas versões diferem em média 0.013 por probabilidade, e a validação local não consegue medir essa aposta por inteiro.

**Por que o plano mudou.** A versão com regime fez 1.01884 no placar público e a versão sem regime fez 1.02175, com todo o resto igual. A diferença de 0.0029 aponta na mesma direção que a análise do treino. Com duas evidências independentes a favor do regime, a proteção contra ele deixou de ser o melhor uso da segunda vaga. A incerteza que sobra é entre as famílias de modelo, que a validação local não separa, e por isso a segunda vaga cobre essa dimensão.

**Custo.** Se a vantagem do mandante no placar privado for menor que no público, as duas vagas perdem juntas. O placar público e o privado são amostras das mesmas temporadas, então esse risco é pequeno, mas não é zero.

## D11. O que foi testado e descartado

- **Gradient boosting, florestas e kNN.** Todos ficam atrás dos modelos lineares na `hard` e na `soft`. A relação entre força e resultado é quase linear, e a base é pequena demais para pagar a flexibilidade.
- **Blend de logística com gradient boosting.** Nenhum peso de mistura melhorou a nota em relação à logística sozinha.
- **Polinômios e interações.** Diferença de Elo ao quadrado e ao cubo, e interações de Elo e pontos por jogo com a rodada, não mexeram na `soft` (1.0219 a 1.0221).
- **Features de calendário.** Mês e jogo no meio da semana pioraram de leve, e a busca evolutiva não as selecionou.
- **Indicadores de time recém-promovido e de primeira rodada.** Não houve diferença mensurável.
- **Peso de recência.** Meia-vida de 2, 4 ou 8 temporadas dá quase o mesmo resultado, e todas ganham por pouco de não usar peso. A busca escolheu 4 e 2.
- **Alvo misto.** Pesos de 0.5 e 0.8 para o mercado perdem para o peso 1.
- **Coeficiente de regime sem encolhimento.** Reduzir a penalização L2 sobre a coluna de regime não muda nenhuma nota nem a previsão média no teste, então o coeficiente já é estimado sem viés relevante.

## D12. O papel do placar público

**Decisão.** Nenhuma decisão de modelagem usa o placar público. Features, famílias, hiperparâmetros, ensemble e temperatura saem da validação local. O placar público entra em uma única escolha, que é qual arquivo ocupa a segunda vaga de submissão final (D10).

**Por quê.** O placar público usa 30% do teste, cerca de 517 partidas. O score absoluto tem ruído de cerca de 0.018. A diferença entre duas submissões parecidas tem ruído menor, de cerca de 0.001, mas isso ainda é do tamanho das diferenças entre os modelos deste projeto. Ajustar o modelo por ele seria ajustar a ruído.

**O que foi feito com as 5 submissões do dia.** Cada uma testou uma hipótese que já estava formulada antes do envio: a v1 conferiu formato e faixa de score, o par com e sem regime testou a D6, e o par `logit` e `ridge` testou a equivalência da D8. Os resultados estão em `experimentos.md`. Não houve sondagem do placar para ajustar intercepto, temperatura ou qualquer parâmetro.
