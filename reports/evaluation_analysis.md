# Análise Acadêmica da Avaliação Quantitativa

## Objetivo

A avaliação quantitativa teve como objetivo comparar o desempenho de um baseline de popularidade com diferentes configurações do recomendador híbrido do Lord of the Cine. O protocolo considerou 599 usuários avaliáveis, com separação de 80% das interações para treino e 20% para teste. Foram considerados relevantes os filmes com nota igual ou superior a 4,0.

As métricas utilizadas foram Precision@10, Recall@10, NDCG@10 e Coverage. As três primeiras medem diferentes aspectos da relevância das recomendações, enquanto Coverage representa a proporção do catálogo recomendada ao menos uma vez.

## Resultados

O Popularity Baseline apresentou **Precision@10 = 0.1290**, **Recall@10 = 0.1063** e **NDCG@10 = 0.1706**. Esses valores mostram que a popularidade global constitui uma referência competitiva, pois filmes amplamente avaliados e bem recebidos possuem maior probabilidade de serem relevantes para diferentes usuários.

O melhor desempenho nas métricas de relevância foi obtido pelo **Hybrid 30/30/40**, que alcançou **Precision@10 = 0.1457**, **Recall@10 = 0.1278** e **NDCG@10 = 0.1942**. Essa configuração superou o baseline nas três métricas, indicando que a combinação de 30% para o perfil do usuário, 30% para a avaliação média e 40% para a popularidade produziu recomendações mais relevantes e melhor ordenadas.

Entre as configurações híbridas avaliadas, observa-se que, à medida que o peso da popularidade aumenta, as métricas de precisão melhoram. A progressão de Hybrid 70/20/10 até Hybrid 30/30/40 é acompanhada pelo crescimento de Precision@10, Recall@10 e NDCG@10. Esse comportamento sugere que, neste conjunto de dados e neste protocolo experimental, o sinal de popularidade possui forte capacidade preditiva.

Por outro lado, o **Hybrid 70/20/10** apresentou a maior cobertura, com **Coverage = 0.0844**. À medida que o peso do perfil do usuário aumenta, a cobertura do catálogo também aumenta. A personalização baseada em conteúdo distribui as recomendações por uma parcela maior do acervo, reduzindo a concentração exclusiva nos filmes mais populares.

## Discussão

Os resultados evidenciam um trade-off entre precisão e diversidade/cobertura. Configurações com maior peso de popularidade concentram as recomendações em itens com aceitação global e obtêm melhor desempenho nas métricas de relevância. Configurações com maior peso do perfil do usuário exploram uma parcela mais ampla do catálogo, mas apresentam menor precisão no top 10.

Esse trade-off deve ser considerado de acordo com o objetivo do sistema. Uma aplicação orientada à maximização imediata da relevância pode priorizar o Hybrid 30/30/40. Em um cenário no qual descoberta, variedade e exposição do catálogo sejam objetivos centrais, o Hybrid 70/20/10 constitui uma alternativa mais adequada.

## Conclusão

Conclui-se que o **Hybrid 30/30/40 foi o melhor modelo para precisão**, além de apresentar os maiores valores de Recall@10 e NDCG@10. O **Hybrid 70/20/10 foi o melhor modelo para diversidade**, pois alcançou a maior cobertura do catálogo.

A Fase 9 demonstra que o ajuste dos pesos altera de maneira mensurável o comportamento do recomendador. Não existe uma configuração universalmente superior em todos os critérios: a escolha final depende do equilíbrio desejado entre relevância e cobertura.
