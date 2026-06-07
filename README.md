# Lord of the Cine

## Avaliação Quantitativa

A Fase 9 avalia o sistema de recomendação por meio de quatro métricas:

- **Precision@10:** proporção de filmes relevantes entre as dez primeiras recomendações.
- **Recall@10:** proporção dos filmes relevantes do conjunto de teste recuperada nas dez primeiras posições.
- **NDCG@10:** qualidade do ranqueamento, atribuindo maior importância aos itens relevantes apresentados nas primeiras posições.
- **Coverage:** proporção do catálogo que foi recomendada ao menos uma vez durante a avaliação.

O experimento compara um **Popularity Baseline** com cinco configurações do recomendador híbrido, variando os pesos do perfil do usuário, da avaliação média e da popularidade. A configuração **Hybrid 30/30/40** apresentou os melhores resultados de Precision@10, Recall@10 e NDCG@10. Já a configuração **Hybrid 70/20/10** obteve a maior cobertura.

Os resultados evidenciam um trade-off entre precisão e cobertura: o aumento do peso da popularidade favoreceu a acurácia das recomendações, enquanto o aumento do peso do perfil do usuário ampliou a variedade de filmes recomendados.

Os resultados completos estão disponíveis em [`reports/evaluation_results.csv`](reports/evaluation_results.csv).
