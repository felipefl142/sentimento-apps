# sentimento-apps

Análise de sentimento **contínua** de reviews em português da Google Play para 5 apps brasileiros (Nubank, iFood, Mercado Livre, gov.br, WhatsApp). O projeto coleta reviews novas todo dia, compara um modelo clássico com dois transformers, registra o melhor no MLflow, detecta picos de reclamação (com os termos que dispararam) e monitora drift. Não é stack de produção: **DuckDB + MLflow + Streamlit**, tudo local.

```
Google Play ──scraper──► bronze (JSONL.gz por coleta, nunca sobrescrito)
                              │ DuckDB: versão mais recente por review, rótulo pelas estrelas
                              ▼
                           silver ──► split temporal ──► tfidf_lr · xlmr_zeroshot · bertimbau_ft
                              │                                  │ MLflow (runs aninhados)
                              │                                  ▼
                              │                    registry `sentimento_reviews@champion` (pyfunc)
                              ▼                                  │
                gold/review_scores ◄──── pontuação incremental ──┘
                              │
      ┌───────────────┬───────┴────────┬────────────────────┬──────────────────┐
 sentimento diário   termos (log-odds)   tópicos (NMF)   alertas (z robusto)   monitoramento
      └───────────────┴────────────────┴────────────────────┴──────────────────┘
                                         Streamlit
```

## Como rodar

```bash
make install        # venv + torch CUDA (cu128) + requirements.txt
make ingest-full    # backfill: até 365 dias ou 80 mil reviews por app (~10 min)
make all            # silver → quality → treino (GPU, ~20 min) → predict → gold → análises → monitor
make dashboard      # http://localhost:8501
make mlflow         # http://localhost:5000
```

- Sem GPU NVIDIA: `TORCH_INDEX=https://download.pytorch.org/whl/cpu make install` e `make all-fast`, que usa só o TF-IDF.
- Rotina diária: `make daily` (cron/systemd) roda coleta incremental, pontuação, análises e monitoramento, e retreina só se o monitor recomendar.
- `make test` roda 18 testes offline, com bronze sintético e MLflow temporário.

## Dados

| app | reviews | período | % neg | % neu | % pos |
|---|---|---|---|---|---|
| Nubank | 70 mil | 365 dias | 20% | 4% | 76% |
| iFood | 72 mil | 365 dias | 17% | 2% | 81% |
| Mercado Livre | 80 mil (teto) | ~6 meses | 12% | 3% | 85% |
| gov.br | 80 mil (teto) | ~6 meses | 33% | 4% | 63% |
| WhatsApp | 80 mil (teto) | ~4 meses | 21% | 5% | 74% |

O rótulo vem das estrelas: 1–2★ negativo, 3★ neutro, 4–5★ positivo. A coleta incremental re-baixa os últimos 7 dias, porque reviews editadas voltam ao topo com o mesmo id; o silver fica com a versão mais recente.

## Resultados

Split **temporal**: teste = últimos 30 dias (53.990 reviews, distribuição real), valid = 30 dias anteriores, treino = o resto. Métrica primária: **macro-F1**.

| modelo | macro-F1 | acurácia | F1 neg | F1 neu | F1 pos | treino |
|---|---|---|---|---|---|---|
| **BERTimbau fine-tune** (champion) | **0,633** | 0,833 | 0,747 | 0,233 | 0,919 | 3 épocas, ~15 min (RTX 3070) |
| TF-IDF (palavra + char) + LogReg | 0,619 | 0,828 | 0,741 | 0,203 | 0,912 | ~4 min CPU |
| XLM-R zero-shot (Cardiff, tweets) | 0,575 | 0,785 | 0,754 | 0,096 | 0,875 | — |

Macro-F1 por app (BERTimbau): iFood 0,71 · Mercado Livre 0,68 · gov.br 0,66 · Nubank 0,63 · WhatsApp 0,57.

### O que os números dizem

- **O teto é o rótulo, não o modelo.** Dos erros do BERTimbau, 17% têm confiança acima de 0,9. Nas trocas de polaridade confiantes, o texto quase sempre discorda das estrelas. Exemplos com 5★: "péssima experiência… não funciona", "fiz uma compra, veio faltando metade dos itens… nunca mais". Com 4★: "Um lixo de aplicativo!!! Simplesmente não funciona". Por isso o fine-tune ganha só +1,4 ponto sobre o TF-IDF, e os dois concordam em 89% das reviews.
- **O neutro é o gargalo do macro-F1.** 3★ são só 2–5% das reviews e misturam elogio com ressalva e reclamação moderada. O F1 do neutro fica entre 0,10 e 0,23, enquanto negativo e positivo passam de 0,74 e 0,91. Acurácia esconderia isso: um modelo que nunca prevê neutro ainda faz ~0,83.
- **O zero-shot é bom em negativo e ruim em neutro.** Ele tem o melhor F1 negativo (0,754), mas foi treinado com tweets, onde "neutro" significa informativo, e não "3 estrelas".
- **O WhatsApp é o mais difícil.** Mistura reclamações de banimento, pedidos de funcionalidade e reviews fora de contexto.

### Alertas encontrados no histórico (sem nenhuma supervisão)

| app | dia | % neg (base) | termos que dispararam |
|---|---|---|---|
| Nubank | 07/02/2026 | 64% (20%) · 1.483 reviews | erro, dando erro, fora do ar, parou |
| Nubank | 15/07 e 04/09/2026 | 40–50% (20%) | **pix**, fora do ar, transferência |
| Nubank | 10–11/04/2026 | 39% (18%) | **mundial, palmeiras, corinthians**, "melhor deixar dinheiro" |
| gov.br | 25–30/04/2026 (6 dias, v3.8.1) | 62–69% (33%) | bloqueado, identidade, reconhecimento, INSS |
| WhatsApp | 03/08/2026 | 45% (24%) · 1.502 reviews | **enviar fotos**, vídeos |
| WhatsApp | 02/07/2026 | 32% (20%) | **tela preta**, tela branca, piscando |
| Mercado Livre | 15/09/2026 | 18% (12%) | fechando sozinho, bug |

## Decisões de projeto

- **Um pyfunc para todos os candidatos** (`SentimentModel`). O dashboard, o `predict` e o gate carregam `models:/sentimento_reviews@champion` e recebem `p_neg, p_neu, p_pos, pred`, seja o modelo sklearn ou HF. A limpeza de texto roda dentro do modelo, então o texto livre do dashboard segue o mesmo caminho do silver.
- **Gate honesto.** O champion atual é reavaliado no **mesmo** teste novo, e o candidato só recebe o alias se `F1 ≥ F1_champion − 0,005`. A versão é registrada de qualquer forma, com a tag `promotion` explicando a decisão. Não há refit: o que é servido é exatamente o que foi avaliado.
- **Artefatos.** O modelo TF-IDF é sempre logado (~7 MB). Transformers, só o vencedor (BERTimbau em bf16, ~210 MB). O zero-shot guarda só `model_id + revision SHA`.
- **Fine-tune com loop torch próprio**, em vez do `Trainer`: bf16 autocast, loss ponderada por classe, early stopping pelo macro-F1 do valid, teto de 6 mil reviews por app×classe. A API do `Trainer` muda a cada versão do transformers; o loop tem 60 linhas.
- **Termos por log-odds com prior de Dirichlet** (Monroe et al., 2008), e não frequência. Termos raros não dominam, e o mesmo método serve para "negativas × positivas" e para "este dia × 90 dias antes".
- **Alertas com z robusto e piso binomial:** `z = (x − mediana₂₈) / max(1,4826·MAD, √(p(1−p)/n), 0,01)`. Sem o termo binomial, um dia com 60 reviews e 45% negativas contra base de 28% vira alarme falso; o teste sintético pegou isso.
- **Tópicos (NMF) com stopwords extras**, que excluem adjetivos de sentimento e nomes dos apps. Sem elas, os "tópicos" viram "péssimo", "lixo" e "horrível". Com elas, aparecem gov.br→reconhecimento facial/código de acesso, WhatsApp→conta banida e Nubank→QR code/pix/limite.
- **Monitoramento com rótulo imediato.** Diferente de previsão, aqui a estrela chega junto com o texto, então o macro-F1 "ao vivo" é calculado já no `make update` seguinte. Drift de vocabulário (JS e OOV) é comparado ao mesmo cálculo feito no período de teste, e não a um limiar absoluto.

## Limitações

- O scraper não é oficial; o Google pode mudar o formato. O histórico acessível é limitado, e por isso há o teto de 80 mil por app.
- O rótulo por estrelas é ruidoso. Um conjunto anotado manualmente, mesmo pequeno (500 reviews), daria uma medida mais honesta da qualidade real.
- Só pt-BR e só Google Play (sem App Store).
