# Curatio — architecture

Renders on GitHub. To produce `architecture.png` for the README, export this
diagram (e.g. mermaid-cli: `mmdc -i assets/architecture.md -o assets/architecture.png`)
or paste the block into the README (GitHub renders mermaid inline).

## Pipeline

```mermaid
flowchart TB
    subgraph offline["Offline — one-time, cached"]
        HF[/"HF web-text slice<br/>(FineWeb / C4)"/]
        SAMPLE["sample ~1k docs"]
        JUDGE["Command A judge<br/>edu score 1-5"]
        EMB0["Embed v4"]
        TRAIN["train LogisticRegression<br/>embedding -> quality"]
        EXPORT[("quality_model.json<br/>coeffs + intercept + dim")]
        HF --> SAMPLE --> JUDGE
        SAMPLE --> EMB0
        JUDGE --> TRAIN
        EMB0 --> TRAIN --> EXPORT
    end

    subgraph online["Online — streaming"]
        PROD["Producer<br/>replay HF slice"]
        KAFKA{{"Redpanda topic: raw-docs<br/>(P partitions)"}}
        PROD -->|"key = MinHash band<br/>(near-dups co-locate)"| KAFKA

        subgraph runtimes["Two runtimes, one stage contract"]
            PY["Python Curator<br/>(reference / correctness)"]
            RS["Rust/Axum worker x N<br/>(performance / scale-out)"]
        end
        KAFKA --> PY
        KAFKA --> RS

        subgraph stages["Curation stages"]
            direction TB
            S1["1. Heuristic filters<br/>(no API)"]
            S2["2. Embed v4<br/>(batched, cached)"]
            S3["3. Quality classifier<br/>(dot product)"]
            S4["4. Online dedup<br/>(incremental HNSW)"]
            S1 -->|pass| S2 --> S3 -->|pass| S4
        end
        PY --> stages
        RS --> stages

        CLEAN[("clean-docs<br/>Parquet")]
        REJ[("rejected<br/>stream + log")]
        S4 -->|keep + add to index| CLEAN
        S1 -.reject reason.-> REJ
        S3 -.low_quality.-> REJ
        S4 -.near_duplicate.-> REJ
    end

    EXPORT -.loaded by both.-> S3

    subgraph obs["Observability"]
        API["FastAPI (Py) / Axum (Rust)<br/>/stats /ws/metrics /health"]
        DASH["React dashboard<br/>live funnel"]
        DUCK["DuckDB stats"]
        CHARTS["eval + benchmark<br/>PNG charts"]
        API --> DASH
        DUCK --> CHARTS
    end
    stages --> API
    CLEAN --> DUCK

    EMB1["Embed v4 API"]
    S2 <-->|"2000 inputs/min<br/>(the only throttle)"| EMB1
```

## Backpressure & offsets (NFR4)

```mermaid
flowchart LR
    POLL["poll raw-docs"] --> RUN["run stages"]
    RUN --> TERM{"terminal state?<br/>(sinked or rejected)"}
    TERM -->|yes| COMMIT["commit offset"] --> POLL
    TERM -->|"Embed 429"| PAUSE["pause partitions<br/>+ backoff"] --> RUN
```

At-least-once, no silent loss: the offset advances only after a doc is sinked or
rejected. On a sustained Embed 429 the consumer pauses its poll (Kafka buffers
upstream) — throughput drops, correctness holds.

## Python ↔ Rust parity (NFR5 / SC6)

Both runtimes load the same `quality_model.json`, apply identical heuristic rules
in the same order, and build the HNSW index with identical `M`/`ef` and a
single-threaded insertion order. `make bench` diffs each runtime's keep/reject
decision set against the Python reference on a fixed pre-embedded input and refuses
to report a throughput number unless they are 100% identical.
