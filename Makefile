.PHONY: install redpanda topic label demo demo-offline demo-rust eval bench stress charts lint test clean

# Mirror .env defaults; override on the CLI, e.g. make topic RAW_PARTITIONS=8
RAW_TOPIC ?= raw-docs
REJECTED_TOPIC ?= rejected
RAW_PARTITIONS ?= 4

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
	cd ui && npm install

redpanda:
	docker compose up -d redpanda

# raw-docs is partitioned (RAW_PARTITIONS) so Friday's Rust scaling test has
# parallelism to consume; auto-create would give 1 partition. Idempotent.
topic: redpanda
	docker exec Curatio-redpanda rpk topic create $(RAW_TOPIC) -p $(RAW_PARTITIONS) 2>/dev/null || true
	docker exec Curatio-redpanda rpk topic create $(REJECTED_TOPIC) -p 1 2>/dev/null || true

# One-time: Command A labels a sample, then distill + export quality_model.json
label:
	python -m src.label.judge
	python -m src.label.train

# Python runtime: producer + FastAPI (which embeds the curator thread + metrics) +
# React dashboard. The consumer no longer runs standalone — it lives inside the API
# process so it shares the in-memory MetricsStore the dashboard reads (mirrors the
# Rust worker: one process consumes Kafka AND serves metrics).
# To replay the cached 200-doc sample, first reset the group:
#   docker exec Curatio-redpanda rpk group seek Curatio-curator --to start
demo: redpanda
	python -m src.source.producer & \
	uvicorn src.api.app:app --port 8000 & \
	cd ui && npm run dev

# Quota-free offline demo: no Kafka, no Embed API. The FastAPI app replays the real
# eval funnel (eval/results/*.json) into the dashboard at ~30 docs/sec. Use this for
# the demo GIF without a production Cohere key. Dashboard at http://localhost:5173.
demo-offline:
	Curatio_FAKE_FEED=1 uvicorn src.api.app:app --port 8000 & \
	cd ui && npm run dev

# Rust runtime: N parallel workers over the partitioned topic (Axum metrics)
demo-rust: redpanda
	cargo run --release --manifest-path rust/Cargo.toml -- --workers 4

eval:        # curation quality: dedup P/R, classifier AUC, funnel
	python -m eval.run

bench:       # parity gate + Python-vs-Rust + scaling throughput
	python -m eval.bench

stress:      # honest stress tests: dedup precision vs N, recall vs partition count
	python -m eval.stress

charts:
	python -m eval.plot

lint:
	ruff check src eval tests
	mypy src eval
	cargo clippy --manifest-path rust/Cargo.toml

test:
	pytest -q
	cargo test --manifest-path rust/Cargo.toml

clean:
	rm -rf .cache data/*.parquet eval/results/*.json
