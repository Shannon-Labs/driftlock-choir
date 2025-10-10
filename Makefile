.PHONY: run-clean run-multipath run-58 test lint docs figures

run-clean:
	python3 run_experiment.py 	  --band 2.4GHz 	  --channel-profile line_of_sight 	  --duration-ms 2.0 	  --sampling-rate-msps 40 	  --tau-ps 13.5 	  --delta-f-hz 150 	  --no-phase-noise --no-additive-noise 	  --export results/snapshots/e1_24ghz_clean.json

run-multipath:
	python3 run_experiment.py 	  --band 2.4GHz 	  --channel-profile rf_multipath 	  --duration-ms 4.0 	  --sampling-rate-msps 40 	  --tau-ps 13.5 	  --delta-f-hz 150 	  --snr-db 15 	  --export results/snapshots/e1_multipath_noisy.json || true

run-58:
	python3 run_experiment.py 	  --band 5.8GHz 	  --channel-profile line_of_sight 	  --duration-ms 2.0 	  --sampling-rate-msps 40 	  --tau-ps 9.5 	  --delta-f-hz 240 	  --no-phase-noise --no-additive-noise 	  --export results/snapshots/e1_58ghz_clean.json

lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

test:
	pytest -v

docs:
	@if command -v bundle >/dev/null 2>&1; then 	  bundle exec jekyll build --source docs --destination site; 	else 	  echo "bundle (Ruby) is required to build docs locally."; 	  exit 1; 	fi

figures:
	python3 generate_hero_beat_note.py
	python3 create_enhanced_visualization.py
