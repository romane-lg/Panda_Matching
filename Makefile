PYTHON := uv run python

.PHONY: eval-free eval-paid

eval-free:
	$(PYTHON) scripts/run_offline_regression_suite.py --local-only

eval-paid:
	$(PYTHON) scripts/run_baseline_eval.py \
		--dataset-name $${DATASET:-panda_chatbot_llm_benchmark_v1} \
		--run-name $${RUN_NAME:-llm-benchmark-manual} \
		--exclude-scorer panda_tool_call_correctness
