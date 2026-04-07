.PHONY: install train deploy stop-deploy app clean help

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Prices Predictor System — available targets:"
	@echo ""
	@echo "  make install      Install all dependencies (editable mode)"
	@echo "  make train        Run the ZenML training pipeline"
	@echo "  make deploy       Run the continuous deployment pipeline"
	@echo "  make stop-deploy  Stop the running MLflow prediction server"
	@echo "  make app          Launch the Streamlit UI"
	@echo "  make clean        Remove __pycache__, .zen, mlruns, and extracted_data"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	pip install -e .

# ── Training ──────────────────────────────────────────────────────────────────
train:
	python run_pipeline.py

# ── Deployment ────────────────────────────────────────────────────────────────
deploy:
	python run_deployment.py

stop-deploy:
	python run_deployment.py --stop-service

# ── Application ───────────────────────────────────────────────────────────────
app:
	streamlit run app.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .zen mlruns mlartifacts extracted_data
	@echo "Workspace cleaned."
