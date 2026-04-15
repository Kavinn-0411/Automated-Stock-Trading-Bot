#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
LSTM_CHECKPOINT="models/saved/multi_lstm.pkl"
PORTFOLIO_DIR="Final_Submission/portfolios"
OUTPUT_DIR="evaluation/outputs"
BALANCE_TOL="15"
AUTO_THRESHOLD="1"
TICKERS_CSV="META,AAPL,AMZN,NFLX,GOOGL"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run evaluation.compare_all across multiple tickers.

Usage:
  scripts/run_compare_all.sh [options] [-- <extra compare_all args>]

Options:
  --tickers-csv <csv>        Comma-separated tickers (default: META,AAPL,AMZN,NFLX,GOOGL)
  --lstm-checkpoint <path>   LSTM checkpoint path (default: models/saved/multi_lstm.pkl)
  --portfolio-dir <path>     PPO portfolio dir (default: Final_Submission/portfolios)
  --output-dir <path>        Output directory for comparison CSV/PNG (default: evaluation/outputs)
  --balance-tol <number>     Start-balance tolerance in dollars (default: 15)
  --no-auto-threshold        Disable --auto-threshold
  --python <bin>             Python executable to use (default: python)
  -h, --help                 Show this help

Examples:
  scripts/run_compare_all.sh
  scripts/run_compare_all.sh --tickers-csv "AAPL,AMZN" --balance-tol 10
  scripts/run_compare_all.sh --no-auto-threshold -- --lstm-threshold 0.001
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tickers-csv)
      TICKERS_CSV="$2"
      shift 2
      ;;
    --lstm-checkpoint)
      LSTM_CHECKPOINT="$2"
      shift 2
      ;;
    --portfolio-dir)
      PORTFOLIO_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --balance-tol)
      BALANCE_TOL="$2"
      shift 2
      ;;
    --no-auto-threshold)
      AUTO_THRESHOLD="0"
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

IFS=',' read -r -a RAW_TICKERS <<< "$TICKERS_CSV"
TICKERS=()
for t in "${RAW_TICKERS[@]}"; do
  clean="${t//[[:space:]]/}"
  if [[ -n "$clean" ]]; then
    TICKERS+=("${clean^^}")
  fi
done

if [[ ${#TICKERS[@]} -eq 0 ]]; then
  echo "[run_compare_all] No tickers provided after parsing --tickers-csv." >&2
  exit 1
fi

echo "[run_compare_all] Tickers       : ${TICKERS[*]}"
echo "[run_compare_all] LSTM checkpoint: $LSTM_CHECKPOINT"
echo "[run_compare_all] Portfolio dir  : $PORTFOLIO_DIR"
echo "[run_compare_all] Output dir     : $OUTPUT_DIR"
echo "[run_compare_all] Balance tol    : $BALANCE_TOL"
echo "[run_compare_all] Auto threshold : $AUTO_THRESHOLD"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "[run_compare_all] Extra args     : ${EXTRA_ARGS[*]}"
fi

for t in "${TICKERS[@]}"; do
  echo
  echo "=============================="
  echo "Running compare_all for $t"
  echo "=============================="

  cmd=(
    "$PYTHON_BIN" -m evaluation.compare_all
    --ticker "$t"
    --lstm-checkpoint "$LSTM_CHECKPOINT"
    --portfolio-dir "$PORTFOLIO_DIR"
    --output-dir "$OUTPUT_DIR"
    --balance-tol "$BALANCE_TOL"
  )

  if [[ "$AUTO_THRESHOLD" == "1" ]]; then
    cmd+=(--auto-threshold)
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  "${cmd[@]}"
done

echo
echo "[run_compare_all] Done."
