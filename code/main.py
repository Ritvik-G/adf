"""
Mathematical Expression Evaluator
Supports multiple evaluation methods: vanilla eval, REST API, SOAP, and LLMs
"""

import argparse
import re
import time
import requests
from rest_client import evaluate_expression
from soap_client import evaluate_expression as soap_evaluate_expression
from config import load_dataset, load_llm_config

# ============================================================================
# Evaluation Methods
# ============================================================================

def vanilla(equation=None):
    """
    Evaluate equation using Python's built-in eval

    Args:
        equation (str): Mathematical equation to evaluate

    Returns:
        float: Result of the equation
    """
    if equation is None:
        raise ValueError("Equation not provided")

    # Clean and prepare expression
    exp = str(equation).strip().replace("^", "**")
    if "=" in exp:
        exp = exp.split("=", 1)[0].strip()

    try:
        return eval(exp, {"__builtins__": None}, {})
    except Exception as e:
        print(f"Error evaluating {exp!r}: {e}")
        return None


def rest_calls(equation=None, verbose=False):
    """
    Evaluate equation using REST API calls

    Args:
        equation (str): Mathematical equation to evaluate

    Returns:
        float: Result of the equation
    """
    API_BASE_URL = "http://localhost:5000"

    if equation is None:
        # Interactive mode
        print("=" * 50)
        print("Expression Evaluator using REST API")
        print("=" * 50)
        equation = input("\nEnter expression: ")

    try:
        result = evaluate_expression(
            equation,
            api_base_url=API_BASE_URL,
            verbose=verbose,
        )
        return result
    except Exception as e:
        print(f"REST API Error: {e}")
        return None


def soap_arch(equation=None, verbose=False):
    """
    Evaluate equation using SOAP architecture

    Args:
        equation (str): Mathematical equation to evaluate

    Returns:
        float: Result of the equation
    """
    SOAP_BASE_URL = "http://localhost:8000/"

    if equation is None:
        raise ValueError("Equation not provided")

    try:
        result = soap_evaluate_expression(
            equation,
            soap_base_url=SOAP_BASE_URL,
            verbose=verbose,
        )
        return result
    except Exception as e:
        print(f"SOAP Error: {e}")
        return None


def extract_number(text):
    """
    Diagnostic-only: pull a numeric value out of free-form text, e.g.
    "The answer is -3.5." -> -3.5. Used only to annotate *why* a response
    failed (did it contain a number at all, just not alone) — never to
    rewrite the graded value. Grading always uses the raw response as-is:
    the instruction was "number only," so a response that isn't already a
    clean number is a failure, not something to salvage.
    """
    if text is None:
        return None
    match = re.search(r"-?\d+\.?\d*", text)
    return match.group(0) if match else None


def llms_groq(equation=None):
    """
    Evaluate equation using Groq LLM API. Single attempt, no retry: an API
    error is an API error, and it's scored as such — not retried away.

    Args:
        equation (str): Mathematical equation to evaluate

    Returns:
        str: Raw LLM response text, or None if the request itself failed
    """
    if equation is None:
        raise ValueError("Equation not provided")

    cfg = load_llm_config()
    api_key = cfg.get("api_key")
    model = cfg.get("model")
    url = cfg.get("url")

    if not api_key or not model:
        raise ValueError("Config must include 'api_key' and 'model'")

    prompt = f"Give the number only as the output for the following equation: {equation}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.7),
        "max_tokens": cfg.get("max_tokens", 256),
        "top_p": cfg.get("top_p", 1.0),
    }

    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq API Error: {e}")
        return None


def llm_ollama(equation=None, model=None):
    """
    Evaluate equation using an Ollama local LLM.

    Args:
        equation (str): Mathematical equation to evaluate
        model (str): Ollama model tag to use. Defaults to the first entry
            in config_llm.json's "ollama_models" list.

    Returns:
        str: LLM response (numerical result)
    """
    if equation is None:
        raise ValueError("Equation not provided")

    cfg = load_llm_config()
    model = model or cfg.get("ollama_models", ["mistral"])[0]
    base_url = cfg.get("ollama_url", "http://localhost:11434")

    prompt = f"Answer with ONLY the final numerical value, no explanations or equations. QUESTION: {equation}\nANSWER:"

    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": cfg.get("temperature", 0.7),
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        res = r.json().get("response", "").strip()
        return res
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama at {base_url}")
        return None
    except Exception as e:
        print(f"Ollama Error: {e}")
        return None


# ============================================================================
# Metrics & Evaluation
# ============================================================================

def classify_result(result, expected):
    """
    Classify a raw method result against the expected answer. No leniency:
    the raw result must itself be a directly parseable number to count as
    correct. For LLM methods this means instruction-following (reply with
    ONLY the number) is part of what's being measured, not papered over.

    Returns one of:
      "api_error"      — the call itself failed (result is None)
      "format_failure" — got a response, but it isn't a clean number
      "wrong_answer"   — got a clean number, but it's the wrong value
      "correct"        — clean number, correct value
    """
    if result is None:
        return "api_error"
    try:
        value = float(result)
    except (ValueError, TypeError):
        return "format_failure"
    return "correct" if value == float(expected) else "wrong_answer"


def measurements(func, n, dataset):
    """
    Measure accuracy, execution time, and failure breakdown for a given
    evaluation function.

    Args:
        func: Function to evaluate equations
        n (int): Number of samples to test
        dataset (list): Dataset containing equations and answers

    Returns:
        dict: Dictionary with accuracy, time, and per-category failure counts
    """
    start_time = time.perf_counter()

    counts = {"correct": 0, "wrong_answer": 0, "format_failure": 0, "api_error": 0}

    for i in range(min(n, len(dataset))):
        item = dataset[i]
        equation = item.get("Equation", "")
        expected_answer = item.get("Answer", "")

        result = func(equation=equation)
        category = classify_result(result, expected_answer)
        counts[category] += 1

        if category == "wrong_answer":
            print(f"❌ WRONG_ANSWER {equation} → Got: {result}, Expected: {expected_answer}")
        elif category == "format_failure":
            hint = extract_number(str(result))
            note = f" (contained {hint}, not counted — instruction was number-only)" if hint else ""
            print(f"❌ FORMAT_FAILURE {equation} → raw: {result!r}{note}")
        elif category == "api_error":
            print(f"❌ API_ERROR {equation} → no result returned")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    accuracy = (counts["correct"] / n) * 100 if n > 0 else 0.0

    return {
        "accuracy": accuracy,
        "time": total_time,
        "correct": counts["correct"],
        "total": n,
        "failures": {k: v for k, v in counts.items() if k != "correct"},
    }


# ============================================================================
# Main Execution
# ============================================================================

METHODS = {
    "vanilla": vanilla,
    "rest": rest_calls,
    "soap": soap_arch,
    "groq": llms_groq,
    "ollama": llm_ollama,
}


def main():
    parser = argparse.ArgumentParser(description="Run one evaluation method over the dataset")
    parser.add_argument(
        "--method",
        choices=sorted(METHODS.keys()),
        required=True,
        help="Which evaluation method to run",
    )
    parser.add_argument("-n", type=int, default=None, help="Number of samples (default: full dataset)")
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Override the ollama model to use (ignores ollama_models[0] from config_llm.json)",
    )
    args = parser.parse_args()

    train, test = load_dataset()
    data = train + test
    print(f"Dataset loaded: {len(train)} train + {len(test)} test = {len(data)} total")

    n = args.n if args.n is not None else len(data)
    print(f"\n{'='*60}")
    print(f"Method: {args.method}  |  Evaluating {n} equations")
    print(f"{'='*60}\n")

    if args.method == "ollama" and args.ollama_model:
        func = lambda equation=None: llm_ollama(equation=equation, model=args.ollama_model)
    else:
        func = METHODS[args.method]

    metrics = measurements(func, n, data)

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"Time: {metrics['time']} seconds")
    print(f"Failures: {metrics['failures']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
