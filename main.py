#!/usr/bin/env python3
"""
Main entry point for the Correlated Knowledge Jailbreak Project.

This script orchestrates the entire jailbreak evaluation pipeline including:
- Data loading and preprocessing
- Model initialization and configuration
- Baseline and proposed method execution
- Evaluation and metric computation
- Results logging and visualization

Usage:
    python main.py --config config/config.yml [--debug] [--dry-run] [--phase full|jailbreak|judge]
"""

import argparse
import logging
import os
import sys
import yaml
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from zoneinfo import ZoneInfo

# Import project modules
from data.dataloader import DataLoader
from model.model_loader import ModelLoader
from methods.method_factory import create_method
from methods.abstract_method import AbstractJailbreakMethod
from evaluation.evaluator import Evaluator
from utils.utils import setup_logging, save_results, create_output_dirs, validate_config
from utils.gpu_manager import initialize_gpu_manager, get_gpu_manager


class JailbreakExperiment:
    """
    Main experiment coordinator for jailbreak evaluation.

    This class manages the entire experimental pipeline from data loading
    to result generation and visualization.
    """

    def __init__(self, config_path: str, phase: str = "full"):
        """
        Initialize the experiment with configuration.

        Args:
            config_path (str): Path to the YAML configuration file
            phase (str): Experiment phase ('full', 'jailbreak', 'judge', 'resume')
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.phase = phase

        # Initialize components (will be set up in setup methods)
        self.dataloader = None
        self.model_loader = None
        self.target_model = None
        self.evaluator = None
        self.methods = {}
        self.results = {}

        # Initialize GPU resource manager
        self.gpu_manager = None

        # Setup output directory based on phase
        if phase == "resume" or phase == "judge":
            # For resume and judge, use the output_dir directly from config
            self.output_dir = self.config.get("experiment", {}).get(
                "output_dir", "results/outputs"
            )
            self.experiment_name = os.path.basename(self.output_dir)
        else:
            # For other phases, generate new experiment name and directory
            self.experiment_name = self._generate_experiment_name()
            self.output_dir = self._setup_output_directory()

        # Setup logging
        self.logger = self._setup_logging()

        self.logger.info(f"Initialized experiment: {self.experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dict[str, Any]: Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Validate configuration
            if not validate_config(config):
                raise ValueError("Invalid configuration")

            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    def _load_method_config(self, method_name: str) -> Dict[str, Any]:
        """
        Load method configuration from separate YAML file.

        Args:
            method_name (str): Name of the method

        Returns:
            Dict[str, Any]: Method configuration dictionary

        Raises:
            FileNotFoundError: If method config file doesn't exist
            yaml.YAMLError: If method config file is malformed
        """
        method_config_path = os.path.join(
            os.path.dirname(self.config_path), "method", f"{method_name}.yml"
        )

        try:
            with open(method_config_path, "r", encoding="utf-8") as f:
                method_config = yaml.safe_load(f)

            self.logger.info(
                f"Loaded method configuration for {method_name} from {method_config_path}"
            )
            return method_config

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Method configuration file not found: {method_config_path}"
            )
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing method configuration file {method_config_path}: {e}"
            )

    def _generate_experiment_name(self) -> str:
        """
        Generate a unique experiment name based on config and timestamp.

        Returns:
            str: Unique experiment identifier
        """
        base_name = self.config.get("experiment", {}).get(
            "name", "jailbreak_experiment"
        )
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"

    def _setup_output_directory(self) -> str:
        """
        Create output directory for this experiment.

        Returns:
            str: Path to the output directory
        """
        base_output_dir = self.config.get("experiment", {}).get(
            "output_dir", "results/outputs"
        )
        output_dir = os.path.join(base_output_dir, self.experiment_name)
        create_output_dirs(output_dir)
        return output_dir

    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.

        Returns:
            logging.Logger: Configured logger instance
        """
        log_config = self.config.get("logging", {})
        log_file = os.path.join(self.output_dir, "experiment.log")

        log_level = log_config.get("console_level", "INFO")
        return setup_logging(
            log_level=log_level, log_file=log_file, logger_name="JailbreakExperiment"
        )

    def setup_data(self) -> None:
        """
        Setup data loading and preprocessing.
        """
        self.logger.info("Setting up data loading...")

        try:
            # Initialize data loader
            data_config = self.config.get("data", {})
            self.dataloader = DataLoader(
                data_dir=data_config.get("data_dir", "data/datasets")
            )

            # Load datasets specified in config
            datasets = data_config.get("datasets", [])
            for dataset_config in datasets:
                dataset_name = dataset_config["name"]
                dataset_size = dataset_config.get("size", "large")

                self.logger.info(f"Loading dataset: {dataset_name} ({dataset_size})")
                try:
                    data = self.dataloader.load(dataset_name, dataset_size=dataset_size)
                    self.logger.info(
                        f"Successfully loaded {len(data)} samples from {dataset_name}"
                    )

                    # Log dataset statistics
                    dataset_key = (
                        f"{dataset_name}_{dataset_size}"
                        if dataset_size != "large"
                        else dataset_name
                    )
                    self._log_dataset_statistics(dataset_key, data)

                except Exception as e:
                    self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                    raise

        except Exception as e:
            self.logger.error(f"Failed to setup data loading: {e}")
            raise

    def _log_dataset_statistics(
        self, dataset_key: str, dataset: List[Dict[str, Any]]
    ) -> None:
        """
        Log statistics about the loaded dataset.

        Args:
            dataset_key (str): Dataset identifier
            dataset (List[Dict[str, Any]]): Dataset samples
        """
        if not dataset:
            return

        # Category distribution
        categories = {}
        sources = {}
        query_lengths = []

        for sample in dataset:
            category = sample.get("category", "unknown")
            source = sample.get("source", "unknown")
            query_length = len(sample.get("query", ""))

            categories[category] = categories.get(category, 0) + 1
            sources[source] = sources.get(source, 0) + 1
            query_lengths.append(query_length)

        self.logger.info(f"\n=== Dataset Statistics: {dataset_key} ===")
        self.logger.info(f"Total Samples: {len(dataset)}")

        if categories:
            self.logger.info("\nCategory Distribution:")
            for cat, count in sorted(categories.items()):
                percentage = (count / len(dataset)) * 100
                self.logger.info(f"  {cat}: {count} ({percentage:.1f}%)")

        if sources:
            self.logger.info("\nSource Distribution:")
            for src, count in sorted(sources.items()):
                percentage = (count / len(dataset)) * 100
                self.logger.info(f"  {src}: {count} ({percentage:.1f}%)")

        if query_lengths:
            avg_length = sum(query_lengths) / len(query_lengths)
            min_length = min(query_lengths)
            max_length = max(query_lengths)
            self.logger.info(f"\nQuery Length Statistics:")
            self.logger.info(f"  Average: {avg_length:.2f} characters")
            self.logger.info(f"  Minimum: {min_length} characters")
            self.logger.info(f"  Maximum: {max_length} characters")

        self.logger.info("-" * 60)

    # Original version without defense
    # def setup_model(self) -> None:
    #     """
    #     Setup target model for jailbreak evaluation.
    #     """
    #     self.logger.info("Setting up target model...")

    #     try:
    #         self.model_loader = ModelLoader(self.config.get('model', {}))
    #         self.target_model = self.model_loader.load_model()

    #         self.logger.info(f"Successfully loaded model: {self.target_model.model_name}")

    #     except Exception as e:
    #         self.logger.error(f"Failed to load model {self.target_model.model_name if self.target_model else 'unknown'}: {e}")
    #         raise

    # New version with defense
    def setup_model(self) -> None:
        """
        Setup target model for jailbreak evaluation with optional defense layer.
        """
        self.logger.info("Setting up target model...")

        try:
            # Initialize GPU resource manager
            self.gpu_manager = initialize_gpu_manager(self.logger)

            # Merge defense config into model config
            # Extract defense configuration from global config
            defense_config = self.config.get("defense", {})

            # Create a copy of model config and add defense
            model_config_with_defense = self.config.get("model", {}).copy()
            if defense_config:
                model_config_with_defense["defense"] = defense_config
                self.logger.info(
                    f"Defense enabled: {defense_config.get('enabled', False)}, type: {defense_config.get('type', 'none')}"
                )

            # Allocate GPU for target model
            target_model_name = model_config_with_defense.get("whitebox", {}).get(
                "name", "unknown"
            )
            if model_config_with_defense.get("type") == "whitebox":
                target_gpus = self.gpu_manager.allocate_target_model(target_model_name)
                self.logger.info(
                    f"Allocated GPUs {target_gpus} for target model: {target_model_name}"
                )

            # Allocate GPU for defense model if enabled (supports llm_guard, grayswanai_guard and rephrasing)
            if defense_config.get("enabled", False):
                d_type = defense_config.get("type")
                defense_model_name = None
                if d_type == "llm_guard":
                    defense_model_name = defense_config.get("llm_guard", {}).get(
                        "guard_model_name", "llm_guard"
                    )
                elif d_type == "grayswanai_guard":
                    defense_model_name = defense_config.get("grayswanai_guard", {}).get(
                        "guard_model_name", "grayswanai_guard"
                    )
                elif d_type == "rephrasing":
                    defense_model_name = defense_config.get("rephrasing", {}).get(
                        "rephrase_model_name", "rephrasing_defense"
                    )
                elif d_type == "perturbation":
                    defense_model_name = None

                if defense_model_name:
                    defense_gpus = self.gpu_manager.allocate_defense_model(
                        defense_model_name
                    )
                    if defense_gpus:
                        self.logger.info(
                            f"Allocated GPUs {defense_gpus} for defense model: {defense_model_name}"
                        )

            # Initialize ModelLoader with merged config
            self.model_loader = ModelLoader(model_config_with_defense)
            self.target_model = self.model_loader.load_model()

            self.logger.info(
                f"Successfully loaded model: {self.target_model.model_name}"
            )

            # Print GPU allocation summary
            self.gpu_manager.print_allocation_summary()

        except Exception as e:
            self.logger.error(
                f"Failed to load model {self.target_model.model_name if self.target_model else 'unknown'}: {e}"
            )
            raise

    def setup_methods(self) -> None:
        """
        Setup jailbreak methods using modular configuration.
        Methods are loaded from separate YAML files in config/method/ directory.
        """
        self.logger.info("Setting up jailbreak methods...")

        methods_config = self.config.get("methods", {})

        # Setup baseline methods
        baseline_methods = methods_config.get("baselines", [])
        for method_name in baseline_methods:
            self.logger.info(f"Initializing baseline method: {method_name}")

            try:
                # Load method configuration from separate file
                method_config = self._load_method_config(method_name)

                # Allocate GPUs for this method
                self._allocate_method_gpus(method_name, method_config)

                # Create method instance
                method = self._load_method(method_name, method_config)
                self.methods[f"baseline_{method_name}"] = method
                self.logger.info(f"Successfully initialized baseline: {method_name}")

            except Exception as e:
                self.logger.error(f"Failed to initialize baseline {method_name}: {e}")
                raise

        # Setup proposed methods
        proposed_methods = methods_config.get("proposed", [])
        for method_name in proposed_methods:
            self.logger.info(f"Initializing proposed method: {method_name}")

            try:
                # Load method configuration from separate file
                method_config = self._load_method_config(method_name)

                # Allocate GPUs for this method
                self._allocate_method_gpus(method_name, method_config)

                # Create method instance
                method = self._load_method(method_name, method_config)
                self.methods[f"proposed_{method_name}"] = method
                self.logger.info(
                    f"Successfully initialized proposed method: {method_name}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to initialize proposed method {method_name}: {e}"
                )
                raise

    def setup_evaluation(self) -> None:
        """
        Setup evaluation metrics and evaluator (delayed initialization).
        Note: Judge model will be initialized later to avoid GPU memory conflicts.
        """
        self.logger.info("Setting up evaluation framework...")

        eval_config = self.config.get("evaluation", {})
        # Don't initialize the evaluator yet - we'll do it after experiments complete
        self.eval_config = eval_config
        self.logger.info(
            "Evaluation framework configured (judge model will be initialized after experiments)"
        )

    def _setup_evaluator_with_progress(self) -> None:
        """
        Setup evaluator with progress bar for judge model initialization.
        """
        self.logger.info("Initializing judge model for evaluation...")

        try:
            # Show progress bar for judge model initialization
            with tqdm(total=100, desc="Initializing Judge Model", unit="%") as pbar:
                pbar.set_description("Loading Llama-Guard-3-8B...")
                pbar.update(10)

                # Initialize evaluator
                self.evaluator = Evaluator(
                    config=self.eval_config,
                    output_dir=os.path.join(self.output_dir, "evaluation"),
                )

                pbar.update(40)
                pbar.set_description("Judge model loaded successfully")
                pbar.update(50)

                # Final update
                pbar.set_description("Evaluation framework ready")
                pbar.update(100)

            self.logger.info("Judge model initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize judge model: {e}")
            raise

    def run_experiments(self) -> None:
        """
        Execute all jailbreak methods on loaded datasets.
        """
        self.logger.info("Starting jailbreak experiments...")

        # Get experiment configuration
        exp_config = self.config.get("experiment", {})
        max_samples = exp_config.get("max_samples_per_dataset", None)

        # Iterate through all loaded datasets
        for dataset_key in self.dataloader.get_loaded_datasets():
            self.logger.info(f"Running experiments on dataset: {dataset_key}")

            # Get dataset samples
            dataset = self.dataloader.get_dataset(dataset_key)
            if max_samples:
                dataset = dataset[:max_samples]
                self.logger.info(f"Limited to {max_samples} samples")

            # Initialize results for this dataset
            if dataset_key not in self.results:
                self.results[dataset_key] = {}

            # Run each method on this dataset
            for method_name, method in self.methods.items():
                self.logger.info(f"Running method: {method_name} on {dataset_key}")

                try:
                    method_results = self._run_method_on_dataset(
                        method, dataset, dataset_key
                    )
                    self.results[dataset_key][method_name] = method_results

                    self.logger.info(f"Completed {method_name} on {dataset_key}")

                except Exception as e:
                    self.logger.error(
                        f"Error running {method_name} on {dataset_key}: {e}"
                    )
                    self.results[dataset_key][method_name] = {"error": str(e)}

        self.logger.info("All experiments completed")

    def load_existing_results(self, output_dir: str) -> Dict[str, Any]:
        """
        Load existing results from a previous run for resume functionality.

        Args:
            output_dir: Directory containing previous results

        Returns:
            Dict containing loaded results and metadata
        """
        self.logger.info(f"Loading existing results from: {output_dir}")

        # Load pending_judge.jsonl to get completed queries
        pending_file = os.path.join(output_dir, "pending_judge.jsonl")
        completed_queries = []
        completed_count = 0

        if os.path.exists(pending_file):
            with open(pending_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            if "result" in data and "original_query" in data["result"]:
                                completed_queries.append(
                                    data["result"]["original_query"]
                                )
                                completed_count += 1
                        except json.JSONDecodeError:
                            continue

        self.logger.info(
            f"Found {completed_count} completed queries in existing results"
        )

        # Determine next index based on pending_result count
        # If autodan has 34 pending_results (indices 0-33), next index should be 34
        next_index = completed_count
        self.logger.info(
            f"Completed queries count: {completed_count}, next index will start from: {next_index}"
        )

        return {
            "completed_queries": completed_queries,
            "completed_count": completed_count,
            "next_index": next_index,
            "pending_file": pending_file,
        }

    def run_resume_pipeline(self) -> None:
        """
        Resume interrupted jailbreak experiments.
        Loads existing results and continues from where it left off.
        """
        self.logger.info("=== Starting Resume Pipeline ===")

        try:
            # Validate output_dir exists and contains previous results
            if not os.path.exists(self.output_dir):
                raise ValueError(f"Output directory does not exist: {self.output_dir}")

            # Load existing results
            existing_data = self.load_existing_results(self.output_dir)
            completed_queries = existing_data["completed_queries"]
            completed_count = existing_data["completed_count"]
            next_index = existing_data["next_index"]
            pending_file = existing_data["pending_file"]

            # Setup data, model, and methods
            self.setup_data()
            self.setup_model()
            self.setup_methods()
            self.setup_evaluation()

            # Get experiment configuration
            exp_config = self.config.get("experiment", {})
            max_samples = exp_config.get("max_samples_per_dataset", None)

            # Iterate through all loaded datasets
            for dataset_key in self.dataloader.get_loaded_datasets():
                self.logger.info(f"Resuming experiments on dataset: {dataset_key}")

                # Get dataset samples
                dataset = self.dataloader.get_dataset(dataset_key)
                if max_samples:
                    dataset = dataset[:max_samples]
                    self.logger.info(f"Limited to {max_samples} samples")

                # Filter out completed queries
                remaining_queries = []
                remaining_indices = []

                for i, sample in enumerate(dataset):
                    query = sample.get("query", "")
                    if query not in completed_queries:
                        remaining_queries.append(sample)
                        remaining_indices.append(i)

                self.logger.info(
                    f"Dataset {dataset_key}: {len(dataset)} total, {completed_count} completed, {len(remaining_queries)} remaining"
                )

                if not remaining_queries:
                    self.logger.info(
                        f"All queries in dataset {dataset_key} are already completed"
                    )
                    continue

                # Initialize results for this dataset
                if dataset_key not in self.results:
                    self.results[dataset_key] = {}

                # Run each method on remaining queries
                for method_name, method in self.methods.items():
                    self.logger.info(
                        f"Resuming method: {method_name} on {dataset_key} ({len(remaining_queries)} remaining queries)"
                    )

                    try:
                        # Set the starting index for intermediate results
                        method._thread_local = type("ThreadLocal", (), {})()
                        method._thread_local.sample_index = next_index
                        method._thread_local.dataset_key = dataset_key

                        # Run method on remaining queries
                        method_results = self._run_method_on_dataset_resume(
                            method,
                            remaining_queries,
                            remaining_indices,
                            dataset_key,
                            next_index,
                            pending_file,
                        )

                        # Update results
                        if dataset_key not in self.results:
                            self.results[dataset_key] = {}
                        if method_name not in self.results[dataset_key]:
                            self.results[dataset_key][method_name] = []

                        self.results[dataset_key][method_name].extend(method_results)

                        # Update next_index for next method
                        next_index += len(remaining_queries)

                        self.logger.info(
                            f"Completed resume for {method_name} on {dataset_key}"
                        )

                    except Exception as e:
                        self.logger.error(
                            f"Error resuming {method_name} on {dataset_key}: {e}"
                        )
                        if dataset_key not in self.results:
                            self.results[dataset_key] = {}
                        self.results[dataset_key][method_name] = {"error": str(e)}

            # Free whitebox resources before judge init
            self._teardown_whitebox_resources()

            # Evaluation phase (judge model initialized here)
            self.evaluate_results()
            self.generate_reports()

            self.logger.info("=== Resume Pipeline completed successfully ===")
            print(f"Resume completed! Results saved to: {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Resume pipeline failed with error: {e}")
            raise

    def _run_method_on_dataset_resume(
        self,
        method: AbstractJailbreakMethod,
        queries: List[Dict],
        original_indices: List[int],
        dataset_key: str,
        start_index: int,
        pending_file: str,
    ) -> List[Dict[str, Any]]:
        """
        Run a method on remaining queries for resume functionality.

        Args:
            method: Jailbreak method instance
            queries: Remaining queries to process
            original_indices: Original indices of the queries in the full dataset
            dataset_key: Dataset identifier
            start_index: Starting index for intermediate results
            pending_file: Path to pending_judge.jsonl file

        Returns:
            List of result dictionaries
        """
        results = []

        # Extract query strings
        query_strings = [q.get("query", "") for q in queries]

        self.logger.info(
            f"Processing {len(query_strings)} remaining queries starting from index {start_index}"
        )

        # Process queries with progress bar
        from tqdm import tqdm

        with tqdm(total=len(query_strings), desc=f"{method.name}") as pbar:
            for idx, (query_data, original_idx) in enumerate(
                zip(queries, original_indices)
            ):
                current_index = start_index + idx

                try:
                    method._thread_local.sample_index = current_index
                    method._thread_local.dataset_key = dataset_key

                    # Extract query string and metadata
                    query = query_data.get("query", "")
                    category = query_data.get("category", "unknown")
                    source = query_data.get("source", "unknown")

                    # Generate jailbreak with metadata
                    result = method.generate_jailbreak(
                        query, category=category, source=source
                    )
                    results.append(result)

                    # Note: Intermediate results are saved by the method itself during execution
                    # We only save the final result to pending_judge.jsonl

                    # Append to pending_judge.jsonl
                    pending_entry = {
                        "dataset_key": dataset_key,
                        "method_name": method.name,
                        "result": result,
                    }

                    with open(pending_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(pending_entry, ensure_ascii=False) + "\n")

                    pbar.update(1)

                except Exception as e:
                    self.logger.error(f"Error processing query {current_index}: {e}")
                    error_result = {
                        "original_query": query,
                        "error": str(e),
                        "success": False,
                        "metadata": {
                            "method": method.name,
                            "category": category,
                            "source": source,
                            "error": str(e),
                        },
                    }
                    results.append(error_result)
                    pbar.update(1)

        return results

    def evaluate_results(self) -> None:
        """
        Evaluate all experimental results using configured metrics.
        Judge model is initialized here to avoid GPU memory conflicts.
        """
        self.logger.info("Evaluating experimental results...")

        # Initialize evaluator with progress bar
        self._setup_evaluator_with_progress()

        if not self.evaluator or not self.evaluator.get_available_metrics():
            self.logger.error(
                "No evaluator or no metrics available; skipping evaluation."
            )
            return

        for dataset_key, dataset_results in self.results.items():
            self.logger.info(f"Evaluating results for dataset: {dataset_key}")

            try:
                evaluation_results = self.evaluator.evaluate_dataset_results(
                    dataset_key, dataset_results
                )
                self.results[dataset_key]["evaluation"] = evaluation_results
                self.logger.info(f"Evaluation completed for {dataset_key}")

                # ---- pretty summary in logs ----
                for method_name, metrics_out in evaluation_results.items():
                    if isinstance(metrics_out, dict):
                        for metric_name, res in metrics_out.items():
                            self.logger.info(
                                f"[SUMMARY] dataset={dataset_key} method={method_name} metric={metric_name} -> {res}"
                            )

            except Exception as e:
                self.logger.error(f"Error evaluating {dataset_key}: {e}")

        self.logger.info("Evaluation completed for all datasets")

    def generate_reports(self) -> None:
        """
        Generate final reports and visualizations.
        """
        self.logger.info("Generating reports and visualizations...")

        try:
            # Save raw results
            results_file = os.path.join(self.output_dir, "results.json")
            save_results(self.results, results_file)
            self.logger.info(f"Raw results saved to: {results_file}")

            # Generate summary report
            self._generate_summary_report()

            # Generate visualizations
            self._generate_visualizations()

            self.logger.info("Report generation completed")

        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            raise

    def _teardown_whitebox_resources(self) -> None:
        """
        Release whitebox model resources to free GPU memory before judge init.
        Only used in full pipeline between jailbreak and judge phases.
        """
        try:
            self.logger.info("Releasing whitebox resources to free GPU memory...")
            # Release method-level resources (custom teardown if provided)
            import gc
            import torch

            if hasattr(self, "methods") and isinstance(self.methods, dict):
                for m_name, m in list(self.methods.items()):
                    # Call method-specific teardown if available
                    if hasattr(m, "teardown") and callable(getattr(m, "teardown")):
                        try:
                            m.teardown()
                        except Exception as e:
                            self.logger.warning(
                                f"Method {m_name} teardown warning: {e}"
                            )
                    for attr in [
                        "whitebox_model",
                        "whitebox_tokenizer",
                        "conv_template",
                        "criterion",
                    ]:
                        if hasattr(m, attr):
                            setattr(m, attr, None)
                # Optional: drop methods if not needed anymore
                # self.methods = {}

            # Release target model (whitebox holder)
            if hasattr(self, "target_model") and self.target_model is not None:
                for attr in [
                    "vllm_model",  # vLLM engine instance
                    "model",  # HF model
                    "tokenizer",
                ]:
                    if hasattr(self.target_model, attr):
                        setattr(self.target_model, attr, None)
                # Optional: drop the wrapper to let GC reclaim
                # self.target_model = None

            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.logger.info("Whitebox resources released.")
        except Exception as e:
            self.logger.warning(f"Failed to fully release whitebox resources: {e}")

    def _allocate_method_gpus(
        self, method_name: str, method_config: Dict[str, Any]
    ) -> None:
        """
        Allocate GPUs for a specific jailbreak method.

        Args:
            method_name (str): Name of the method
            method_config (Dict[str, Any]): Method configuration
        """
        if not self.gpu_manager:
            self.logger.warning("GPU manager not initialized, skipping GPU allocation")
            return

        # Determine method requirements based on method name
        needs_controller = False
        needs_judge = False
        controller_model = ""
        judge_model = ""

        # Get actual method config (handle nested config structure)
        actual_config = method_config.get("config", method_config)

        if method_name == "cka-agent":
            # CKA-Agent: uses controller_model and judge_model
            controller_config = actual_config.get("controller_model", {})
            if controller_config.get("name"):
                needs_controller = True
                controller_model = controller_config["name"]

            judge_config = actual_config.get("judge_model", {})
            if judge_config.get("type") == "whitebox" and judge_config.get(
                "whitebox", {}
            ).get("name"):
                needs_judge = True
                judge_model = judge_config["whitebox"]["name"]
            elif judge_config.get("type") == "blackbox" and judge_config.get(
                "blackbox", {}
            ).get("name"):
                needs_judge = True
                judge_model = judge_config["blackbox"]["name"]

        elif method_name in [
            "agent_self_response",
            "multi_agent_jailbreak",
            "x_teaming",
        ]:
            # These methods use attack_model directly (no whitebox/blackbox distinction)
            attack_config = actual_config.get("attack_model", {})
            if attack_config.get("name"):
                needs_controller = True
                controller_model = attack_config["name"]

        elif method_name in ["pap", "pair", "actor_attack"]:
            # These methods have attack_model with whitebox/blackbox distinction
            attack_config = actual_config.get("attack_model", {})
            if attack_config.get("type") == "whitebox" and attack_config.get(
                "whitebox", {}
            ).get("name"):
                needs_controller = True
                controller_model = attack_config["whitebox"]["name"]
            elif attack_config.get("type") == "blackbox" and attack_config.get(
                "blackbox", {}
            ).get("name"):
                # Blackbox doesn't need GPU allocation
                pass
        elif method_name == "parley":
            # Parley uses attacker_model (not attack_model)
            attacker_config = actual_config.get("attacker_model", {})
            if attacker_config.get("type") == "whitebox" and attacker_config.get(
                "whitebox", {}
            ).get("name"):
                needs_controller = True
                controller_model = attacker_config["whitebox"]["name"]
            elif attacker_config.get("type") == "blackbox":
                # Blackbox doesn't need GPU allocation
                pass

        elif method_name in ["vanilla", "autodan"]:
            # These methods don't need GPU allocation
            pass

        else:
            # Fallback: try to detect any attack model
            attack_config = actual_config.get("attack_model", {})
            if attack_config.get("name"):
                needs_controller = True
                controller_model = attack_config["name"]

        # Allocate GPUs for this method
        allocations = self.gpu_manager.allocate_attack_method(
            method_name=method_name,
            needs_controller=needs_controller,
            needs_judge=needs_judge,
            controller_model=controller_model,
            judge_model=judge_model,
        )

        if allocations:
            self.logger.info(f"GPU allocations for {method_name}: {allocations}")
        else:
            self.logger.warning(f"No GPU allocations available for {method_name}")

    def _load_method(
        self, method_name: str, method_config: Dict[str, Any]
    ) -> AbstractJailbreakMethod:
        """
        Load a specific jailbreak method using the method factory.

        Args:
            method_name (str): Name of the method to load
            method_config (Dict[str, Any]): Method configuration

        Returns:
            AbstractJailbreakMethod: Initialized method instance
        """
        try:
            method_cfg = method_config.get("config", None)
            if (
                method_cfg is None
                or not isinstance(method_cfg, dict)
                or len(method_cfg) == 0
            ):
                method_cfg = method_config
            method = create_method(
                method_name=method_name, config=method_cfg, model=self.target_model
            )
            # Provide output directory to methods so they can write intermediates next to pending_judge.jsonl
            try:
                setattr(method, "output_dir", self.output_dir)
            except Exception:
                pass

            self.logger.info(f"Successfully loaded method: {method_name}")
            return method

        except Exception as e:
            self.logger.error(f"Failed to load method {method_name}: {e}")
            raise

    def _run_method_on_dataset(
        self,
        method: AbstractJailbreakMethod,
        dataset: List[Dict[str, Any]],
        dataset_key: str,
    ) -> Dict[str, Any]:
        """
        Run a specific method on a dataset.

        Args:
            method (AbstractJailbreakMethod): The jailbreak method to run
            dataset (List[Dict[str, Any]]): Dataset samples

        Returns:
            Dict[str, Any]: Method execution results
        """
        self.logger.info(f"Running {method.name} on {len(dataset)} samples")

        results = []
        start_time = time.time()

        # Extract queries and metadata from dataset
        queries = [sample["query"] for sample in dataset]
        categories = [sample.get("category", "unknown") for sample in dataset]
        sources = [sample.get("source", "unknown") for sample in dataset]
        target_strs = [sample.get("target_str", "") for sample in dataset]

        # Get batch size from model configuration (dataset batch processing)
        batch_size = 1
        if hasattr(self.target_model, "batch_size"):
            batch_size = self.target_model.batch_size

        # Prepare pending path and lock for realtime append
        try:
            import threading

            if not hasattr(self, "_pending_file_lock"):
                self._pending_file_lock = threading.Lock()
        except Exception:
            self._pending_file_lock = None
        pending_path = os.path.join(self.output_dir, "pending_judge.jsonl")

        def _append_pending_item(item_result: Dict[str, Any]):
            try:
                line = {
                    "dataset_key": dataset_key,
                    "method_name": method.name,
                    "result": item_result,
                }
                pending_dir = os.path.dirname(pending_path)
                if pending_dir:
                    os.makedirs(pending_dir, exist_ok=True)
                # Use lock if available for thread-safety
                if getattr(self, "_pending_file_lock", None) is not None:
                    with self._pending_file_lock:
                        with open(pending_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(line, ensure_ascii=False) + "\n")
                            f.flush()
                else:
                    with open(pending_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                        f.flush()
            except Exception as e:
                try:
                    self.logger.debug(f"Realtime pending append failed: {e}")
                except Exception:
                    pass

        # Run method on all samples
        try:
            if hasattr(method, "generate_jailbreak_batch"):
                # Use batch processing if available
                if batch_size > 1:
                    self.logger.info(
                        f"Processing {len(queries)} queries in dataset batches of {batch_size}"
                    )
                    for i in range(0, len(queries), batch_size):
                        batch_queries = queries[i : i + batch_size]
                        batch_categories = categories[i : i + batch_size]
                        batch_sources = sources[i : i + batch_size]
                        batch_target_strs = target_strs[i : i + batch_size]

                        # Attach dataset context for intermediate saves using thread-local storage
                        try:
                            method._thread_local.sample_index = i
                            method._thread_local.dataset_key = dataset_key
                        except Exception:
                            pass
                        batch_results = method.generate_jailbreak_batch(
                            batch_queries,
                            category=batch_categories,
                            source=batch_sources,
                            target_str=batch_target_strs,
                            dataset_key=dataset_key,
                            base_index=i,
                        )
                        # Realtime append per item
                        for r in batch_results:
                            _append_pending_item(r)
                        results.extend(batch_results)
                else:
                    # Process samples individually to enable per-item realtime append
                    for idx, (q, c, s, ts) in enumerate(
                        tqdm(
                            list(zip(queries, categories, sources, target_strs)),
                            total=len(queries),
                            desc=f"Running {method.name}",
                            unit="sample",
                        )
                    ):
                        try:
                            method._thread_local.sample_index = idx
                            method._thread_local.dataset_key = dataset_key
                        except Exception:
                            pass
                        batch_results = method.generate_jailbreak_batch(
                            [q],
                            category=[c],
                            source=[s],
                            target_str=[ts],
                            dataset_key=dataset_key,
                            base_index=idx,
                        )
                        # Expect 1 result
                        if batch_results:
                            _append_pending_item(batch_results[0])
                            results.append(batch_results[0])
            else:
                # Process samples individually with progress bar
                for idx, sample in enumerate(
                    tqdm(dataset, desc=f"Running {method.name}", unit="sample")
                ):
                    try:
                        method._thread_local.sample_index = idx
                        method._thread_local.dataset_key = dataset_key
                    except Exception:
                        pass
                    result = method.generate_jailbreak(
                        query=sample["query"],
                        category=sample.get("category", "unknown"),
                        source=sample.get("source", "unknown"),
                        target_str=sample.get("target_str", ""),
                    )
                    _append_pending_item(result)
                    results.append(result)

        except Exception as e:
            self.logger.error(f"Error running method {method.name}: {e}")
            return {
                "error": str(e),
                "method": method.name,
                "num_samples": len(dataset),
                "processing_time": time.time() - start_time,
            }

        # Compile results
        processing_time = time.time() - start_time
        success_count = sum(1 for r in results if r.get("success", False))
        error_count = sum(1 for r in results if "error" in r)

        method_results = {
            "method": method.name,
            "num_samples": len(dataset),
            "results": results,
            "processing_time": processing_time,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(results) if results else 0.0,
            "stats": method.get_stats(),
        }

        self.logger.info(
            f"Completed {method.name}: {success_count}/{len(results)} successful "
            f"({method_results['success_rate']:.2%}) in {processing_time:.2f}s"
        )

        return method_results

    def _generate_summary_report(self) -> None:
        """
        Generate a summary report of all experimental results.
        """
        # TODO: Implement summary report generation
        self.logger.info("Generating summary report (placeholder)")

    def _generate_visualizations(self) -> None:
        """
        Generate visualizations for experimental results.
        """
        # TODO: Implement visualization generation
        self.logger.info("Generating visualizations (placeholder)")

    # -----------------------------
    # Pending judge IO helpers
    # -----------------------------
    def save_pending_for_judge(self, pending_path: Optional[str] = None) -> str:
        """Save items to be judged into a JSONL file for later evaluation.

        Each line contains: {
            'dataset_key', 'method_name', 'result'
        }
        where 'result' mirrors a single item from method_results['results'].
        """
        if pending_path is None:
            pending_path = os.path.join(self.output_dir, "pending_judge.jsonl")

        # Ensure directory exists
        pending_dir = os.path.dirname(pending_path)
        if pending_dir:
            os.makedirs(pending_dir, exist_ok=True)

        count = 0
        with open(pending_path, "w", encoding="utf-8") as f:
            for dataset_key, dataset_results in self.results.items():
                for method_name, payload in dataset_results.items():
                    if method_name == "evaluation":
                        continue
                    results = (
                        payload.get("results", []) if isinstance(payload, dict) else []
                    )
                    for r in results:
                        line = {
                            "dataset_key": dataset_key,
                            "method_name": method_name,
                            "result": r,
                        }
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                        count += 1
        self.logger.info(f"Saved {count} pending items for judge to: {pending_path}")
        return pending_path

    def load_pending_results(self, pending_path: str) -> None:
        """Load pending judge items from JSONL and reconstruct self.results structure."""
        self.results = {}

        if not os.path.exists(pending_path):
            raise FileNotFoundError(f"Pending file not found: {pending_path}")

        with open(pending_path, "r", encoding="utf-8") as f:
            total_lines = 0
            loaded_count = 0
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    dataset_key = item["dataset_key"]
                    method_name = item["method_name"]
                    result = item["result"]

                    if dataset_key not in self.results:
                        self.results[dataset_key] = {}
                    if method_name not in self.results[dataset_key]:
                        self.results[dataset_key][method_name] = {
                            "method": method_name,
                            "num_samples": 0,
                            "results": [],
                            "processing_time": 0.0,
                            "success_count": 0,
                            "error_count": 0,
                            "success_rate": 0.0,
                            "stats": {},
                        }
                    self.results[dataset_key][method_name]["results"].append(result)
                    loaded_count += 1
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error on line {line_num}: {e}")
                    raise
                except KeyError as e:
                    self.logger.error(f"Missing key on line {line_num}: {e}")
                    raise

        # Post-update counts
        for dataset_key, dataset_results in self.results.items():
            for method_name, payload in dataset_results.items():
                results = payload.get("results", [])
                payload["num_samples"] = len(results)
                payload["success_count"] = sum(
                    1 for r in results if r.get("success", False)
                )
                payload["error_count"] = sum(1 for r in results if "error" in r)
                payload["success_rate"] = (
                    payload["success_count"] / len(results) if results else 0.0
                )

        self.logger.info(f"Loaded pending items from: {pending_path}")

    def run_full_pipeline(self) -> None:
        """
        Execute the complete experimental pipeline.
        Judge model is initialized after experiments to avoid GPU memory conflicts.
        """
        self.logger.info("=== Starting Full Jailbreak Evaluation Pipeline ===")

        try:
            # Setup phase
            self.setup_data()
            self.setup_model()
            self.setup_methods()
            self.setup_evaluation()  # Only configures, doesn't initialize judge model

            # Execution phase
            self.run_experiments()

            # Save pending for judge (for potential re-evaluation)
            self.save_pending_for_judge()

            # Free whitebox resources before judge init (only in full pipeline)
            self._teardown_whitebox_resources()

            # Evaluation phase (judge model initialized here)
            self.evaluate_results()
            self.generate_reports()

            self.logger.info("=== Pipeline completed successfully ===")
            print(f"Experiment completed! Results saved to: {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Correlated Knowledge Jailbreak Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yml",
        help="Path to configuration file (default: config/config.yml)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing methods",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--phase",
        choices=["full", "jailbreak", "judge", "resume"],
        default="full",
        help="Choose which phase to run: full | jailbreak | judge | resume",
    )
    parser.add_argument(
        "--pending-file",
        type=str,
        default=None,
        help="Path to pending judge JSONL file (defaults to <output_dir>/pending_judge.jsonl)",
    )

    return parser.parse_args()


def main():
    """
    Main entry point of the program.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Basic validation
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        # Initialize and run experiment
        experiment = JailbreakExperiment(config_path=args.config, phase=args.phase)

        # TODO: Handle dry-run and debug flags
        if args.dry_run:
            experiment.logger.info("Dry run mode - methods will not be executed")

        if args.debug:
            experiment.logger.info("Debug mode enabled")

        # Route by phase
        phase = args.phase
        pending_path = args.pending_file

        if phase == "full":
            experiment.run_full_pipeline()
        elif phase == "jailbreak":
            experiment.logger.info("=== Starting Jailbreak-Only Pipeline ===")
            experiment.setup_data()
            experiment.setup_model()
            experiment.setup_methods()
            experiment.run_experiments()
            # Save pending then raw results
            out_path = experiment.save_pending_for_judge(pending_path)
            results_file = os.path.join(experiment.output_dir, "results.json")
            save_results(experiment.results, results_file)
            experiment.logger.info(
                f"Jailbreak-only completed. Pending saved to {out_path}"
            )
            print(f"Jailbreak-only completed. Outputs: {experiment.output_dir}")
        elif phase == "resume":
            experiment.logger.info("=== Starting Resume Pipeline ===")
            experiment.run_resume_pipeline()
        elif phase == "judge":
            experiment.logger.info("=== Starting Judge-Only Pipeline ===")
            # Only evaluation setup
            experiment.setup_evaluation()
            # Determine default pending path if not provided
            if pending_path is None:
                # Use the output_dir directly from config (not subdirectories)
                base_output_dir = experiment.config.get("experiment", {}).get(
                    "output_dir", "results/outputs"
                )
                pending_path = os.path.join(base_output_dir, "pending_judge.jsonl")
                experiment.logger.info(
                    f"Looking for pending_judge.jsonl in: {pending_path}"
                )

                # Check if the file exists
                if not os.path.exists(pending_path):
                    experiment.logger.error(
                        f"pending_judge.jsonl not found at: {pending_path}"
                    )
                    experiment.logger.error(
                        "Please ensure the file exists in the output directory specified in config.yml"
                    )
                    sys.exit(1)

                # Use the base output directory as the experiment output directory
                experiment.output_dir = base_output_dir
                experiment.logger.info(
                    f"Using output directory: {experiment.output_dir}"
                )

            experiment.load_pending_results(pending_path)
            experiment.evaluate_results()
            experiment.generate_reports()
            experiment.logger.info("Judge-only completed.")
            print(f"Judge-only completed. Outputs: {experiment.output_dir}")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
