import os
import tempfile
from typing import Sequence, List, Tuple
import json
import dataclasses
from datetime import datetime, timedelta
import concurrent.futures

import numpy as np

from .contract.ethereum_address import ZERO_ADDRESS
from .model import ModelFactory, OthersWork

MAX_WORKERS = 4

@dataclasses.dataclass(frozen=True)
class FakeContractParameters:
    num_dp_rounds: int
    bond_reserve_fraction: float
    refund_fraction: float

def train_client(x) -> None:
    client, dp_round, tempdir, model_factory = x
    print(f"Training client {client}")
    model_factory.training_model.train(dp_round)
    model_path = os.path.join(tempdir, f"client_{client}_model.dat")
    model_factory.training_model.save(model_path)
    print(f"Finished training client {client} to {model_path}")

def score_model(x) -> None:
    client, auditor, tempdir, auditor_model_factory, score_matrix = x
    model_path = os.path.join(tempdir, f"client_{client}_model.dat")
    score = auditor_model_factory.load(model_path).score()
    score_matrix[auditor, client] = score

def compute_score_matrix(model_factories: Sequence[ModelFactory], dp_round: int, ground_truth_evaluator: ModelFactory) -> Tuple[np.ndarray, np.ndarray]:
    with tempfile.TemporaryDirectory() as tempdir, concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(train_client, [(client, dp_round, tempdir, model_factory) for client, model_factory in enumerate(model_factories)])
        for result in results:
            pass

        num_clients = len(model_factories)
        score_matrix = np.zeros((num_clients, num_clients))
        score_model_inputs = []
        for auditor, auditor_model_factory in enumerate(model_factories):
            print(f"Auditor {auditor} is scoring")
            for client, _ in enumerate(model_factories):
                score_model_inputs.append((client, auditor, tempdir, auditor_model_factory, score_matrix))
        result = executor.map(score_model, score_model_inputs)
        for x in result:
            pass  # iterate over to wait

        ground_truth_scores = np.zeros((num_clients, ))
        for client, _ in enumerate(model_factories):
            model_path = os.path.join(tempdir, f"client_{client}_model.dat")
            score = ground_truth_evaluator.load(model_path).score()
            ground_truth_scores[client] = score
    return (score_matrix, ground_truth_scores)

def compute_other_scores(score_matrix: np.ndarray, dp_round: int, reward_pool: np.ndarray, contract_parameters: FakeContractParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median_model_scores = np.quantile(score_matrix, q=0.5, axis=0, interpolation='lower', keepdims=True)
    num_clients = len(score_matrix)
    assert median_model_scores.shape == (1, num_clients)
    auditor_abs_diff = np.abs(score_matrix - median_model_scores)
    assert auditor_abs_diff.shape == (num_clients, num_clients)
    scaled_auditor_abs_diff = (0.5 - auditor_abs_diff)/(0.5 + auditor_abs_diff)
    scaled_auditor_abs_diff[scaled_auditor_abs_diff < 0.0] = 0.0
    assert scaled_auditor_abs_diff.shape == (num_clients, num_clients)
    auditor_score_prescale = np.min(scaled_auditor_abs_diff, axis=1, keepdims=True)
    assert auditor_score_prescale.shape == (num_clients, 1)
    auditor_scores = auditor_score_prescale / np.max(auditor_score_prescale)
    model_scores = median_model_scores / np.max(median_model_scores)
    assert model_scores.shape == (1, num_clients)
    overall_scores = np.min(np.vstack([model_scores, np.transpose(auditor_scores)]), axis=0, keepdims=True)
    assert overall_scores.shape == (1, num_clients)
    reward_shares = overall_scores / np.sum(overall_scores)
    assert reward_shares.shape == (1, num_clients)
    if dp_round < contract_parameters.num_dp_rounds:
        refund_amount = contract_parameters.refund_fraction * (reward_pool - num_clients * contract_parameters.bond_reserve_fraction) * reward_shares
    else:
        refund_amount = reward_pool * reward_shares
    reward_pool -= np.sum(refund_amount)
    return median_model_scores, auditor_score_prescale, overall_scores, refund_amount

def save(results_folder: str, dp_round: int, score_matrix: np.ndarray, ground_truth_scores: np.ndarray, median_model_scores: np.ndarray, auditor_score_prescale: np.ndarray, overall_scores: np.ndarray, refund_amount: np.ndarray) -> None:
    np.savetxt(os.path.join(results_folder, f"score_matrix_dp_round_{dp_round}.csv"), score_matrix, delimiter=",")
    np.savetxt(os.path.join(results_folder, f"ground_truth_dp_round_{dp_round}.csv"), ground_truth_scores, delimiter=",")
    np.savetxt(os.path.join(results_folder, f"model_scores_dp_round_{dp_round}.csv"), median_model_scores, delimiter=",")
    np.savetxt(os.path.join(results_folder, f"auditor_scores_dp_round_{dp_round}.csv"), auditor_score_prescale, delimiter=",")
    np.savetxt(os.path.join(results_folder, f"overall_scores_dp_round_{dp_round}.csv"), overall_scores, delimiter=",")
    np.savetxt(os.path.join(results_folder, f"refund_amounts_{dp_round}.csv"), refund_amount, delimiter=",")

    np.savez(os.path.join(results_folder, f"data_dp_round_{dp_round}.npz"),
        score_matrix=score_matrix,
        model_scores=median_model_scores,
        auditor_score=auditor_score_prescale,
        overall_score=overall_scores,
        refund_amount=refund_amount,
        ground_truth_scores=ground_truth_scores
    )

def run_simulation(model_factories: Sequence[ModelFactory], contract_parameters: FakeContractParameters, results_folder: str, ground_truth_evaluator: ModelFactory) -> None:
    with open(os.path.join(results_folder, "contract_parameters.json"), "w+") as f:
        json.dump(dataclasses.asdict(contract_parameters), f)
    num_clients = len(model_factories)
    client_balances = np.zeros((num_clients, ))
    client_balances[...] = -1.0
    reward_pool = np.array(float(num_clients))
    for dp_round in range(1, contract_parameters.num_dp_rounds + 1):
        print(f"starting dp round {dp_round}")
        score_matrix, ground_truth_scores = compute_score_matrix(model_factories, dp_round, ground_truth_evaluator)
        median_model_scores, auditor_score_prescale, overall_scores, refund_amount = compute_other_scores(score_matrix, dp_round, reward_pool, contract_parameters)
        others_work: List[OthersWork] = []
        squeezed_scores = np.squeeze(overall_scores).tolist()
        if isinstance(squeezed_scores, float):
            squeezed_scores = [squeezed_scores]
        for model_factory, score in zip(model_factories, squeezed_scores):
            others_work.append(OthersWork(client=ZERO_ADDRESS, model=model_factory.training_model, score=score))
        for model_factory in model_factories:
            model_factory.training_model.update(dp_round, others_work)
        save(results_folder, dp_round, score_matrix=score_matrix, ground_truth_scores=ground_truth_scores, median_model_scores=median_model_scores, auditor_score_prescale=auditor_score_prescale, overall_scores=overall_scores, refund_amount=refund_amount)
