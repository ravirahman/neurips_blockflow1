import os
import argparse
import warnings
from typing import List
import json
import shutil

from blockflow.simulator import run_simulation, FakeContractParameters

from ..config import RESULTS_FOLDER_PATH
from ..dp.dp_base import DPBase
from ..dp.laplace_output import LaplaceOutputDP
from .model import LogisticRegressionModelFactory, LogisticRegressionEvaluationModelParams, LogisticRegressionTrainingModelParams
from .adult.dataset import ADULT_OUTPUT_CLASSES, ADULT_NUM_COEFS
from .kdd99.dataset import KDD99_OUTPUT_CLASSES, KDD99_NUM_COEFS
from .dataset import LogisticRegressionDataset

warnings.filterwarnings('ignore')

def simulate_clients(*, experiment_name: str, dataset_parent_folder: str, apply_dp: bool, inv_lambda: float, epsilon: float, ground_truth_dataset_path: str, num_dp_rounds: int) -> None:
    client_dataset_folders = os.listdir(dataset_parent_folder)
    client_dataset_folders.sort(key=lambda x: int(x.split("_")[-1]))  # sort by the # as an int, not string 

    num_clients = len(client_dataset_folders)
    run_data_folder = os.path.basename(dataset_parent_folder)

    if experiment_name == "kdd99":
        output_classes = KDD99_OUTPUT_CLASSES
        num_coefs = KDD99_NUM_COEFS
    elif experiment_name == "adult":
        output_classes = ADULT_OUTPUT_CLASSES
        num_coefs = ADULT_NUM_COEFS
    else:
        raise Exception("unknown experiment")

    safe_name = os.path.join(run_data_folder.replace("/", "_").replace("\\", "_"), "inv_lambda-" + str(inv_lambda) + "_epsilon-" + str(epsilon) + "_apply_dp-" + str(apply_dp))
    run_folder = os.path.join(RESULTS_FOLDER_PATH, experiment_name, safe_name)
    os.makedirs(run_folder, exist_ok=True)

    with open(os.path.join(run_folder, "dataset_params.json"), "w+") as f:
        json.dump({
            "experiment_name": experiment_name,
            "dataset_parent_folder": dataset_parent_folder,
            "apply_dp": apply_dp,
            "inv_lambda": inv_lambda,
            "epsilon": epsilon,
            "ground_truth_dataset": ground_truth_dataset_path
        }, f)

    contract_parameters = FakeContractParameters(
        num_dp_rounds=num_dp_rounds,
        bond_reserve_fraction=0.25,
        refund_fraction=0.5)

    ground_truth_dataset = LogisticRegressionDataset.load(ground_truth_dataset_path)

    gr_training_model_parameters = LogisticRegressionTrainingModelParams(
        training_dataset=ground_truth_dataset,
        evaluation_params=LogisticRegressionEvaluationModelParams(
            output_classes=output_classes,
            num_coefs=num_coefs,
            validation_dataset=ground_truth_dataset,
            sklearn_lr_kwargs={"C": inv_lambda}
        ),
        max_num_epochs=20)
    gt_score_model_params = LogisticRegressionEvaluationModelParams(
        output_classes=output_classes,
        num_coefs=num_coefs,
        validation_dataset=ground_truth_dataset,
        sklearn_lr_kwargs={"C": inv_lambda}
    )
    ground_truth_model_factory = LogisticRegressionModelFactory(gr_training_model_parameters, gt_score_model_params)

    g = 1.0  # g = 1.0 for logistic regressions
    model_factories: List[LogisticRegressionModelFactory] = []
    for i, client_folder in enumerate(client_dataset_folders):
        training_dataset_path = os.path.join(dataset_parent_folder, client_folder, "train.dat")
        validation_dataset_path = os.path.join(dataset_parent_folder, client_folder, "validation.dat")
        score_dataset_path = os.path.join(dataset_parent_folder, client_folder, "score.dat")  # when evaluating others' models, use all ur data

        model_checkpoints = os.path.join(run_folder, "model_checkpoints", client_folder)
        shutil.rmtree(model_checkpoints, ignore_errors=True)
        os.makedirs(model_checkpoints)

        training_dataset = LogisticRegressionDataset.load(training_dataset_path)
        if apply_dp:
            dp: DPBase = LaplaceOutputDP(inv_reg_term=inv_lambda, epsilon=epsilon)
        else:
            dp = DPBase()
        training_model_parameters = LogisticRegressionTrainingModelParams(
            training_dataset=training_dataset,
            evaluation_params=LogisticRegressionEvaluationModelParams(
                output_classes=output_classes,
                num_coefs=num_coefs,
                validation_dataset=LogisticRegressionDataset.load(validation_dataset_path),
                sklearn_lr_kwargs={"C": inv_lambda}
            ),
            max_num_epochs=20,
            checkpoint_folder=model_checkpoints,
            dp=dp)
        score_model_params = LogisticRegressionEvaluationModelParams(
            output_classes=output_classes,
            num_coefs=num_coefs,
            validation_dataset=LogisticRegressionDataset.load(score_dataset_path),
            sklearn_lr_kwargs={"C": inv_lambda}
        )
        model_factory=LogisticRegressionModelFactory(training_model_parameters, score_model_params)
        model_factories.append(model_factory)
    run_simulation(model_factories, contract_parameters, run_folder, ground_truth_model_factory)

def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", choices=('kdd99', 'adult'), required=True)
    parser.add_argument("--inv_lambda", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--apply_dp", action='store_true')
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--num_dp_rounds", type=int, default=3)
    parser.add_argument("--ground_truth_dataset", type=str, required=True)
    args = parser.parse_args()
    simulate_clients(experiment_name=args.exp_name,
                     dataset_parent_folder=args.dataset_folder,
                     apply_dp=args.apply_dp,
                     inv_lambda=args.inv_lambda,
                     epsilon=args.epsilon,
                     num_dp_rounds=args.num_dp_rounds,
                     ground_truth_dataset_path=args.ground_truth_dataset)


if __name__ == "__main__":
    _main()
